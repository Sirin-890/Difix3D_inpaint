import os
import gc
import lpips
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
import transformers
from torchvision.transforms.functional import crop
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from glob import glob
from einops import rearrange

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

import wandb

# ── Import both models ────────────────────────────────────────────────────────
from model import Difix, load_ckpt_from_state_dict, save_ckpt
from model_inpaint import DifixInpaint   # the inpaint model from previous step
from dataset2 import PairedDataset        # see note below — dataset must supply masks
from loss import gram_loss


# ─────────────────────────────────────────────────────────────────────────────
# Distillation loss
#   teacher_out : (BV, 3, H, W)  — frozen Difix output on the full image
#   student_out : (BV, 3, H, W)  — DifixInpaint output
#   mask_pix    : (BV, 1, H, W)  — 1 = inpaint region  (pixel resolution)
#
# We compute L2 between student and teacher ONLY inside the mask.
# This lets the teacher act as a "pseudo ground-truth" generator so
# you don't need large labelled inpaint datasets.
# ─────────────────────────────────────────────────────────────────────────────
def distillation_loss(student_out, teacher_out, mask_pix):
    diff = (student_out.float() - teacher_out.float()) ** 2   # (BV,3,H,W)
    masked = diff * mask_pix                                   # zero outside mask
    denom  = mask_pix.sum().clamp(min=1) * 3                  # num active pixels×channels
    return masked.sum() / denom


def masked_l2(pred, target, mask_pix):
    """MSE restricted to the inpainted region."""
    diff   = (pred.float() - target.float()) ** 2
    masked = diff * mask_pix
    denom  = mask_pix.sum().clamp(min=1) * 3
    return masked.sum() / denom


def masked_lpips(net_lpips, pred, target, mask_pix):
    """
    LPIPS is a patch metric — we blank out non-mask pixels so the
    network only 'sees' the inpainted region.
    """
    pred_m   = pred   * mask_pix
    target_m = target * mask_pix
    return net_lpips(pred_m.float(), target_m.float()).mean()


# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    # ── Student: inpaint model (trained) ─────────────────────────────────────
    net_student = DifixInpaint(
        lora_rank_vae=args.lora_rank_vae,
        timestep=args.timestep,
        mv_unet=args.mv_unet,
    )
    net_student.set_train()

    # ── Teacher: original Difix, frozen ──────────────────────────────────────
    # Loaded from your existing pretrained checkpoint.
    # It never receives gradients — pure inference.
    net_teacher = None
    if args.teacher_ckpt is not None:
        print("=" * 50)
        print(f"Loading teacher from {args.teacher_ckpt}")
        print("=" * 50)
        net_teacher = Difix(
            pretrained_path=args.teacher_ckpt,
            lora_rank_vae=args.lora_rank_vae,
            timestep=args.timestep,
            mv_unet=args.mv_unet,
        )
        net_teacher.set_eval()          # freeze everything
        net_teacher.requires_grad_(False)
        net_teacher.to(accelerator.device)
    else:
        print("No teacher checkpoint provided — running without distillation loss.")

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_student.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers not available. Install with `pip install xformers`.")

    if args.gradient_checkpointing:
        net_student.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    net_vgg = torchvision.models.vgg16(pretrained=True).features
    for param in net_vgg.parameters():
        param.requires_grad_(False)

    # ── Optimizer: student parameters only ───────────────────────────────────
    layers_to_opt = list(net_student.unet.parameters())
    for n, _p in net_student.vae.named_parameters():
        if "lora" in n and "vae_skip" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt += (
        list(net_student.vae.decoder.skip_conv_1.parameters()) +
        list(net_student.vae.decoder.skip_conv_2.parameters()) +
        list(net_student.vae.decoder.skip_conv_3.parameters()) +
        list(net_student.vae.decoder.skip_conv_4.parameters())
    )

    optimizer = torch.optim.AdamW(
        layers_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    # PairedDataset must now return a "mask_pixel_values" key:
    #   shape (V, 1, H, W), float32, values in {0.0, 1.0}
    #   1 = region to inpaint, 0 = known region
    # If your dataset doesn't have real masks you can generate random
    # rectangular / blob masks inside the dataset __getitem__.
    dataset_train = PairedDataset(
        dataset_path=args.dataset_path, split="train",
        tokenizer=net_student.tokenizer,
    )
    dl_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.train_batch_size,
        shuffle=True, num_workers=args.dataloader_num_workers,
    )
    dataset_val = PairedDataset(
        dataset_path=args.dataset_path, split="test",
        tokenizer=net_student.tokenizer,
    )
    random.Random(42).shuffle(dataset_val.img_names)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    global_step = 0
    if args.resume is not None:
        if os.path.isdir(args.resume):
            ckpt_files = sorted(
                glob(os.path.join(args.resume, "*.pkl")),
                key=lambda x: int(x.split("/")[-1].replace("model_", "").replace(".pkl", ""))
            )
            assert len(ckpt_files) > 0
            print("=" * 50); print(f"Resuming from {ckpt_files[-1]}"); print("=" * 50)
            global_step = int(ckpt_files[-1].split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_student, optimizer = load_ckpt_from_state_dict(net_student, optimizer, ckpt_files[-1])
        elif args.resume.endswith(".pkl"):
            print("=" * 50); print(f"Resuming from {args.resume}"); print("=" * 50)
            global_step = int(args.resume.split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_student, optimizer = load_ckpt_from_state_dict(net_student, optimizer, args.resume)
        else:
            raise NotImplementedError(f"Invalid resume path: {args.resume}")
    else:
        # ── Warm-start student from teacher ──────────────────────────────────
        # Copy all weights that are shape-compatible.
        # conv_in (4→5 ch) will be skipped safely — keeps the zero-init mask channel.
        if net_teacher is not None and args.warmstart_from_teacher:
            print("=" * 50)
            print("Warm-starting student from teacher weights (shape-compatible layers).")
            print("=" * 50)
            teacher_sd = net_teacher.state_dict()
            student_sd = net_student.state_dict()
            copied, skipped = 0, 0
            for k, v in teacher_sd.items():
                if k in student_sd and student_sd[k].shape == v.shape:
                    student_sd[k] = v.clone()
                    copied += 1
                else:
                    skipped += 1
            net_student.load_state_dict(student_sd)
            print(f"  Copied {copied} tensors, skipped {skipped} (shape mismatch or new layers).")
        else:
            print("=" * 50); print("Training from scratch."); print("=" * 50)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    net_student.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    net_vgg.to(accelerator.device, dtype=weight_dtype)

    net_student, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        net_student, optimizer, dl_train, lr_scheduler
    )
    net_lpips, net_vgg = accelerator.prepare(net_lpips, net_vgg)

    t_vgg_renorm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if accelerator.is_main_process:
        init_kwargs = {"wandb": {"name": args.tracker_run_name, "dir": args.output_dir}}
        accelerator.init_trackers(
            args.tracker_project_name,
            config=dict(vars(args)),
            init_kwargs=init_kwargs,
        )

    progress_bar = tqdm(
        range(0, args.max_train_steps), initial=global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────────────────────
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            with accelerator.accumulate(net_student):
                x_src  = batch["conditioning_pixel_values"]   # (B, V, 3, H, W)
                x_tgt  = batch["output_pixel_values"]         # (B, V, 3, H, W)
                mask   = batch["mask_pixel_values"]           # (B, V, 1, H, W)  ← new
                B, V, C, H, W = x_src.shape

                # ── Student forward (inpaint) ─────────────────────────────
                x_tgt_pred = net_student(
                    x_src,
                    mask,
                    prompt_tokens=batch["input_ids"],
                )  # (B, V, 3, H, W)

                # ── Teacher forward (no grad, no mask) ───────────────────
                teacher_pred = None
                if net_teacher is not None:
                    with torch.no_grad():
                        teacher_pred = net_teacher(
                            x_src.float(),
                            prompt_tokens=batch["input_ids"].cuda(),
                        ).to(weight_dtype)   # (B, V, 3, H, W)

                # Flatten views for loss computation
                x_tgt      = rearrange(x_tgt,      "b v c h w -> (b v) c h w")
                x_tgt_pred = rearrange(x_tgt_pred, "b v c h w -> (b v) c h w")
                mask_flat  = rearrange(mask,        "b v c h w -> (b v) c h w")

                # ── Masked reconstruction losses ──────────────────────────
                loss_l2    = masked_l2(x_tgt_pred, x_tgt, mask_flat) * args.lambda_l2
                loss_lpips = masked_lpips(net_lpips, x_tgt_pred, x_tgt, mask_flat) * args.lambda_lpips
                loss       = loss_l2 + loss_lpips

                # ── Distillation loss ─────────────────────────────────────
                # Teach the student to match the pretrained Difix output
                # inside the masked region. This is the key trick for
                # limited-data settings: teacher generates free supervision.
                loss_distill = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
                if teacher_pred is not None and args.lambda_distill > 0:
                    teacher_flat = rearrange(teacher_pred, "b v c h w -> (b v) c h w")
                    loss_distill = distillation_loss(
                        x_tgt_pred, teacher_flat, mask_flat
                    ) * args.lambda_distill
                    loss = loss + loss_distill

                # ── Gram / style loss ─────────────────────────────────────
                loss_gram = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
                if args.lambda_gram > 0 and global_step > args.gram_loss_warmup_steps:
                    # Apply gram loss only on masked crop region
                    mask_bool     = (mask_flat[:, 0] > 0.5)          # (BV, H, W) bool
                    rows           = mask_bool.any(dim=-1)            # (BV, H)
                    cols           = mask_bool.any(dim=-2)            # (BV, W)
                    # Fallback to random crop if mask is degenerate
                    r_idx = rows[0].nonzero(as_tuple=False)
                    c_idx = cols[0].nonzero(as_tuple=False)
                    crop_h, crop_w = 400, 400
                    if len(r_idx) >= crop_h and len(c_idx) >= crop_w:
                        top  = r_idx[0].item()
                        left = c_idx[0].item()
                    else:
                        top  = random.randint(0, max(H - crop_h, 0))
                        left = random.randint(0, max(W - crop_w, 0))

                    x_pred_crop  = crop(t_vgg_renorm(x_tgt_pred * 0.5 + 0.5), top, left, crop_h, crop_w)
                    x_tgt_crop   = crop(t_vgg_renorm(x_tgt       * 0.5 + 0.5), top, left, crop_h, crop_w)
                    loss_gram    = gram_loss(
                        x_pred_crop.to(weight_dtype),
                        x_tgt_crop.to(weight_dtype),
                        net_vgg,
                    ) * args.lambda_gram
                    loss = loss + loss_gram

                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                x_tgt      = rearrange(x_tgt,      "(b v) c h w -> b v c h w", v=V)
                x_tgt_pred = rearrange(x_tgt_pred, "(b v) c h w -> b v c h w", v=V)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {
                        "loss_l2":       loss_l2.detach().item(),
                        "loss_lpips":    loss_lpips.detach().item(),
                        "loss_distill":  loss_distill.detach().item(),
                    }
                    if args.lambda_gram > 0:
                        logs["loss_gram"] = loss_gram.detach().item()
                    progress_bar.set_postfix(**logs)

                    # ── Visualise ─────────────────────────────────────────
                    if global_step % args.viz_freq == 1:
                        log_imgs = {}
                        x_src_b = batch["conditioning_pixel_values"]
                        for idx in range(B):
                            src_img  = rearrange(x_src_b,   "b v c h w -> b c (v h) w")[idx].float().detach().cpu()
                            tgt_img  = rearrange(x_tgt,     "b v c h w -> b c (v h) w")[idx].float().detach().cpu()
                            pred_img = rearrange(x_tgt_pred,"b v c h w -> b c (v h) w")[idx].float().detach().cpu()
                            mask_img = rearrange(mask,      "b v c h w -> b c (v h) w")[idx].float().detach().cpu()
                        log_imgs = {
                            "train/source":       [wandb.Image(rearrange(x_src_b,    "b v c h w -> b c (v h) w")[i].float().detach().cpu(), caption=f"idx={i}") for i in range(B)],
                            "train/target":       [wandb.Image(rearrange(x_tgt,      "b v c h w -> b c (v h) w")[i].float().detach().cpu(), caption=f"idx={i}") for i in range(B)],
                            "train/model_output": [wandb.Image(rearrange(x_tgt_pred, "b v c h w -> b c (v h) w")[i].float().detach().cpu(), caption=f"idx={i}") for i in range(B)],
                            "train/mask":         [wandb.Image(rearrange(mask,        "b v c h w -> b c (v h) w")[i].float().detach().cpu(), caption=f"idx={i}") for i in range(B)],
                        }
                        # Also log teacher output if available
                        if teacher_pred is not None:
                            log_imgs["train/teacher"] = [
                                wandb.Image(rearrange(teacher_pred, "b v c h w -> b c (v h) w")[i].float().detach().cpu(), caption=f"idx={i}")
                                for i in range(B)
                            ]
                        logs.update(log_imgs)

                    # ── Checkpoint ────────────────────────────────────────
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        save_ckpt(accelerator.unwrap_model(net_student), optimizer, outf)

                    # ── Validation ────────────────────────────────────────
                    if args.eval_freq > 0 and global_step % args.eval_freq == 1:
                        l_l2, l_lpips = [], []
                        log_dict = {"sample/source": [], "sample/target": [], "sample/model_output": []}
                        for vstep, batch_val in enumerate(dl_val):
                            if vstep >= args.num_samples_eval:
                                break
                            x_src_v  = batch_val["conditioning_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                            x_tgt_v  = batch_val["output_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                            mask_v   = batch_val["mask_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                            with torch.no_grad():
                                x_pred_v = accelerator.unwrap_model(net_student)(
                                    x_src_v, mask_v,
                                    prompt_tokens=batch_val["input_ids"].cuda(),
                                )
                            if vstep % 10 == 0:
                                log_dict["sample/source"].append(wandb.Image(rearrange(x_src_v, "b v c h w -> b c (v h) w")[0].float().detach().cpu()))
                                log_dict["sample/target"].append(wandb.Image(rearrange(x_tgt_v, "b v c h w -> b c (v h) w")[0].float().detach().cpu()))
                                log_dict["sample/model_output"].append(wandb.Image(rearrange(x_pred_v, "b v c h w -> b c (v h) w")[0].float().detach().cpu()))

                            # Eval only on the inpainted view
                            x_tgt_v0  = x_tgt_v[:, 0]
                            x_pred_v0 = x_pred_v[:, 0]
                            mask_v0   = mask_v[:, 0]
                            l_l2.append(masked_l2(x_pred_v0, x_tgt_v0, mask_v0).item())
                            l_lpips.append(masked_lpips(net_lpips, x_pred_v0, x_tgt_v0, mask_v0).item())

                        logs["val/l2"]    = np.mean(l_l2)
                        logs["val/lpips"] = np.mean(l_lpips)
                        logs.update(log_dict)
                        gc.collect()
                        torch.cuda.empty_cache()

                    accelerator.log(logs, step=global_step)

        if global_step >= args.max_train_steps:
            break


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # loss weights
    parser.add_argument("--lambda_lpips",    default=1.0,  type=float)
    parser.add_argument("--lambda_l2",       default=1.0,  type=float)
    parser.add_argument("--lambda_gram",     default=1.0,  type=float)
    parser.add_argument("--lambda_distill",  default=2.0,  type=float,
        help="Weight of teacher→student distillation loss. Set 0 to disable.")
    parser.add_argument("--gram_loss_warmup_steps", default=2000, type=int)

    # dataset
    parser.add_argument("--dataset_path",    required=True, type=str)
    parser.add_argument("--train_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--test_image_prep",  default="resized_crop_512", type=str)
    parser.add_argument("--prompt",           default=None, type=str)

    # teacher / distillation
    parser.add_argument("--teacher_ckpt",         default=None, type=str,
        help="Path to pretrained Difix .pkl checkpoint used as distillation teacher.")
    parser.add_argument("--warmstart_from_teacher", action="store_true",
        help="Copy shape-compatible teacher weights into student before training.")

    # validation
    parser.add_argument("--eval_freq",        default=100, type=int)
    parser.add_argument("--num_samples_eval", default=100, type=int)
    parser.add_argument("--viz_freq",         default=100, type=int)
    parser.add_argument("--tracker_project_name", default="difix-inpaint", type=str)
    parser.add_argument("--tracker_run_name",     required=True, type=str)

    # model
    parser.add_argument("--pretrained_model_name_or_path")
    parser.add_argument("--revision",   default=None, type=str)
    parser.add_argument("--variant",    default=None, type=str)
    parser.add_argument("--tokenizer_name", default=None, type=str)
    parser.add_argument("--lora_rank_vae",  default=4,   type=int)
    parser.add_argument("--timestep",       default=199, type=int)
    parser.add_argument("--mv_unet",        action="store_true")

    # training
    parser.add_argument("--output_dir",         required=True)
    parser.add_argument("--seed",               default=None, type=int)
    parser.add_argument("--resolution",         default=512,  type=int)
    parser.add_argument("--train_batch_size",   default=4,    type=int)
    parser.add_argument("--num_training_epochs",default=10,   type=int)
    parser.add_argument("--max_train_steps",    default=10_000, type=int)
    parser.add_argument("--checkpointing_steps",default=500,  type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate",      default=5e-6, type=float)
    parser.add_argument("--lr_scheduler",       default="constant", type=str)
    parser.add_argument("--lr_warmup_steps",    default=500,  type=int)
    parser.add_argument("--lr_num_cycles",      default=1,    type=int)
    parser.add_argument("--lr_power",           default=1.0,  type=float)
    parser.add_argument("--dataloader_num_workers", default=0, type=int)
    parser.add_argument("--adam_beta1",         default=0.9,  type=float)
    parser.add_argument("--adam_beta2",         default=0.999,type=float)
    parser.add_argument("--adam_weight_decay",  default=1e-2, type=float)
    parser.add_argument("--adam_epsilon",        default=1e-8, type=float)
    parser.add_argument("--max_grad_norm",      default=1.0,  type=float)
    parser.add_argument("--allow_tf32",         action="store_true")
    parser.add_argument("--report_to",          default="wandb", type=str)
    parser.add_argument("--mixed_precision",    default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--set_grads_to_none",  action="store_true")
    parser.add_argument("--resume",             default=None, type=str)
    parser.add_argument("--cache_dir",          default=None)

    args = parser.parse_args()
    main(args)






    python  src/train4.py \
  --dataset_path ./dataset_synthetic \
  --output_dir ./output_inpaint \
  --tracker_run_name inpaint_v1 \
  --prompt "unoccluded image"