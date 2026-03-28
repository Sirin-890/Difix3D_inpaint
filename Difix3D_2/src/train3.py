

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
from pipeline_difix import DifixPipeline

import wandb

from model import load_ckpt_from_state_dict, save_ckpt
from model2 import DifixInpaint
from dataset2 import PairedDataset
from loss import gram_loss


# ─────────────────────────────────────────────────────────────────────────────
# Teacher wrapper
# ─────────────────────────────────────────────────────────────────────────────
class DifixTeacher(torch.nn.Module):
    def __init__(self, pretrained_name="nvidia/difix_ref"):
        super().__init__()
        print("=" * 50)
        print(f"Loading teacher pipeline from {pretrained_name}")
        print("=" * 50)
        self.pipe = DifixPipeline.from_pretrained(
            pretrained_name, trust_remote_code=True
        )
        self.pipe.to("cuda")
        for p in self.pipe.parameters() if hasattr(self.pipe, "parameters") else []:
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, V, 3, H, W)  float32, values in [-1, 1]
        returns (B, V, 3, H, W) float32, values in [-1, 1]

        Memory optimisations vs original:
        - process all frames in one vectorised pass where possible
        - delete intermediates immediately
        - flush cache after each batch item
        """
        B, V, C, H, W = x.shape
        outputs = []

        for b in range(B):
            view_outputs = []
            for v in range(V):
                frame = x[b, v]                          # (3, H, W)  [-1, 1]
                frame_01 = (frame * 0.5 + 0.5).clamp(0, 1)
                pil_img = transforms.ToPILImage()(frame_01.cpu())
                del frame, frame_01                      # ← free before pipeline call

                result = self.pipe(image=pil_img)
                del pil_img

                if hasattr(result, "images"):
                    out_pil = result.images[0]
                    out_t = transforms.ToTensor()(out_pil).to(x.device)  # [0,1]
                    del out_pil
                elif isinstance(result, torch.Tensor):
                    out_t = result.squeeze(0).to(x.device)
                    if out_t.max() > 1.1:
                        out_t = out_t / 255.0
                else:
                    raise ValueError(f"Unexpected pipeline output type: {type(result)}")
                del result

                if out_t.shape[-2:] != (H, W):
                    out_t = F.interpolate(
                        out_t.unsqueeze(0), size=(H, W),
                        mode="bilinear", align_corners=False
                    ).squeeze(0)

                out_t = out_t * 2.0 - 1.0               # → [-1, 1]
                view_outputs.append(out_t)

            # stack views, free list
            stacked = torch.stack(view_outputs, dim=0)   # (V, 3, H, W)
            del view_outputs
            outputs.append(stacked)

            # flush after every batch item to reclaim pipeline intermediates
            torch.cuda.empty_cache()

        result_t = torch.stack(outputs, dim=0)           # (B, V, 3, H, W)
        del outputs
        return result_t


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────
def distillation_loss(student_out, teacher_out, mask_pix):
    diff   = (student_out.float() - teacher_out.float()) ** 2
    masked = diff * mask_pix
    denom  = mask_pix.sum().clamp(min=1) * 3
    return masked.sum() / denom


def masked_l2(pred, target, mask_pix):
    diff   = (pred.float() - target.float()) ** 2
    masked = diff * mask_pix
    denom  = mask_pix.sum().clamp(min=1) * 3
    return masked.sum() / denom


def masked_lpips(net_lpips, pred, target, mask_pix):
    pred_m   = pred   * mask_pix
    target_m = target * mask_pix
    return net_lpips(pred_m.float(), target_m.float()).mean()


# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    # ── set PYTORCH_ALLOC_CONF early to reduce fragmentation ─────────────────
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

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

    # ── Student ───────────────────────────────────────────────────────────────
    net_student = DifixInpaint(
        lora_rank_vae=args.lora_rank_vae,
        timestep=args.timestep,
        mv_unet=args.mv_unet,
    )
    net_student.set_train()

    # ── Teacher (always frozen, on CPU between steps to save VRAM) ────────────
    net_teacher = None
    if not args.no_teacher:
        net_teacher = DifixTeacher(pretrained_name=args.teacher_repo)
        net_teacher.eval()
        net_teacher.requires_grad_(False)
        # Keep teacher on CPU; move to CUDA only during its forward pass
        # to avoid occupying VRAM while the student is running.
        if args.teacher_on_cpu:
            net_teacher.pipe.to("cpu")
            print("Teacher pipeline parked on CPU between steps.")
    else:
        print("Teacher disabled — running without distillation loss.")

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_student.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers not available.")

    if args.gradient_checkpointing:
        net_student.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # ── Perceptual nets — use fp16 to halve their VRAM footprint ─────────────
    # LPIPS/VGG do not need full fp32 precision for a perceptual loss signal.
    _perc_dtype = torch.float16 if accelerator.mixed_precision != "no" else torch.float32
    net_lpips = lpips.LPIPS(net='vgg').cuda().to(_perc_dtype)
    net_lpips.requires_grad_(False)

    net_vgg = torchvision.models.vgg16(pretrained=True).features.cuda().to(_perc_dtype)
    for param in net_vgg.parameters():
        param.requires_grad_(False)

    # ── Optimizer ─────────────────────────────────────────────────────────────
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
    dataset_train = PairedDataset(
        dataset_path=args.dataset_path, split="train",
        tokenizer=net_student.tokenizer,
        prompt=args.prompt or "",
    )
    dl_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.train_batch_size,
        shuffle=True, num_workers=args.dataloader_num_workers,
        pin_memory=True,                                 # ← faster CPU→GPU transfers
    )
    dataset_val = PairedDataset(
        dataset_path=args.dataset_path, split="test",
        tokenizer=net_student.tokenizer,
        prompt=args.prompt or "",
    )
    random.Random(42).shuffle(dataset_val.img_names)
    dl_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=0
    )

    # ── Resume / warm-start ───────────────────────────────────────────────────
    global_step = 0
    if args.resume is not None:
        if os.path.isdir(args.resume):
            ckpt_files = sorted(
                glob(os.path.join(args.resume, "*.pkl")),
                key=lambda x: int(x.split("/")[-1].replace("model_", "").replace(".pkl", ""))
            )
            assert len(ckpt_files) > 0
            print(f"Resuming from {ckpt_files[-1]}")
            global_step = int(ckpt_files[-1].split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_student, optimizer = load_ckpt_from_state_dict(net_student, optimizer, ckpt_files[-1])
        elif args.resume.endswith(".pkl"):
            print(f"Resuming from {args.resume}")
            global_step = int(args.resume.split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_student, optimizer = load_ckpt_from_state_dict(net_student, optimizer, args.resume)
        else:
            raise NotImplementedError(f"Invalid resume path: {args.resume}")

    elif args.warmstart_from_teacher and net_teacher is not None:
        print("=" * 50)
        print("Warm-starting student from teacher pipeline weights.")
        print("=" * 50)
        student_sd = net_student.state_dict()
        copied, skipped = 0, 0
        pipe = net_teacher.pipe
        # temporarily move teacher to GPU if it was parked on CPU
        if args.teacher_on_cpu:
            pipe.to("cuda")
        teacher_parts = {}
        if hasattr(pipe, "unet"):
            for k, v in pipe.unet.state_dict().items():
                teacher_parts[f"unet.{k}"] = v
        if hasattr(pipe, "vae"):
            for k, v in pipe.vae.state_dict().items():
                teacher_parts[f"vae.{k}"] = v
        for k, v in teacher_parts.items():
            if k in student_sd and student_sd[k].shape == v.shape:
                student_sd[k] = v.clone()
                copied += 1
            else:
                skipped += 1
        net_student.load_state_dict(student_sd)
        del teacher_parts, student_sd
        if args.teacher_on_cpu:
            pipe.to("cpu")
        torch.cuda.empty_cache()
        print(f"  Copied {copied} tensors, skipped {skipped}.")
    else:
        print("Training from scratch.")

    # ── dtype + device ────────────────────────────────────────────────────────
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    net_student.to(accelerator.device, dtype=weight_dtype)
    # net_lpips / net_vgg already cast above

    net_student, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        net_student, optimizer, dl_train, lr_scheduler
    )
    net_lpips, net_vgg = accelerator.prepare(net_lpips, net_vgg)

    t_vgg_renorm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if accelerator.is_main_process:
        accelerator.init_trackers(
            args.tracker_project_name,
            config=dict(vars(args)),
            init_kwargs={"wandb": {"name": args.tracker_run_name, "dir": args.output_dir}},
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
                x_src = batch["conditioning_pixel_values"]   # (B, V, 3, H, W)
                x_tgt = batch["output_pixel_values"]         # (B, V, 3, H, W)
                mask  = batch["mask_pixel_values"]           # (B, V, 1, H, W)
                B, V, C, H, W = x_src.shape

                # ── student forward ───────────────────────────────────────
                x_tgt_pred = net_student(
                    x_src, mask,
                    prompt_tokens=batch["input_ids"],
                )   # (B, V, 3, H, W)

                # ── teacher forward ───────────────────────────────────────
                # Move teacher to CUDA only for its forward pass, then back
                # to CPU so it doesn't hold VRAM during backward.
                teacher_pred = None
                if net_teacher is not None and args.lambda_distill > 0:
                    if args.teacher_on_cpu:
                        net_teacher.pipe.to("cuda")

                    with torch.no_grad():
                        teacher_pred = net_teacher(
                            x_src.float()           # teacher always in fp32
                        ).to(dtype=weight_dtype, device=accelerator.device)

                    if args.teacher_on_cpu:
                        net_teacher.pipe.to("cpu")
                        torch.cuda.empty_cache()    # reclaim teacher VRAM

                # ── flatten views ─────────────────────────────────────────
                x_tgt_flat      = rearrange(x_tgt,      "b v c h w -> (b v) c h w")
                x_tgt_pred_flat = rearrange(x_tgt_pred, "b v c h w -> (b v) c h w")
                mask_flat       = rearrange(mask,        "b v c h w -> (b v) c h w")

                # ── losses ────────────────────────────────────────────────
                loss_l2    = masked_l2(x_tgt_pred_flat, x_tgt_flat, mask_flat) * args.lambda_l2
                loss_lpips = masked_lpips(net_lpips, x_tgt_pred_flat, x_tgt_flat, mask_flat) * args.lambda_lpips
                loss       = loss_l2 + loss_lpips

                loss_distill = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
                if teacher_pred is not None:
                    teacher_flat = rearrange(teacher_pred, "b v c h w -> (b v) c h w")
                    loss_distill = distillation_loss(
                        x_tgt_pred_flat, teacher_flat, mask_flat
                    ) * args.lambda_distill
                    loss = loss + loss_distill
                    del teacher_flat, teacher_pred   # ← free teacher tensor ASAP

                loss_gram = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
                if args.lambda_gram > 0 and global_step > args.gram_loss_warmup_steps:
                    mask_bool = (mask_flat[:, 0] > 0.5)
                    r_idx = mask_bool.any(dim=-1)[0].nonzero(as_tuple=False)
                    c_idx = mask_bool.any(dim=-2)[0].nonzero(as_tuple=False)
                    crop_h, crop_w = 400, 400
                    if len(r_idx) >= crop_h and len(c_idx) >= crop_w:
                        top, left = r_idx[0].item(), c_idx[0].item()
                    else:
                        top  = random.randint(0, max(H - crop_h, 0))
                        left = random.randint(0, max(W - crop_w, 0))
                    x_pred_crop = crop(t_vgg_renorm(x_tgt_pred_flat * 0.5 + 0.5), top, left, crop_h, crop_w)
                    x_tgt_crop  = crop(t_vgg_renorm(x_tgt_flat      * 0.5 + 0.5), top, left, crop_h, crop_w)
                    loss_gram   = gram_loss(
                        x_pred_crop.to(weight_dtype),
                        x_tgt_crop.to(weight_dtype),
                        net_vgg,
                    ) * args.lambda_gram
                    loss = loss + loss_gram
                    del x_pred_crop, x_tgt_crop

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)   # always set_to_none for VRAM

                # keep unflattened versions for logging only; free flat tensors
                del x_tgt_flat, mask_flat

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {
                        "loss_l2":      loss_l2.detach().item(),
                        "loss_lpips":   loss_lpips.detach().item(),
                        "loss_distill": loss_distill.detach().item(),
                    }
                    if args.lambda_gram > 0:
                        logs["loss_gram"] = loss_gram.detach().item()
                    progress_bar.set_postfix(**logs)

                    if global_step % args.viz_freq == 1:
                        x_src_b = batch["conditioning_pixel_values"]
                        # x_tgt_pred is still (B, V, C, H, W) here
                        log_imgs = {
                            "train/source":       [wandb.Image(rearrange(x_src_b,    "b v c h w -> b c (v h) w")[i].float().detach().cpu(), caption=f"idx={i}") for i in range(B)],
                            "train/target":       [wandb.Image(rearrange(x_tgt,      "b v c h w -> b c (v h) w")[i].float().detach().cpu(), caption=f"idx={i}") for i in range(B)],
                            "train/model_output": [wandb.Image(rearrange(x_tgt_pred, "b v c h w -> b c (v h) w")[i].float().detach().cpu(), caption=f"idx={i}") for i in range(B)],
                            "train/mask":         [wandb.Image(rearrange(mask,        "b v c h w -> b c (v h) w")[i].float().detach().cpu(), caption=f"idx={i}") for i in range(B)],
                        }
                        logs.update(log_imgs)

                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        save_ckpt(accelerator.unwrap_model(net_student), optimizer, outf)

                    if args.eval_freq > 0 and global_step % args.eval_freq == 1:
                        # move teacher to cuda for eval if needed
                        if net_teacher is not None and args.teacher_on_cpu:
                            net_teacher.pipe.to("cuda")

                        l_l2, l_lpips = [], []
                        log_dict = {"sample/source": [], "sample/target": [], "sample/model_output": []}
                        for vstep, batch_val in enumerate(dl_val):
                            if vstep >= args.num_samples_eval:
                                break
                            x_src_v = batch_val["conditioning_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                            x_tgt_v = batch_val["output_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                            mask_v  = batch_val["mask_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                            with torch.no_grad():
                                x_pred_v = accelerator.unwrap_model(net_student)(
                                    x_src_v, mask_v,
                                    prompt_tokens=batch_val["input_ids"].cuda(),
                                )
                            if vstep % 10 == 0:
                                log_dict["sample/source"].append(wandb.Image(rearrange(x_src_v, "b v c h w -> b c (v h) w")[0].float().detach().cpu()))
                                log_dict["sample/target"].append(wandb.Image(rearrange(x_tgt_v, "b v c h w -> b c (v h) w")[0].float().detach().cpu()))
                                log_dict["sample/model_output"].append(wandb.Image(rearrange(x_pred_v, "b v c h w -> b c (v h) w")[0].float().detach().cpu()))

                            x_tgt_v0  = x_tgt_v[:, 0]
                            x_pred_v0 = x_pred_v[:, 0]
                            mask_v0   = mask_v[:, 0]
                            l_l2.append(masked_l2(x_pred_v0, x_tgt_v0, mask_v0).item())
                            l_lpips.append(masked_lpips(net_lpips, x_pred_v0, x_tgt_v0, mask_v0).item())
                            del x_pred_v, x_src_v, x_tgt_v, mask_v   # ← free per val step

                        if net_teacher is not None and args.teacher_on_cpu:
                            net_teacher.pipe.to("cpu")
                            torch.cuda.empty_cache()

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
    parser.add_argument("--lambda_lpips",           default=1.0,  type=float)
    parser.add_argument("--lambda_l2",              default=1.0,  type=float)
    parser.add_argument("--lambda_gram",            default=1.0,  type=float)
    parser.add_argument("--lambda_distill",         default=2.0,  type=float)
    parser.add_argument("--gram_loss_warmup_steps", default=2000, type=int)

    # dataset
    parser.add_argument("--dataset_path",     required=True, type=str)
    parser.add_argument("--prompt",           default=None,  type=str)

    # teacher
    parser.add_argument("--teacher_repo",           default="nvidia/difix_ref", type=str)
    parser.add_argument("--no_teacher",             action="store_true")
    parser.add_argument("--warmstart_from_teacher", action="store_true")
    parser.add_argument("--teacher_on_cpu",         action="store_true",
        help="Park teacher pipeline on CPU between steps to save VRAM. "
             "Slower but avoids OOM when student + teacher don't fit together.")

    # validation
    parser.add_argument("--eval_freq",            default=100,          type=int)
    parser.add_argument("--num_samples_eval",     default=100,          type=int)
    parser.add_argument("--viz_freq",             default=100,          type=int)
    parser.add_argument("--tracker_project_name", default="difix-inpaint", type=str)
    parser.add_argument("--tracker_run_name",     required=True,        type=str)

    # model
    parser.add_argument("--lora_rank_vae", default=4,   type=int)
    parser.add_argument("--timestep",      default=199, type=int)
    parser.add_argument("--mv_unet",       action="store_true")

    # training
    parser.add_argument("--output_dir",                   required=True)
    parser.add_argument("--seed",                         default=None, type=int)
    parser.add_argument("--train_batch_size",             default=4,    type=int)
    parser.add_argument("--num_training_epochs",          default=10,   type=int)
    parser.add_argument("--max_train_steps",              default=10_000, type=int)
    parser.add_argument("--checkpointing_steps",          default=500,  type=int)
    parser.add_argument("--gradient_accumulation_steps",  default=1,    type=int)
    parser.add_argument("--gradient_checkpointing",       action="store_true")
    parser.add_argument("--learning_rate",                default=5e-6, type=float)
    parser.add_argument("--lr_scheduler",                 default="constant", type=str)
    parser.add_argument("--lr_warmup_steps",              default=500,  type=int)
    parser.add_argument("--lr_num_cycles",                default=1,    type=int)
    parser.add_argument("--lr_power",                     default=1.0,  type=float)
    parser.add_argument("--dataloader_num_workers",       default=0,    type=int)
    parser.add_argument("--adam_beta1",                   default=0.9,  type=float)
    parser.add_argument("--adam_beta2",                   default=0.999,type=float)
    parser.add_argument("--adam_weight_decay",            default=1e-2, type=float)
    parser.add_argument("--adam_epsilon",                 default=1e-8, type=float)
    parser.add_argument("--max_grad_norm",                default=1.0,  type=float)
    parser.add_argument("--allow_tf32",                   action="store_true")
    parser.add_argument("--report_to",                    default="wandb", type=str)
    parser.add_argument("--mixed_precision",              default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--set_grads_to_none",            action="store_true")
    parser.add_argument("--resume",                       default=None, type=str)

    args = parser.parse_args()
    main(args)