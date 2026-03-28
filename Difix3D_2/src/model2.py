import os
import requests
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from peft import LoraConfig
p = "src/"
sys.path.append(p)
from einops import rearrange, repeat


def make_1step_sched():
    noise_scheduler_1step = DDPMScheduler.from_pretrained("stabilityai/sd-turbo", subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step


def my_vae_encoder_fwd(self, sample):
    sample = self.conv_in(sample)
    l_blocks = []
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample


def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


def download_url(url, outf):
    if not os.path.exists(outf):
        print(f"Downloading checkpoint to {outf}")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(outf, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
        print(f"Downloaded successfully to {outf}")
    else:
        print(f"Skipping download, {outf} already exists")


class DifixInpaint(torch.nn.Module):
    def __init__(
        self,
        pretrained_name=None,
        pretrained_path=None,
        ckpt_folder="checkpoints",
        lora_rank_vae=4,
        mv_unet=False,
        timestep=999,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()
        self.sched = make_1step_sched()

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.ignore_skip = False

        if mv_unet:
            from mv_unet import UNet2DConditionModel
        else:
            from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")

        # ── Inpainting: expand conv_in from 4 → 5 channels (4 latent + 1 mask) ──
        # We copy the existing 4-channel weights and zero-init the new mask channel
        # so the model starts behaving identically to the original on unmasked regions.
        old_conv = unet.conv_in
        new_conv = torch.nn.Conv2d(
            in_channels=5,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight[:, :4, :, :] = old_conv.weight          # copy latent channels
            new_conv.weight[:, 4:, :, :] = 0.0                      # zero-init mask channel
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        unet.conv_in = new_conv
        unet.config["in_channels"] = 5  # keep config consistent

        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            vae_lora_config = LoraConfig(
                r=sd["rank_vae"],
                init_lora_weights="gaussian",
                target_modules=sd["vae_lora_target_modules"],
            )
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                # conv_in shape may differ from checkpoint – skip if mismatch
                if k in _sd_unet and _sd_unet[k].shape == sd["state_dict_unet"][k].shape:
                    _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_name is None and pretrained_path is None:
            print("Initializing model with random weights")
            torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
            target_modules_vae = [
                "conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                "to_k", "to_q", "to_v", "to_out.0",
            ]
            target_modules = []
            for id, (name, param) in enumerate(vae.named_modules()):
                if "decoder" in name and any(name.endswith(x) for x in target_modules_vae):
                    target_modules.append(name)
            target_modules_vae = target_modules
            vae.encoder.requires_grad_(False)
            vae_lora_config = LoraConfig(
                r=lora_rank_vae,
                init_lora_weights="gaussian",
                target_modules=target_modules_vae,
            )
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            self.lora_rank_vae = lora_rank_vae
            self.target_modules_vae = target_modules_vae

        unet.to("cuda")
        vae.to("cuda")
        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([timestep], device="cuda").long()
        self.text_encoder.requires_grad_(False)

        print("=" * 50)
        print(f"UNet trainable params: {sum(p.numel() for p in unet.parameters() if p.requires_grad) / 1e6:.2f}M")
        print(f"VAE  trainable params: {sum(p.numel() for p in vae.parameters() if p.requires_grad) / 1e6:.2f}M")
        print("=" * 50)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _encode_prompt(self, prompt, prompt_tokens):
        if prompt is not None:
            tokens = self.tokenizer(
                prompt,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.cuda()
            return self.text_encoder(tokens)[0]
        return self.text_encoder(prompt_tokens)[0]

    @staticmethod
    def _mask_to_latent(mask_pixel: torch.Tensor, latent_h: int, latent_w: int) -> torch.Tensor:
        """
        Downsample a pixel-space mask  (B, 1, H, W)  →  latent-space  (B, 1, lH, lW).
        Values are kept in [0, 1]; nearest interpolation preserves hard edges.
        """
        return F.interpolate(mask_pixel, size=(latent_h, latent_w), mode="nearest")

    # ──────────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,           # (B, V, 3, H, W)  values in [-1, 1]
        mask: torch.Tensor,        # (B, V, 1, H, W)  values in {0, 1}  — 1 = inpaint region
        timesteps=None,
        prompt=None,
        prompt_tokens=None,
    ):
        assert (prompt is None) != (prompt_tokens is None), \
            "Provide exactly one of prompt or prompt_tokens"

        caption_enc = self._encode_prompt(prompt, prompt_tokens)

        num_views = x.shape[1]
        x_flat    = rearrange(x,    "b v c h w -> (b v) c h w")
        mask_flat = rearrange(mask, "b v c h w -> (b v) c h w")

        # Encode to latent space
        z = self.vae.encode(x_flat).latent_dist.sample() * self.vae.config.scaling_factor
        # z: (BV, 4, lH, lW)

        # Downsample mask to latent resolution
        _, _, lH, lW = z.shape
        mask_lat = self._mask_to_latent(mask_flat, lH, lW)  # (BV, 1, lH, lW)

        # Concatenate latent + mask as UNet input  (5 channels total)
        unet_input = torch.cat([z, mask_lat], dim=1)        # (BV, 5, lH, lW)

        caption_enc = repeat(caption_enc, "b n c -> (b v) n c", v=num_views)

        t = self.timesteps if timesteps is None else timesteps
        model_pred = self.unet(unet_input, t, encoder_hidden_states=caption_enc).sample

        z_denoised = self.sched.step(model_pred, t, z, return_dict=True).prev_sample

        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        x_pred = self.vae.decode(z_denoised / self.vae.config.scaling_factor).sample.clamp(-1, 1)

        # ── Pixel-space compositing ──────────────────────────────────────────
        # Upsample latent mask back to pixel resolution for clean blending
        mask_pix = F.interpolate(mask_flat, size=x_flat.shape[-2:], mode="nearest")
        output_flat = x_flat * (1.0 - mask_pix) + x_pred * mask_pix

        output = rearrange(output_flat, "(b v) c h w -> b v c h w", v=num_views)
        return output

    # ──────────────────────────────────────────────────────────────────────────
    # Inference helper
    # ──────────────────────────────────────────────────────────────────────────

    def sample(
        self,
        image: Image.Image,
        mask: Image.Image,          # PIL grayscale – white (255) = inpaint region
        width: int,
        height: int,
        ref_image: Image.Image = None,
        ref_mask: Image.Image = None,
        timesteps=None,
        prompt=None,
        prompt_tokens=None,
    ):
        input_width, input_height = image.size
        new_width  = image.width  - image.width  % 8
        new_height = image.height - image.height % 8
        image = image.resize((new_width, new_height), Image.LANCZOS)
        mask  = mask.resize((new_width, new_height), Image.NEAREST)

        to_tensor = transforms.Compose([
            transforms.Resize((height, width), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        mask_to_tensor = transforms.Compose([
            transforms.Resize((height, width), interpolation=Image.NEAREST),
            transforms.ToTensor(),   # converts [0,255] → [0,1]
        ])

        img_t  = to_tensor(image)                   # (3, H, W)
        mask_t = (mask_to_tensor(mask.convert("L")) > 0.5).float()  # (1, H, W) binary

        if ref_image is None:
            # Single-view: shape (1, 1, C, H, W)
            x    = img_t.unsqueeze(0).unsqueeze(0).cuda()
            m    = mask_t.unsqueeze(0).unsqueeze(0).cuda()
        else:
            ref_image = ref_image.resize((new_width, new_height), Image.LANCZOS)
            ref_mask  = ref_mask.resize((new_width, new_height), Image.NEAREST) \
                        if ref_mask is not None \
                        else Image.new("L", (new_width, new_height), 0)
            ref_mask_t = (mask_to_tensor(ref_mask.convert("L")) > 0.5).float()
            ref_img_t  = to_tensor(ref_image)
            x = torch.stack([img_t, ref_img_t], dim=0).unsqueeze(0).cuda()  # (1,2,3,H,W)
            m = torch.stack([mask_t, ref_mask_t], dim=0).unsqueeze(0).cuda()

        output = self.forward(x, m, timesteps, prompt, prompt_tokens)[:, 0]  # first view
        output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
        output_pil = output_pil.resize((input_width, input_height), Image.LANCZOS)
        return output_pil

    # ──────────────────────────────────────────────────────────────────────────
    # Train / eval mode
    # ──────────────────────────────────────────────────────────────────────────

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.unet.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

    # ──────────────────────────────────────────────────────────────────────────
    # Checkpoint helpers
    # ──────────────────────────────────────────────────────────────────────────

    def save_model(self, outf, optimizer):
        sd = {
            "vae_lora_target_modules": self.target_modules_vae,
            "rank_vae": self.lora_rank_vae,
            "state_dict_unet": {
                k: v for k, v in self.unet.state_dict().items()
                if "lora" in k or "conv_in" in k   # conv_in now has 5 channels – save it
            },
            "state_dict_vae": {
                k: v for k, v in self.vae.state_dict().items()
                if "lora" in k or "skip" in k
            },
            "optimizer": optimizer.state_dict(),
        }
        torch.save(sd, outf)