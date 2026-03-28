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
    # down
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    # middle
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample


def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    # middle
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        # up
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
            # add skip
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)
    # post-process
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
        block_size = 1024  # 1 Kibibyte
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


def load_ckpt_from_state_dict(net_difix, optimizer, pretrained_path):
    sd = torch.load(pretrained_path, map_location="cpu")

    if "state_dict_vae" in sd:
        _sd_vae = net_difix.vae.state_dict()
        for k in sd["state_dict_vae"]:
            _sd_vae[k] = sd["state_dict_vae"][k]
        net_difix.vae.load_state_dict(_sd_vae)
    _sd_unet = net_difix.unet.state_dict()
    for k in sd["state_dict_unet"]:
        _sd_unet[k] = sd["state_dict_unet"][k]
    net_difix.unet.load_state_dict(_sd_unet)

    optimizer.load_state_dict(sd["optimizer"])

    return net_difix, optimizer


def save_ckpt(net_difix, optimizer, outf):
    sd = {}
    sd["vae_lora_target_modules"] = net_difix.target_modules_vae
    sd["rank_vae"] = net_difix.lora_rank_vae
    sd["state_dict_unet"] = net_difix.unet.state_dict()
    sd["state_dict_vae"] = {k: v for k, v in net_difix.vae.state_dict().items() if "lora" in k or "skip" in k}
    sd["optimizer"] = optimizer.state_dict()
    torch.save(sd, outf)


def prepare_mask_latents(mask, image, vae, height, width):
    """
    Encodes the masked image and downsamples the mask to latent resolution.

    Args:
        mask  : (B, 1, H, W) float tensor, 1 = region to inpaint, 0 = keep
        image : (B, C, H, W) float tensor normalised to [-1, 1]
        vae   : AutoencoderKL
        height, width: latent spatial dims (H//8, W//8)

    Returns:
        mask_latent        : (B, 1, H//8, W//8)  — downsampled binary mask
        masked_image_latent: (B, 4, H//8, W//8)  — VAE encoding of image * (1-mask)
    """
    # masked image: keep original pixels outside the mask region
    masked_image = image * (1 - mask)

    # encode masked image
    masked_image_latent = vae.encode(masked_image).latent_dist.sample()
    masked_image_latent = masked_image_latent * vae.config.scaling_factor

    # downsample mask to latent resolution
    mask_latent = F.interpolate(mask, size=(height // 8, width // 8), mode="nearest")

    return mask_latent, masked_image_latent


class DifixInpaint(torch.nn.Module):
    """
    Inpainting variant of Difix.

    The UNet now receives a 9-channel input:
        [noisy latent (4) | downsampled mask (1) | masked-image latent (4)]

    The mask convention: 1 = pixel to be inpainted, 0 = pixel to preserve.
    """

    def __init__(self, pretrained_name=None, pretrained_path=None, ckpt_folder="checkpoints",
                 lora_rank_vae=4, mv_unet=False, timestep=999):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()
        self.sched = make_1step_sched()

        # ── VAE ──────────────────────────────────────────────────────────────
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.ignore_skip = False

        # ── UNet ─────────────────────────────────────────────────────────────
        if mv_unet:
            from mv_unet import UNet2DConditionModel
        else:
            from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")

        # ── Expand conv_in from 4 → 9 channels for inpainting ───────────────
        # Standard SD-inpainting trick: copy pretrained weights for first 4 ch,
        # zero-init the extra 5 channels so training starts from a stable point.
        self._expand_unet_conv_in(unet, in_channels=9)

        # ── Load checkpoint if provided ──────────────────────────────────────
        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian",
                                         target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_name is None and pretrained_path is None:
            print("Initializing model with random weights")
            torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)

            target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                                   "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                                   "to_k", "to_q", "to_v", "to_out.0"]
            target_modules = []
            for id, (name, param) in enumerate(vae.named_modules()):
                if 'decoder' in name and any(name.endswith(x) for x in target_modules_vae):
                    target_modules.append(name)
            target_modules_vae = target_modules
            vae.encoder.requires_grad_(False)

            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                                         target_modules=target_modules_vae)
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
        print(f"Number of trainable parameters in UNet: {sum(p.numel() for p in unet.parameters() if p.requires_grad) / 1e6:.2f}M")
        print(f"Number of trainable parameters in VAE:  {sum(p.numel() for p in vae.parameters() if p.requires_grad) / 1e6:.2f}M")
        print("=" * 50)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _expand_unet_conv_in(unet, in_channels: int = 9):
        """
        Replace the UNet's first conv (4-ch → in_channels) with zero-padded weights.
        Preserves pretrained weights for the original 4 channels.
        """
        old_conv = unet.conv_in
        old_weight = old_conv.weight.data          # (out_ch, 4, kH, kW)
        old_bias   = old_conv.bias.data if old_conv.bias is not None else None

        new_conv = torch.nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_bias is not None),
        )
        # zero-init all weights, then copy the original 4-channel slice
        torch.nn.init.zeros_(new_conv.weight)
        new_conv.weight.data[:, :old_conv.in_channels] = old_weight
        if old_bias is not None:
            new_conv.bias.data = old_bias

        unet.conv_in = new_conv
        unet.config["in_channels"] = in_channels  # keep config in sync

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

    # ─────────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, x, mask, timesteps=None, prompt=None, prompt_tokens=None):
        """
        Args:
            x      : (B, V, C, H, W)  input images, values in [-1, 1]
            mask   : (B, V, 1, H, W)  inpainting mask, 1 = inpaint, 0 = keep
            timesteps     : optional override
            prompt        : text string (mutually exclusive with prompt_tokens)
            prompt_tokens : pre-tokenised ids (mutually exclusive with prompt)
        """
        assert (prompt is None) != (prompt_tokens is None), \
            "Provide exactly one of prompt or prompt_tokens"

        # ── Text encoding ────────────────────────────────────────────────────
        if prompt is not None:
            caption_tokens = self.tokenizer(
                prompt, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]

        num_views = x.shape[1]

        # flatten views into batch dimension
        x    = rearrange(x,    'b v c h w -> (b v) c h w')
        mask = rearrange(mask, 'b v 1 h w -> (b v) 1 h w')

        # ── VAE encode full image ────────────────────────────────────────────
        z = self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor

        # ── Prepare mask conditioning ────────────────────────────────────────
        _, _, H, W = x.shape
        mask_latent, masked_image_latent = prepare_mask_latents(
            mask, x, self.vae, H, W
        )

        # ── Build 9-channel UNet input ───────────────────────────────────────
        # [noisy_latent(4) | mask(1) | masked_image_latent(4)]
        unet_input = torch.cat([z, mask_latent, masked_image_latent], dim=1)

        caption_enc = repeat(caption_enc, 'b n c -> (b v) n c', v=num_views)

        # ── Denoising step ───────────────────────────────────────────────────
        model_pred = self.unet(
            unet_input, self.timesteps,
            encoder_hidden_states=caption_enc,
        ).sample

        z_denoised = self.sched.step(
            model_pred, self.timesteps, z, return_dict=True
        ).prev_sample

        # ── Decode ───────────────────────────────────────────────────────────
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        output_image = (
            self.vae.decode(z_denoised / self.vae.config.scaling_factor).sample
        ).clamp(-1, 1)

        # ── Composite: paste original pixels outside the mask ────────────────
        # Upsample mask back to image resolution for compositing
        mask_up = F.interpolate(mask, size=(H, W), mode="nearest")
        output_image = output_image * mask_up + x * (1 - mask_up)

        output_image = rearrange(output_image, '(b v) c h w -> b v c h w', v=num_views)
        return output_image

    # ─────────────────────────────────────────────────────────────────────────
    # Inference helper
    # ─────────────────────────────────────────────────────────────────────────

    def sample(self, image, mask, width, height,
               ref_image=None, ref_mask=None,
               timesteps=None, prompt=None, prompt_tokens=None):
        """
        High-level inference entry point.

        Args:
            image    : PIL.Image — input image
            mask     : PIL.Image (grayscale) or np.ndarray — inpainting mask
                       White (255) = inpaint, Black (0) = keep
            width, height : target latent resolution (multiples of 8)
            ref_image : optional second-view PIL.Image
            ref_mask  : optional second-view mask (required when ref_image given)
            prompt / prompt_tokens : text conditioning

        Returns:
            PIL.Image of the same size as the input
        """
        input_width, input_height = image.size
        new_width  = image.width  - image.width  % 8
        new_height = image.height - image.height % 8
        image = image.resize((new_width, new_height), Image.LANCZOS)

        T_img = transforms.Compose([
            transforms.Resize((height, width), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        T_mask = transforms.Compose([
            transforms.Resize((height, width), interpolation=Image.NEAREST),
            transforms.ToTensor(),          # [0,1] float, shape (1, H, W)
        ])

        # ── Convert mask input to PIL grayscale ──────────────────────────────
        def _to_mask_tensor(m):
            if isinstance(m, np.ndarray):
                m = Image.fromarray(m)
            if m.mode != "L":
                m = m.convert("L")
            m = m.resize((new_width, new_height), Image.NEAREST)
            return T_mask(m)   # values in [0, 1]

        img_t  = T_img(image)
        mask_t = _to_mask_tensor(mask)

        if ref_image is None:
            # single-view: shape (1, 1, C, H, W) and (1, 1, 1, H, W)
            x      = img_t.unsqueeze(0).unsqueeze(0).cuda()
            mask_b = mask_t.unsqueeze(0).unsqueeze(0).cuda()
        else:
            assert ref_mask is not None, "ref_mask must be provided together with ref_image"
            ref_image = ref_image.resize((new_width, new_height), Image.LANCZOS)
            ref_mask_t = _to_mask_tensor(ref_mask)
            x      = torch.stack([img_t, T_img(ref_image)], dim=0).unsqueeze(0).cuda()
            mask_b = torch.stack([mask_t, ref_mask_t],        dim=0).unsqueeze(0).cuda()

        with torch.no_grad():
            output_image = self.forward(x, mask_b, timesteps, prompt, prompt_tokens)[:, 0]

        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
        output_pil = output_pil.resize((input_width, input_height), Image.LANCZOS)
        return output_pil

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpoint I/O
    # ─────────────────────────────────────────────────────────────────────────

    def save_model(self, outf, optimizer):
        sd = {}
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_vae"] = self.lora_rank_vae
        # save full conv_in weights so the 9-ch expansion is preserved
        sd["state_dict_unet"] = {
            k: v for k, v in self.unet.state_dict().items()
            if "lora" in k or "conv_in" in k
        }
        sd["state_dict_vae"] = {
            k: v for k, v in self.vae.state_dict().items()
            if "lora" in k or "skip" in k
        }
        sd["optimizer"] = optimizer.state_dict()
        torch.save(sd, outf)