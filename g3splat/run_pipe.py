
import torch
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

from src.config import load_typed_root_config
from src.dataset.data_module import get_data_shim
from src.model.decoder import get_decoder
from src.model.encoder import get_encoder

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import cv2

DEVICE = "cuda"
IMG_SIZE = 256

# -------- PATHS --------
DATA_ROOT = "data"
OUTPUT_DIR = "outputs"
CHECKPOINT = "pretrained_weights/g3splat_mast3r_3dgs_align_orient_re10k.ckpt"
# ----------------------

# -------- intrinsics --------
def get_K():
    fx, fy, cx, cy = 0.86, 0.86, 0.5, 0.5
    K = torch.tensor([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32).to(DEVICE)
    return K

# -------- load model --------
def load_model():
    from pathlib import Path

    GlobalHydra.instance().clear()

    config_dir = str(Path(__file__).parent / "config")

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="main", overrides=[
            "+experiment=re10k_align_orient",
        ])

    cfg = load_typed_root_config(cfg)

    encoder, _ = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder)

    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    encoder.load_state_dict(
        {k[8:]: v for k, v in state_dict.items() if k.startswith("encoder.")},
        strict=False
    )

    encoder = encoder.to(DEVICE).eval()
    decoder = decoder.to(DEVICE)

    data_shim = get_data_shim(encoder)

    return encoder, decoder, data_shim

# -------- load image --------
def load_img(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    t = torch.from_numpy(np.array(img)).float() / 255.0
    return t.permute(2, 0, 1).to(DEVICE)

# -------- pose --------
def estimate_pose(pts3d, opacity, K, H, W):
    pts3d = pts3d.reshape(-1, 3).cpu().numpy()
    opacity = opacity.reshape(-1).cpu().numpy()

    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    pixels = np.stack([xs, ys], axis=-1).reshape(-1, 2)

    K = K.cpu().numpy().copy()
    K[0, :] *= W
    K[1, :] *= H

    mask = opacity > 0.1
    if mask.sum() < 10:
        mask = np.ones_like(opacity, dtype=bool)

    pts3d = pts3d[mask]
    pixels = pixels[mask]

    if len(pts3d) < 6:
        pose = np.eye(4)
        pose[0, 3] = 0.1
        return torch.tensor(pose, dtype=torch.float32).to(DEVICE)

    try:
        success, rvec, tvec, _ = cv2.solvePnPRansac(
            pts3d.astype(np.float32),
            pixels.astype(np.float32),
            K.astype(np.float32),
            None
        )

        if not success:
            raise RuntimeError

        R, _ = cv2.Rodrigues(rvec)

        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = tvec.flatten()

        c2w = np.linalg.inv(w2c)
        return torch.tensor(c2w, dtype=torch.float32).to(DEVICE)

    except:
        pose = np.eye(4)
        pose[0, 3] = 0.1
        return torch.tensor(pose, dtype=torch.float32).to(DEVICE)

# -------- MAIN LOOP --------
@torch.no_grad()
def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    encoder, decoder, data_shim = load_model()

    folders = sorted(os.listdir(DATA_ROOT))

    for folder in tqdm(folders):
        folder_path = os.path.join(DATA_ROOT, folder)

        if not os.path.isdir(folder_path):
            continue

        img1_path = os.path.join(folder_path, "0.png")
        img2_path = os.path.join(folder_path, "3.png")

        if not (os.path.exists(img1_path) and os.path.exists(img2_path)):
            print(f"Skipping {folder}")
            continue

        try:
            img1 = load_img(img1_path)
            img2 = load_img(img2_path)

            images = torch.stack([img1, img2]).unsqueeze(0)

            K = get_K().unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1)

            extrinsics = torch.eye(4, device=DEVICE).float()
            extrinsics = extrinsics.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1)

            batch = {
                "context": {
                    "image": images,
                    "intrinsics": K,
                    "extrinsics": extrinsics,
                    "near": torch.tensor([[0.1, 0.1]], device=DEVICE),
                    "far": torch.tensor([[100., 100.]], device=DEVICE),
                },
                "target": {
                    "image": images,
                    "intrinsics": K,
                    "extrinsics": extrinsics,
                    "near": torch.tensor([[0.1, 0.1]], device=DEVICE),
                    "far": torch.tensor([[100., 100.]], device=DEVICE),
                }
            }

            batch = data_shim(batch)

            visualization_dump = {}
            gaussians = encoder(
                batch["context"],
                global_step=0,
                visualization_dump=visualization_dump
            )

            pts3d = visualization_dump['means'][0, 1].squeeze(-2)
            opacity = visualization_dump['opacities'][0, 1].squeeze(-1).squeeze(-1)

            pose1 = torch.eye(4, device=DEVICE)
            pose2 = estimate_pose(pts3d, opacity, K[0, 1], IMG_SIZE, IMG_SIZE)

            out = decoder.forward(
                gaussians,
                pose1.unsqueeze(0).unsqueeze(0),
                K[:, 0:1],
                torch.tensor([[0.1]], device=DEVICE),
                torch.tensor([[100.]], device=DEVICE),
                (IMG_SIZE, IMG_SIZE),
                depth_mode=None,
                decoder_type="3D"
            )

            color = out.color[0, 0].permute(1, 2, 0).cpu().numpy()

            save_path = os.path.join(OUTPUT_DIR, f"{folder}.png")
            Image.fromarray((color * 255).astype(np.uint8)).save(save_path)

            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error in {folder}: {e}")

    print("Done. Outputs saved in 'outputs/'")

if __name__ == "__main__":
    run()