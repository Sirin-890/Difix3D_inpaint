# import json
# import random
# import torch
# from PIL import Image
# import torchvision.transforms.functional as F


# def random_mask(h, w, min_ratio=0.1, max_ratio=0.5):
#     """
#     Generate a single random rectangular mask.
#     Returns (1, H, W) float32 tensor — 1 = inpaint region, 0 = known.
#     """
#     mask = torch.zeros(1, h, w)
#     mh = random.randint(int(h * min_ratio), int(h * max_ratio))
#     mw = random.randint(int(w * min_ratio), int(w * max_ratio))
#     top  = random.randint(0, h - mh)
#     left = random.randint(0, w - mw)
#     mask[:, top:top + mh, left:left + mw] = 1.0
#     return mask


# def load_mask(mask_path, image_size):
#     """
#     Load a mask from disk (grayscale PNG — white = inpaint region).
#     Returns (1, H, W) float32 tensor with values in {0, 1}.
#     """
#     mask = Image.open(mask_path).convert("L")
#     mask_t = F.to_tensor(mask)                          # (1, H, W)  [0, 1]
#     mask_t = F.resize(mask_t, image_size,
#                       interpolation=F.InterpolationMode.NEAREST)
#     return (mask_t > 0.5).float()


# class PairedDataset(torch.utils.data.Dataset):
#     """
#     JSON schema per split entry:
#     {
#       "<img_id>": {
#         "image":        "<path>",          # degraded / source input
#         "target_image": "<path>",          # clean ground truth
#         "ref_image":    "<path>",          # optional second view
#         "mask":         "<path>",          # optional — grayscale PNG mask
#         "ref_mask":     "<path>",          # optional — mask for ref view
#         "prompt":       "<text>"
#       }, ...
#     }

#     If "mask" is absent the dataset generates a random rectangular mask
#     on the fly (useful when you have no labelled inpaint pairs).
#     Set mask_min_ratio / mask_max_ratio to control random mask coverage.
#     """

#     def __init__(
#         self,
#         dataset_path,
#         split,
#         height=576,
#         width=1024,
#         tokenizer=None,
#         mask_min_ratio=0.1,
#         mask_max_ratio=0.5,
#     ):
#         super().__init__()
#         with open(dataset_path, "r") as f:
#             self.data = json.load(f)[split]
#         self.img_ids = list(self.data.keys())
#         self.image_size = (height, width)
#         self.tokenizer = tokenizer
#         self.mask_min_ratio = mask_min_ratio
#         self.mask_max_ratio = mask_max_ratio

#     def __len__(self):
#         return len(self.img_ids)

#     # ── internal helpers ──────────────────────────────────────────────────────

#     def _to_tensor(self, pil_img):
#         """Convert PIL → normalised float tensor (3, H, W) in [-1, 1]."""
#         t = F.to_tensor(pil_img.convert("RGB"))         # (3, H, W)  [0, 1]
#         t = F.resize(t, self.image_size,
#                      interpolation=F.InterpolationMode.LANCZOS)
#         t = F.normalize(t, mean=[0.5, 0.5, 0.5],
#                            std= [0.5, 0.5, 0.5])
#         return t

#     def _get_mask(self, entry_data, key="mask"):
#         """
#         Return a (1, H, W) float mask.
#         Uses the on-disk mask when available, otherwise generates randomly.
#         """
#         h, w = self.image_size
#         if key in entry_data and entry_data[key] is not None:
#             return load_mask(entry_data[key], self.image_size)
#         return random_mask(h, w, self.mask_min_ratio, self.mask_max_ratio)

#     # ── main item loader ──────────────────────────────────────────────────────

#     def __getitem__(self, idx):
#         img_id = self.img_ids[idx]
#         entry  = self.data[img_id]

#         # ── load images ───────────────────────────────────────────────────────
#         try:
#             input_img  = Image.open(entry["image"])
#             output_img = Image.open(entry["target_image"])
#         except Exception as e:
#             print(f"Error loading images for id={img_id}: {e}")
#             return self.__getitem__((idx + 1) % len(self))   # safe wrap

#         input_t  = self._to_tensor(input_img)     # (3, H, W)
#         output_t = self._to_tensor(output_img)    # (3, H, W)

#         # ── load / generate masks ─────────────────────────────────────────────
#         mask_t = self._get_mask(entry, key="mask")    # (1, H, W)

#         # ── optional reference view ───────────────────────────────────────────
#         has_ref = "ref_image" in entry and entry["ref_image"] is not None
#         if has_ref:
#             try:
#                 ref_img = Image.open(entry["ref_image"])
#             except Exception as e:
#                 print(f"Error loading ref_image for id={img_id}: {e}")
#                 has_ref = False

#         if has_ref:
#             ref_t      = self._to_tensor(ref_img)                     # (3, H, W)
#             ref_mask_t = self._get_mask(entry, key="ref_mask")        # (1, H, W)

#             # Stack along view dimension → (V=2, C, H, W)
#             input_t  = torch.stack([input_t,  ref_t],      dim=0)
#             output_t = torch.stack([output_t, ref_t],      dim=0)
#             mask_t   = torch.stack([mask_t,   ref_mask_t], dim=0)
#         else:
#             # Single view → (V=1, C, H, W)
#             input_t  = input_t.unsqueeze(0)
#             output_t = output_t.unsqueeze(0)
#             mask_t   = mask_t.unsqueeze(0)

#         # ── build output dict ─────────────────────────────────────────────────
#         caption = entry.get("prompt", "")
#         out = {
#             "conditioning_pixel_values": input_t,    # (V, 3, H, W)
#             "output_pixel_values":       output_t,   # (V, 3, H, W)
#             "mask_pixel_values":         mask_t,     # (V, 1, H, W)
#             "caption":                   caption,
#         }

#         if self.tokenizer is not None:
#             out["input_ids"] = self.tokenizer(
#                 caption,
#                 max_length=self.tokenizer.model_max_length,
#                 padding="max_length",
#                 truncation=True,
#                 return_tensors="pt",
#             ).input_ids

#         return out





import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode


def random_mask(h, w, min_ratio=0.1, max_ratio=0.5):
    """1 = inpaint region, 0 = known. Returns (1, H, W) float32."""
    mask = torch.zeros(1, h, w)
    mh = random.randint(int(h * min_ratio), int(h * max_ratio))
    mw = random.randint(int(w * min_ratio), int(w * max_ratio))
    top  = random.randint(0, h - mh)
    left = random.randint(0, w - mw)
    mask[:, top:top + mh, left:left + mw] = 1.0
    return mask


class PairedDataset(Dataset):
    """
    Layout
    ------
    dataset_synthetic/
        0/          ← scene folder
            0.png
            1.png
            2.png
            3.png
            ...
        1/
            ...

    Each sliding window triplet (i, i+1, i+2) is one sample:
        i     → degraded input   (view 0 conditioning)
        i+1   → ground truth     (view 0 target)
        i+2   → reference view   (view 1 conditioning + target)

    A scene with N images yields (N-2) samples.
    Example: 9 images → 7 samples per scene.

    Output keys
    -----------
    conditioning_pixel_values : (2, 3, H, W)
    output_pixel_values       : (2, 3, H, W)
    mask_pixel_values         : (2, 1, H, W)  mask on view 0, zeros on view 1
    caption                   : str
    input_ids                 : (1, L)         if tokenizer given
    """

    def __init__(
        self,
        dataset_path: str,
        split: str,
        height: int = 512,
        width:  int = 512,
        tokenizer=None,
        prompt: str = "",
        mask_min_ratio: float = 0.1,
        mask_max_ratio: float = 0.5,
        train_split_ratio: float = 0.9,
        seed: int = 42,
    ):
        super().__init__()
        self.image_size     = (height, width)
        self.tokenizer      = tokenizer
        self.prompt         = prompt
        self.mask_min_ratio = mask_min_ratio
        self.mask_max_ratio = mask_max_ratio

        # ── discover scene folders ────────────────────────────────────────────
        scene_dirs = sorted([
            os.path.join(dataset_path, d)
            for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ], key=lambda p: int(os.path.basename(p)) if os.path.basename(p).isdigit() else 0)

        assert len(scene_dirs) > 0, f"No scene folders found in {dataset_path}"

        # ── build flat sample list: (scene_dir, i) where triplet = i, i+1, i+2 ──
        all_samples = []
        for scene_dir in scene_dirs:
            indices = sorted([
                int(f[:-4])
                for f in os.listdir(scene_dir)
                if f.endswith(".png") and f[:-4].isdigit()
            ])

            if len(indices) < 3:
                print(f"[PairedDataset] Skipping {scene_dir} — needs at least 3 images.")
                continue

            idx_set = set(indices)
            for i in indices:
                # only add if full triplet (i, i+1, i+2) exists on disk
                if (i + 1) in idx_set and (i + 2) in idx_set:
                    all_samples.append((scene_dir, i))

        assert len(all_samples) > 0, (
            "No valid triplets found. Each scene needs at least 3 "
            "consecutively numbered .png files (e.g. 0.png, 1.png, 2.png)."
        )

        # ── deterministic train / test split on samples (not scenes) ─────────
        # Splitting on samples rather than scenes keeps the split stable and
        # avoids data leakage only when scenes are fully independent. If you
        # prefer a scene-level split (safer), set train_split_ratio on scenes
        # before building all_samples — but for a small dataset sample-level
        # maximises the number of training examples.
        rng = random.Random(seed)
        shuffled = all_samples[:]
        rng.shuffle(shuffled)
        n_train = max(1, int(len(shuffled) * train_split_ratio))

        if split == "train":
            self.samples = shuffled[:n_train]
        else:
            self.samples = shuffled[n_train:] if len(shuffled) > n_train else shuffled

        self.img_names = [f"{s[0]}_{s[1]}" for s in self.samples]

        print(f"[PairedDataset] split={split}  scenes={len(scene_dirs)}  "
              f"total_triplets={len(all_samples)}  this_split={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    # ── helpers ───────────────────────────────────────────────────────────────

    # def _to_tensor(self, pil_img: Image.Image) -> torch.Tensor:
    #     t = F.to_tensor(pil_img.convert("RGB"))
    #     t = F.resize(t, self.image_size, interpolation=InterpolationMode.LANCZOS)
    #     t = F.normalize(t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    #     return t
    def _to_tensor(self, pil_img: Image.Image) -> torch.Tensor:
        h, w = self.image_size
        pil_img = pil_img.resize((w, h), Image.LANCZOS)  # resize PIL first (W, H order)
        t = F.to_tensor(pil_img)
        t = F.normalize(t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return t

    def _load(self, scene_dir: str, idx: int) -> Image.Image:
        return Image.open(os.path.join(scene_dir, f"{idx}.png")).convert("RGB")

    # ── main ──────────────────────────────────────────────────────────────────

    def __getitem__(self, idx: int):
        scene_dir, i = self.samples[idx]

        try:
            input_t  = self._to_tensor(self._load(scene_dir, i))        # i   → input
            target_t = self._to_tensor(self._load(scene_dir, i + 1))    # i+1 → gt
            ref_t    = self._to_tensor(self._load(scene_dir, i + 2))    # i+2 → ref
        except Exception as e:
            print(f"[PairedDataset] Load error scene={scene_dir} i={i}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        h, w = self.image_size
        input_mask = random_mask(h, w, self.mask_min_ratio, self.mask_max_ratio)
        ref_mask   = torch.zeros(1, h, w)

        conditioning = torch.stack([input_t,    ref_t],    dim=0)  # (2, 3, H, W)
        output       = torch.stack([target_t,   ref_t],   dim=0)  # (2, 3, H, W)
        mask         = torch.stack([input_mask, ref_mask], dim=0)  # (2, 1, H, W)

        caption_path = os.path.join(scene_dir, "caption.txt")
        caption = (
            open(caption_path).read().strip()
            if os.path.exists(caption_path)
            else self.prompt
        )

        out = {
            "conditioning_pixel_values": conditioning,
            "output_pixel_values":       output,
            "mask_pixel_values":         mask,
            "caption":                   caption,
        }

        if self.tokenizer is not None:
            out["input_ids"] = self.tokenizer(
                caption,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids

        return out