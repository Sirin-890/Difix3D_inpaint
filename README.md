# Difix3D+ Inpainting  
**Mask-Based Inpainting & Text-Guided Enhancement for 3D Reconstruction**

---

## General Overview
This project extends **DIFIX3D+**, a single-step diffusion-based refinement framework for improving 3D reconstruction outputs (NeRF / 3D Gaussian Splatting).

We introduce:
- **Mask-Guided Inpainting** → fixes only corrupted regions instead of entire images  
- **Text-Guided Enhancement** → allows semantic control using prompts  

The goal is to improve **visual quality, structural consistency, and controllability** of reconstructed views.

---

## Key Contributions
- Localized artifact correction using **latent-space masking**
- Integration of **text-guided diffusion** for semantic refinement
- Built on **single-step diffusion (SD-Turbo)** for fast inference
- No retraining of base Difix3D backbone required

---

## Experiments Performed

### Datasets
- **NeRF-Synthetic Dataset**
  - Scenes: `playroom`, `truck`, `train`, `dr_johnson`

- **Nersemble Dataset (G3Splat)**
  - Human-centric dataset (subset used due to compute limits)

- **Synthetic 3D Face Dataset by Microsoft Research**

---
### YouTube video link : [YouTube](https://www.youtube.com/watch?v=CqEvETOnCuo)

### Hugging Face model Link : [Model Link](https://huggingface.co/bappaiitj/DL_project_difix_tuned)
---

### Setup
- GPU: NVIDIA A100  
- Backbone: SD-Turbo  
- Noise level: τ = 200  
- Masks: Manually generated to simulate missing regions  

---

### Evaluation Metrics
- **PSNR** (↑) – Pixel accuracy  
- **SSIM** (↑) – Structural similarity  
- **LPIPS** (↓) – Perceptual similarity  
- **FID** (↓) – Visual realism  

---

### Results
| Metric | Value |
|------|------|
| PSNR | 10.386 |
| SSIM | 0.338 |
| LPIPS | 0.697 |
| FID | 43.47 |

### Observations
- Better **completion of missing regions**
- Improved **multi-view consistency**
- Slight blurring in very fine details

---

## How to Run Difix3D+ Inpainting model

### 1. Clone the Repository
```bash
git clone https://github.com/Sirin-890/Difix3D_inpaint.git
cd Difix3D_inpaint
```
### 2. Create environment and install the dependencies
``` bash
pip install -r Difix3D_2/requirements.txt
```
### 3. Run the streamlit file
``` bash
streamlit run Difix3D_2/src/demo_stream.py
```

## How to Run g3splat model
```bash
cd g3splat
```
### 2. Create environment and install the dependencies
``` bash
pip install -r g3splat/requirements.txt
```
### 3. Run the gradio file
``` bash
python demo.py
```
Please Note - The environment dependencies of all the 3 models are different
---
## Authors
- Sirin Changulani (B23CM1038)
- Sonam Sikarwar (B23CM1060)
- Harshita Vachhani (B23EE1026)

### Pretrained Models & Resources

- Difix (NVIDIA): https://huggingface.co/nvidia/difix
- gsplat (CUDA library): https://github.com/nerfstudio-project/gsplat
- gsplat viewer (HF): https://github.com/huggingface/gsplat.js


