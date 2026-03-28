


import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image

st.set_page_config(page_title="DiFix Pipeline", layout="wide")

# ─────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────
st.title("DiFix Pipeline")
st.caption("Reference-guided inpainting · Nvidia DiFix + Stable Diffusion")

# ─────────────────────────────────────────────
# Resize utilities (IMPORTANT FIX)
# ─────────────────────────────────────────────
def resize_to_multiple_of_8(img):
    w, h = img.size
    w = (w // 8) * 8
    h = (h // 8) * 8
    return img.resize((w, h), Image.BICUBIC)


def match_sizes(input_pil, mask_pil, ref_pil):
    input_pil = resize_to_multiple_of_8(input_pil)
    w, h = input_pil.size

    ref_pil = ref_pil.resize((w, h), Image.BICUBIC)
    mask_pil = mask_pil.resize((w, h), Image.NEAREST)

    return input_pil, mask_pil, ref_pil


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────
def difix_input(image_pil, mask_pil):
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    mask = np.array(mask_pil.convert("L"))

    h, w = image.shape[:2]
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    blurred = cv2.GaussianBlur(image, (21, 21), 0)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_norm = mask_3ch / 255.0

    output = (blurred * mask_norm + image * (1 - mask_norm)).astype(np.uint8)
    output_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    mask_pil_out = Image.fromarray(mask)

    return mask_pil_out, output_pil


# ─────────────────────────────────────────────
# Load models (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_difix():
    from pipeline_difix import DifixPipeline
    pipe = DifixPipeline.from_pretrained(
        "bappaiitj/DL_project_difix_tuned", trust_remote_code=True
    ).to("cuda")
    return pipe


@st.cache_resource
def load_sd():
    from diffusers import StableDiffusionInpaintPipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
    ).to("cuda")
    return pipe


# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────
def run_pipeline(input_pil, mask_pil, ref_pil, prompt):
    # 🔥 FIX: enforce same size
    input_pil, mask_pil, ref_pil = match_sizes(input_pil, mask_pil, ref_pil)

    pipe_difix = load_difix()

    mask_pil_proc, blurred_pil = difix_input(input_pil, mask_pil)

    difix_out = pipe_difix(
        "add dog",
        image=blurred_pil,
        ref_image=ref_pil,
        num_inference_steps=1,
        timesteps=[199],
        guidance_scale=0.0,
    ).images[0]

    if prompt and prompt.strip():
        pipe_sd = load_sd()

        result = pipe_sd(
            prompt=prompt.strip(),
            image=difix_out,
            mask_image=mask_pil_proc,
        ).images[0]

        return result

    return difix_out


# ─────────────────────────────────────────────
# UI Layout
# ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Inputs")

    input_file = st.file_uploader("Input Image", type=["png", "jpg", "jpeg"])
    mask_file = st.file_uploader("Mask (white = inpaint)", type=["png", "jpg", "jpeg"])
    ref_file = st.file_uploader("Reference Image", type=["png", "jpg", "jpeg"])

    prompt = st.text_input(
        "Prompt (optional)",
        placeholder="e.g. add a pink flower"
    )

    run = st.button("▶ Run Pipeline")

with col2:
    st.subheader("Output")

    if run:
        if not input_file or not mask_file or not ref_file:
            st.error("Please upload all required images.")
        else:
            input_pil = Image.open(input_file).convert("RGB")
            mask_pil = Image.open(mask_file).convert("RGB")
            ref_pil = Image.open(ref_file).convert("RGB")

            # Debug (optional)
            st.write("Input size:", input_pil.size)

            with st.spinner("Generating..."):
                result = run_pipeline(input_pil, mask_pil, ref_pil, prompt)

            st.image(result, caption="Result", use_column_width=True)

            # Free memory
            torch.cuda.empty_cache()
