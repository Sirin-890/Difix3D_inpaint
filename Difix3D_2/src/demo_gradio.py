# # import cv2
# # import numpy as np
# # import torch
# # import gradio as gr
# # from PIL import Image


# # def difix_input(image_pil, mask_pil):
# #     image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
# #     mask = np.array(mask_pil.convert("L"))

# #     h, w = image.shape[:2]
# #     mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
# #     _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# #     blurred = cv2.GaussianBlur(image, (21, 21), 0)
# #     mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
# #     mask_norm = mask_3ch / 255.0

# #     output = (blurred * mask_norm + image * (1 - mask_norm)).astype(np.uint8)
# #     output_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
# #     mask_pil_out = Image.fromarray(mask)

# #     return mask_pil_out, output_pil


# # def difix_process(input_pil, ref_pil):
# #     from pipeline_difix import DifixPipeline

# #     pipe = DifixPipeline.from_pretrained(
# #         "nvidia/difix_ref", trust_remote_code=True
# #     ).to("cuda")

# #     prompt = "add dog"

# #     output = pipe(
# #         prompt,
# #         image=input_pil,
# #         ref_image=ref_pil,
# #         num_inference_steps=1,
# #         timesteps=[199],
# #         guidance_scale=0.0,
# #     ).images[0]

# #     return output


# # def text_based(img_pil, mask_pil, prompt):
# #     from diffusers import StableDiffusionInpaintPipeline

# #     pipe = StableDiffusionInpaintPipeline.from_pretrained(
# #         "stable-diffusion-v1-5/stable-diffusion-inpainting",
# #         torch_dtype=torch.float16,
# #     ).to("cuda")

# #     result = pipe(prompt=prompt, image=img_pil, mask_image=mask_pil).images[0]
# #     return result


# # def flow(input_pil, mask_pil, ref_pil, prompt=None):
# #     if input_pil is None or mask_pil is None or ref_pil is None:
# #         raise gr.Error("Please upload an input image, mask, and reference image.")

# #     mask_pil_proc, blurred_pil = difix_input(input_pil, mask_pil)
# #     difix_out = difix_process(blurred_pil, ref_pil)

# #     if prompt and prompt.strip():
# #         return text_based(difix_out, mask_pil_proc, prompt.strip())
# #     return difix_out


# # # ── Gradio UI ──────────────────────────────────────────────────────────────

# # css = """
# # @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Mono:wght@400;500&display=swap');

# # :root {
# #     --bg: #0d0d0f;
# #     --surface: #16161a;
# #     --surface2: #1e1e24;
# #     --border: #2a2a35;
# #     --accent: #7c6af7;
# #     --accent2: #e85d96;
# #     --text: #e8e8f0;
# #     --muted: #6b6b80;
# #     --radius: 12px;
# # }

# # * { box-sizing: border-box; }

# # body, .gradio-container {
# #     background: var(--bg) !important;
# #     font-family: 'DM Mono', monospace !important;
# #     color: var(--text) !important;
# # }

# # .gradio-container { max-width: 1100px !important; margin: 0 auto !important; padding: 32px 20px !important; }

# # #header {
# #     text-align: center;
# #     margin-bottom: 40px;
# #     padding: 40px 0 32px;
# #     border-bottom: 1px solid var(--border);
# # }

# # #header h1 {
# #     font-family: 'Syne', sans-serif !important;
# #     font-size: 2.8rem;
# #     font-weight: 800;
# #     letter-spacing: -0.03em;
# #     background: linear-gradient(135deg, var(--accent), var(--accent2));
# #     -webkit-background-clip: text;
# #     -webkit-text-fill-color: transparent;
# #     background-clip: text;
# #     margin: 0 0 8px;
# # }

# # #header p {
# #     color: var(--muted);
# #     font-size: 0.85rem;
# #     margin: 0;
# #     letter-spacing: 0.08em;
# #     text-transform: uppercase;
# # }

# # .panel {
# #     background: var(--surface) !important;
# #     border: 1px solid var(--border) !important;
# #     border-radius: var(--radius) !important;
# #     padding: 20px !important;
# # }

# # .section-label {
# #     font-family: 'Syne', sans-serif;
# #     font-size: 0.7rem;
# #     font-weight: 600;
# #     letter-spacing: 0.15em;
# #     text-transform: uppercase;
# #     color: var(--accent);
# #     margin-bottom: 14px;
# # }

# # label span, .label-wrap span {
# #     font-family: 'DM Mono', monospace !important;
# #     font-size: 0.78rem !important;
# #     color: var(--muted) !important;
# #     letter-spacing: 0.05em !important;
# # }

# # .gr-image, .image-container {
# #     border-radius: 8px !important;
# #     border: 1px solid var(--border) !important;
# #     background: var(--surface2) !important;
# #     overflow: hidden !important;
# # }

# # textarea, input[type="text"] {
# #     background: var(--surface2) !important;
# #     border: 1px solid var(--border) !important;
# #     border-radius: 8px !important;
# #     color: var(--text) !important;
# #     font-family: 'DM Mono', monospace !important;
# #     font-size: 0.85rem !important;
# #     padding: 12px 14px !important;
# #     transition: border-color 0.2s !important;
# # }

# # textarea:focus, input[type="text"]:focus {
# #     border-color: var(--accent) !important;
# #     outline: none !important;
# #     box-shadow: 0 0 0 2px rgba(124,106,247,0.15) !important;
# # }

# # button.primary-btn, #run-btn {
# #     background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
# #     border: none !important;
# #     border-radius: 8px !important;
# #     color: #fff !important;
# #     font-family: 'Syne', sans-serif !important;
# #     font-weight: 600 !important;
# #     font-size: 0.95rem !important;
# #     letter-spacing: 0.04em !important;
# #     padding: 14px 28px !important;
# #     cursor: pointer !important;
# #     width: 100% !important;
# #     transition: opacity 0.2s, transform 0.15s !important;
# # }

# # button.primary-btn:hover, #run-btn:hover {
# #     opacity: 0.88 !important;
# #     transform: translateY(-1px) !important;
# # }

# # .note {
# #     font-size: 0.72rem;
# #     color: var(--muted);
# #     margin-top: 6px;
# #     line-height: 1.5;
# # }
# # """

# # with gr.Blocks(css=css, title="DiFix Pipeline") as demo:

# #     gr.HTML("""
# #         <div id="header">
# #             <h1>DiFix Pipeline</h1>
# #             <p>Reference-guided inpainting · Nvidia DiFix + Stable Diffusion</p>
# #         </div>
# #     """)

# #     with gr.Row(equal_height=False):
# #         # ── Left column: inputs ──────────────────────────────────────────
# #         with gr.Column(scale=1, elem_classes="panel"):
# #             gr.HTML('<div class="section-label">Inputs</div>')

# #             input_image = gr.Image(
# #                 label="Input Image",
# #                 type="pil",
# #                 sources=["upload"],
# #             )

# #             mask_image = gr.Image(
# #                 label="Mask  (white = inpaint region)",
# #                 type="pil",
# #                 sources=["upload"],
# #             )

# #             ref_image = gr.Image(
# #                 label="Reference Image",
# #                 type="pil",
# #                 sources=["upload"],
# #             )

# #             prompt_box = gr.Textbox(
# #                 label="Text Prompt  (optional — triggers SD inpainting pass)",
# #                 placeholder="e.g. add a pink flower",
# #                 lines=2,
# #             )
# #             gr.HTML('<div class="note">Leave blank to return the DiFix output directly.</div>')

# #             run_btn = gr.Button("▶  Run Pipeline", elem_id="run-btn", variant="primary")

# #         # ── Right column: output ─────────────────────────────────────────
# #         with gr.Column(scale=1, elem_classes="panel"):
# #             gr.HTML('<div class="section-label">Output</div>')

# #             output_image = gr.Image(
# #                 label="Result",
# #                 type="pil",
# #                 interactive=False,
# #             )

# #     run_btn.click(
# #         fn=flow,
# #         inputs=[input_image, mask_image, ref_image, prompt_box],
# #         outputs=[output_image],
# #     )

# # if __name__ == "__main__":
# #     demo.launch()



# import streamlit as st
# import cv2
# import numpy as np
# import torch
# from PIL import Image

# st.set_page_config(page_title="DiFix Pipeline", layout="wide")

# # ─────────────────────────────────────────────
# # Title
# # ─────────────────────────────────────────────
# st.title("DiFix Pipeline")
# st.caption("Reference-guided inpainting · Nvidia DiFix + Stable Diffusion")

# # ─────────────────────────────────────────────
# # Utils
# # ─────────────────────────────────────────────
# def difix_input(image_pil, mask_pil):
#     image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
#     mask = np.array(mask_pil.convert("L"))

#     h, w = image.shape[:2]
#     mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
#     _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

#     blurred = cv2.GaussianBlur(image, (21, 21), 0)
#     mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
#     mask_norm = mask_3ch / 255.0

#     output = (blurred * mask_norm + image * (1 - mask_norm)).astype(np.uint8)
#     output_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
#     mask_pil_out = Image.fromarray(mask)

#     return mask_pil_out, output_pil


# # ─────────────────────────────────────────────
# # Load models ONCE (important for GPU)
# # ─────────────────────────────────────────────
# @st.cache_resource
# def load_difix():
#     from pipeline_difix import DifixPipeline
#     pipe = DifixPipeline.from_pretrained(
#         "nvidia/difix_ref", trust_remote_code=True
#     ).to("cuda")
#     return pipe


# @st.cache_resource
# def load_sd():
#     from diffusers import StableDiffusionInpaintPipeline
#     pipe = StableDiffusionInpaintPipeline.from_pretrained(
#         "stable-diffusion-v1-5/stable-diffusion-inpainting",
#         torch_dtype=torch.float16,
#     ).to("cuda")
#     return pipe


# # ─────────────────────────────────────────────
# # Processing
# # ─────────────────────────────────────────────
# def run_pipeline(input_pil, mask_pil, ref_pil, prompt):
#     pipe_difix = load_difix()

#     mask_pil_proc, blurred_pil = difix_input(input_pil, mask_pil)

#     difix_out = pipe_difix(
#         "add dog",
#         image=blurred_pil,
#         ref_image=ref_pil,
#         num_inference_steps=1,
#         timesteps=[199],
#         guidance_scale=0.0,
#     ).images[0]

#     if prompt and prompt.strip():
#         pipe_sd = load_sd()
#         result = pipe_sd(
#             prompt=prompt,
#             image=difix_out,
#             mask_image=mask_pil_proc,
#         ).images[0]
#         return result

#     return difix_out


# # ─────────────────────────────────────────────
# # UI Layout
# # ─────────────────────────────────────────────
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Inputs")

#     input_file = st.file_uploader("Input Image", type=["png", "jpg", "jpeg"])
#     mask_file = st.file_uploader("Mask (white = inpaint)", type=["png", "jpg", "jpeg"])
#     ref_file = st.file_uploader("Reference Image", type=["png", "jpg", "jpeg"])

#     prompt = st.text_input(
#         "Prompt (optional)",
#         placeholder="e.g. add a pink flower"
#     )

#     run = st.button("▶ Run Pipeline")

# with col2:
#     st.subheader("Output")

#     if run:
#         if not input_file or not mask_file or not ref_file:
#             st.error("Please upload all required images.")
#         else:
#             input_pil = Image.open(input_file).convert("RGB")
#             mask_pil = Image.open(mask_file).convert("RGB")
#             ref_pil = Image.open(ref_file).convert("RGB")

#             with st.spinner("Generating..."):
#                 result = run_pipeline(input_pil, mask_pil, ref_pil, prompt)

#             st.image(result, caption="Result", use_column_width=True)

#             torch.cuda.empty_cache()



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
        "nvidia/difix_ref", trust_remote_code=True
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