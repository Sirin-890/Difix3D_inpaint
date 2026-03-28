# import cv2
# import numpy as np
# def difix_input():
#     image = cv2.imread("original.jpg")
#     mask = cv2.imread("mask.png", 0)
#     h, w = image.shape[:2]
#     mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
#     _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
#     blurred = cv2.GaussianBlur(image, (21, 21), 0)
#     mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
#     mask_norm = mask_3ch / 255.0
#     output = (blurred * mask_norm + image * (1 - mask_norm)).astype(np.uint8)
#     cv2.imwrite("output.jpg", output)
#     return mask,output


# def difix_process(input_path,ref_path):
#     from pipeline_difix import DifixPipeline
#     from diffusers.utils import load_image

#     pipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
#     pipe.to("cuda")

#     input_image = load_image("masked_input.jpg")
#     ref_image = load_image("assets/example_ref.png")
#     prompt = "add dog "

#     output_image = pipe(prompt, image=input_image, ref_image=ref_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
#     output_image.save("example_output.png")
#     return output_image



# def text_based(img,mask,prompt):
#     from diffusers import StableDiffusionInpaintPipeline
   
#     init_image = img
#     mask_image = mask

#     pipe = StableDiffusionInpaintPipeline.from_pretrained(
#         "stable-diffusion-v1-5/stable-diffusion-inpainting", torch_dtype=torch.float16
#     )
#     pipe = pipe.to("cuda")

#     prompt = prompt
#     image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]



# def flow(input_path,mask_path,ref_path,prompt=None):
#     mask,gus_img=difix_input(input_path,mask_path)
#     out=difix_process(gus_img,ref_path)
#     if prompt==None:
#         return out
#     else:
#         final_out=text_based(out,mask,prompt)
#         return final_out


# if __name__=="__main__":
#     flow()


import cv2
import numpy as np
import torch
from PIL import Image



import numpy as np
from PIL import Image, ImageDraw
import random


def generate_random_mask(image_path):
    image = Image.open(image_path)
    width, height = image.size

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    num_shapes = random.randint(5, 15)

    for _ in range(num_shapes):
        shape_type = random.choice(["ellipse", "rectangle", "line"])

        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)

        # ✅ FIX: sort coordinates
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])

        if shape_type == "ellipse":
            draw.ellipse([x_min, y_min, x_max, y_max], fill=255)

        elif shape_type == "rectangle":
            draw.rectangle([x_min, y_min, x_max, y_max], fill=255)

        elif shape_type == "line":
            thickness = random.randint(10, 40)
            draw.line([x1, y1, x2, y2], fill=255, width=thickness)

    return mask

def difix_input(image_path, mask_path):
    
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)

    
    h, w = image.shape[:2]
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

   
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    blurred = cv2.GaussianBlur(image, (21, 21), 0)

    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_norm = mask_3ch / 255.0

    output = (blurred * mask_norm + image * (1 - mask_norm)).astype(np.uint8)
    output_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(mask)

    return mask_pil, output_pil


def difix_process(input_pil, ref_path):
    from pipeline_difix import DifixPipeline
    from diffusers.utils import load_image

    pipe = DifixPipeline.from_pretrained(
        "nvidia/difix_ref", trust_remote_code=True
    ).to("cuda")

    ref_image = load_image(ref_path)

    prompt = "add dog"

    output = pipe(
        prompt,
        image=input_pil,
        ref_image=ref_image,
        num_inference_steps=1,
        timesteps=[199],
        guidance_scale=0.0
    ).images[0]

    return output

def text_based(img_pil, mask_pil, prompt):
    from diffusers import StableDiffusionInpaintPipeline

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-inpainting",
        torch_dtype=torch.float16
    ).to("cuda")

    result = pipe(
        prompt=prompt,
        image=img_pil,
        mask_image=mask_pil
    ).images[0]

    return result

def flow(input_path, mask_path, ref_path, prompt=None):
    mask_pil, blurred_pil = difix_input(input_path, mask_path)

    difix_out = difix_process(blurred_pil, ref_path)

    if prompt is None:
        return difix_out
    else:
        final_out = text_based(difix_out, mask_pil, prompt)
        return final_out


if __name__ == "__main__":

    mask = generate_random_mask("assets/example_input.png")

    mask.save("random_mask.png")

    output = flow(
        input_path="assets/example_input.png",
        mask_path="random_mask.png",
        ref_path="assets/example_ref.png",
        prompt="add a pink flower"
    )

    output.save("final_output.png")



