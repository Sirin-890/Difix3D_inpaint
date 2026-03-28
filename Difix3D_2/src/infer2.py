# # from pipeline_difix import DifixPipeline
# # from diffusers.utils import load_image
# # pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
# # pipe.to("cuda")
# # input_image = load_image("assets/example_input.png")
# # prompt = "add a dog in this photo"
# # output_image = pipe(prompt, image=input_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
# # output_image.save("example_output2.png")


from pipeline_difix import DifixPipeline
from diffusers.utils import load_image

pipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
pipe.to("cuda")

input_image = load_image("masked_input.jpg")
ref_image = load_image("assets/example_ref.png")
prompt = "add dog "

output_image = pipe(prompt, image=input_image, ref_image=ref_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
output_image.save("exam_output.png")



# import cv2

# # Load image
# img = cv2.imread("assets/example_input.png")

# # Get image dimensions
# h, w, _ = img.shape

# # Define patch size (e.g., 1/4 of width & height)
# patch_w, patch_h = w // 4, h // 4

# # Compute center coordinates
# x1 = w // 2 - patch_w // 2
# y1 = h // 2 - patch_h // 2
# x2 = x1 + patch_w
# y2 = y1 + patch_h

# # Draw black rectangle (filled)
# cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)

# # Save output
# cv2.imwrite("masked_input.jpg", img)