#!pip install torch torchvision torchaudio transformers diffusers
#These are run and tested on Google Collab

def generate_image(prompt):
    import torch
    from diffusers import StableDiffusionPipeline
    # Use the Runway stable-diffusion-v1-5 model ID
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    # Generate the image
    image = pipe(prompt, guidance_scale=7.5).images[0]
    return image

# Define the prompt
prompt = "Your_Prompt"
# Generate and return the image
generate_image(prompt)