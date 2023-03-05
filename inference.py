import torch
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from tqdm import tqdm
import numpy as np


class StableDiffusion:
    def __init__(self, height=256, width=256):
        self.height = height
        self.width = width
        
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

        self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
        self.scheduler = LMSDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

        self.to_device()
        
    def to_device(self):
        self.vae.to(self.device)
        self.text_encoder.to(self.device)
        self.unet.to(self.device) 
    
    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def encode_text(self, prompt, max_length=50):
        tokens = self.tokenizer(prompt, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
        with torch.no_grad():
            return self.text_encoder(tokens.input_ids.to(self.device))[0]
        
    def encode_image(self, image):
        processor = CLIPImageProcessor()
        image = processor(image).unsqueeze(0).to(self.device)
        return self.vae.encode(image).to(self.device)
    
    def decode(self, latents):
        latents = 1 / 0.18215 * latents
        return self.vae.decode(latents).sample
    
    def set_seed(self, seed):
        return torch.manual_seed(seed)
    
    def get_latent(self, seed, batch_size=1):
        return torch.randn((batch_size, self.unet.in_channels, self.height // 8, self.width // 8), generator=seed).to(self.device)
    
    
    def generate(self, prompt, batch_size=1, steps=25, guidance_scale=7.5, seed=42):
        seed = self.set_seed(seed)
        latents = self.get_latent(seed)
        text_embedding = self.encode_text([prompt])
        text_embedding_unconditional = self.encode_text([""] * batch_size)
        text_embeddings = torch.cat([text_embedding_unconditional, text_embedding])
        
        self.scheduler.set_timesteps(steps)
        latents = latents * self.scheduler.init_noise_sigma
        
        for t in tqdm(self.scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
        return self.decode(latents)
    
    def show(self, image):
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images    


if __name__ == "__main__":
    model = StableDiffusion()
    model.to_device()
    prompt = "A photo of a cat"
    image = model.generate(prompt)
    model.show(image)[0]
            