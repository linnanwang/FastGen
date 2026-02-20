import torch
import numpy as np
from PIL import Image
from diffusers.pipelines.ltx2.export_utils import encode_video
from fastgen.networks.LTX2.network import LTX2

def test_ltx2_generation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 # LTX-2 is optimized for bfloat16
    
    # 1. Initialize the LTX2 model
    print("Initializing LTX-2 model...")
    model = LTX2(model_id="Lightricks/LTX-2", load_pretrained=True).to(device, dtype=dtype)
    model.init_preprocessors()
    model.eval()

    # 2. Prepare Prompts
    prompt = "A majestic dragon flying over a snowy mountain range, cinematic lighting, 4k"
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
    
    print(f"Encoding prompt: {prompt}")
    # Gemma 3 encoding
    condition = model.text_encoder.encode(prompt, precision=dtype).to(device)
    neg_condition = model.text_encoder.encode(negative_prompt, precision=dtype).to(device)

    # 3. Define Video Parameters
    # LTX-2 video dimensions must be divisible by 32 spatially and 8+1 temporally
    height, width = 480, 704
    num_frames = 81 # (8 * 10) + 1
    batch_size = 1
    
    # Calculate latent dimensions (VAE compresses 8x spatially and 8x temporally)
    latent_f = (num_frames - 1) // 8 + 1
    latent_h = height // 32
    latent_w = width // 32
    
    # 4. Generate Initial Noise
    # LTX-2 VAE latent channels is typically 128
    latent_channels = model.vae.vae.config.latent_channels
    noise = torch.randn(batch_size, latent_channels, latent_f, latent_h, latent_w, device=device, dtype=dtype)

    # 5. Run Sampling (Inference)
    print("Starting sampling process...")
    with torch.no_grad():
        latents = model.sample(
            noise=noise,
            condition=condition,
            neg_condition=neg_condition,
            guidance_scale=4.0,
            num_steps=40
        )

    # 6. Decode Latents to Video
    print("Decoding latents to video...")
    with torch.no_grad():
        # LTX-2 uses flow-matching, so we can pass a zero/final timestep if needed by the VAE
        video_tensor = model.vae.decode(latents)

    # 7. Post-process and Save
    # Video tensor is [B, C, F, H, W] in range [-1, 1]
    video_np = ((video_tensor[0].cpu().permute(1, 2, 3, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    
    print("Saving video to ltx2_test.mp4...")
    encode_video(
        video_np,
        fps=24,
        output_path="ltx2_test.mp4"
    )
    print("Done!")

if __name__ == "__main__":
    test_ltx2_generation()