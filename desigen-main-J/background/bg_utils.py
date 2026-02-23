import numpy as np
import jittor as jt

def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image

def add_noise(init_latents, scheduler, noise=None, strength=0.8, num_inference_steps=50):
    batch_size = init_latents.shape[0]
    offset = scheduler.config.get("steps_offset", 0)
    init_timestep = int(num_inference_steps * strength) + offset
    init_timestep = min(init_timestep, num_inference_steps)

    timesteps = scheduler.timesteps[-init_timestep]
    timesteps = jt.array([timesteps] * batch_size)

    # add noise to latents using the timesteps
    if noise == None:
        noise = jt.randn(init_latents.shape)
    init_latents = scheduler.add_noise(init_latents, noise, timesteps)
    
    t_start = max(num_inference_steps - init_timestep + offset, 0)
    return init_latents, t_start


def set_seeds(seed):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    
