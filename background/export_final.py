import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel

base_model = "/mnt/data/desigen/models/stable-diffusion-v1-4"
ckpt_dir   = "/mnt/data/desigen/logs/background/checkpoint-5000"
out_dir    = "/mnt/data/desigen/logs/background/final-5000"

# 1. 加载完整 base pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.float16
)

# 2. 加载 UNet（accelerate checkpoint）
unet_state = torch.load(f"{ckpt_dir}/pytorch_model.bin", map_location="cpu")
pipe.unet.load_state_dict(unet_state, strict=False)

# 3. 如果你训练了 text encoder（你用了 --train_text_encoder）
text_state = torch.load(f"{ckpt_dir}/pytorch_model_1.bin", map_location="cpu")
pipe.text_encoder.load_state_dict(text_state, strict=False)

# 4. 保存为标准 diffusers 模型
pipe.save_pretrained(out_dir)

print(f"Exported final model to {out_dir}")
