import torch
from diffusers import StableDiffusionPipeline

# 选择模型（两个都可以，换路径即可）
model_path = "/mnt/data/desigen/logs/background/final-5000"
# model_path = "/mnt/data/desigen/logs/background/final-10000"

pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16
).to("cuda")

# 五个“不同内容”的背景 prompt（非常适合 PPT）
prompts = [
    "A background image of a quiet forest with soft light",
    "A background image of a modern city street at night",
    "A background image of a clean indoor room with natural light",
    "A background image of a beach under a clear blue sky",
    "A background image of a minimalist office workspace"
]

images = pipe(
    prompts,
    num_inference_steps=30,
    guidance_scale=7.5
).images

# 保存
for i, img in enumerate(images):
    img.save(f"final_5000_diff_{i}.png")

print("✅ 5 different images generated.")
