import argparse
import json
import os

import jittor as jt
import jittor.nn as nn
from diffusers import StableDiffusionPipeline
from einops import rearrange
from PIL import Image
from bg_utils import add_noise, latent2image, set_seeds

parser = argparse.ArgumentParser()
parser.add_argument(
    "--pretrained_model_name_or_path",
    type=str,
    default="../logs/background",
)
parser.add_argument(
    "--save_path",
    type=str,
    default='validation/ours',
)
parser.add_argument(
    "--val_path",
    type=str,
    default='../data/background/val/metadata.jsonl',
)
parser.add_argument(
    "--seed",
    type=int,
    default=11,
) 
parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
) 
args = parser.parse_args()

# 启用CUDA
jt.flags.use_cuda = 1

# load model
model_path = args.pretrained_model_name_or_path
# 注意：StableDiffusionPipeline需要适配Jittor
# 这里假设已经有了Jittor版本的Diffusers
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=jt.float16)
pipe.to("cuda")

seed = args.seed
set_seeds(seed)
print('using seed:', seed)
print('using model:', model_path)

# load validation data
val_path = args.val_path
save_dir = args.save_path
os.makedirs(save_dir, exist_ok=True)
batch_size = args.batch_size

def resize_mask(mask, size):
    """调整掩码大小"""
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    # Jittor的插值函数
    mask = nn.interpolate(mask.float(), size=size, mode="nearest")
    return mask.squeeze(0)


def inj_forward(degrate=1, no_attn_mask=None):
    """注入前向传播函数"""
    def forward(self, x, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, dim = x.shape
        h = self.heads
        q = self.to_q(x)  # (batch_size, 64*64, 320)
        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if is_cross else x
        k = self.to_k(encoder_hidden_states) # (batch_size, 77, 320)
        v = self.to_v(encoder_hidden_states)
        q = self.head_to_batch_dim(q)
        k = self.head_to_batch_dim(k)
        v = self.head_to_batch_dim(v)

        sim = jt.einsum("b i d, b j d -> b i j", q, k) * self.scale

        if attention_mask is not None:
            attention_mask = attention_mask.reshape(batch_size, -1)
            max_neg_value = -jt.inf
            attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
            # Jittor的条件赋值
            sim = jt.where(~attention_mask, max_neg_value, sim)

        attn = nn.softmax(sim, dim=-1)
        # attention degradation
        if is_cross:
            attn = rearrange(attn, 'b (h w) t -> b t h w', h=int((attn.shape[1])**0.5))
            if degrate != 1 and no_attn_mask is not None:
                cur_mask = resize_mask(no_attn_mask, attn.shape[2:]).bool().squeeze()
                attn[:, :, cur_mask] *= degrate
            attn = rearrange(attn, 'b t h w -> b (h w) t')
        out = jt.einsum("b i j, b j d -> b i d", attn, v)
        out = self.batch_to_head_dim(out)
        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out
    return forward


@jt.no_grad()
def text2image(prompt, latent=None, negative_prompt=None, strength=0.8, guidance_scale=7.5, height=512, width=512):
    """文本到图像生成"""
    batch_size = len(prompt)
    if latent is None:
        latent = jt.randn((1, pipe.unet.in_channels, height // 8, width // 8))
        t_start = 0
    else:
        latent, t_start = add_noise(latent, pipe.scheduler, strength=strength)
    
    # encode prompt embeddings
    text_input = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    # 将PyTorch tensor转换为Jittor Var
    input_ids = jt.array(text_input.input_ids.numpy())
    text_embeddings = pipe.text_encoder(input_ids)[0]
    
    max_length = input_ids.shape[-1]
    
    if negative_prompt is None:
        negative_prompt = [''] * batch_size
    
    uncond_input = pipe.tokenizer(
        negative_prompt,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    
    uncond_input_ids = jt.array(uncond_input.input_ids.numpy())
    uncond_embeddings = pipe.text_encoder(uncond_input_ids)[0]
    
    context = jt.contrib.concat([uncond_embeddings, text_embeddings])
    latents = latent.expand(batch_size, pipe.unet.in_channels, height // 8, width // 8)
    latents = latents.astype(text_embeddings.dtype)
    
    pipe.scheduler.set_timesteps(num_inference_steps=50)
    timesteps = pipe.scheduler.timesteps
    
    for t in timesteps[t_start:]:
        input_latents = jt.contrib.concat([latents] * 2)
        input_latents = pipe.scheduler.scale_model_input(input_latents, t)
        noise_pred = pipe.unet(input_latents, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = jt.chunk(noise_pred, 2, dim=0)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        latents = pipe.scheduler.step(noise_pred, t, latents)["prev_sample"]

    image = latent2image(pipe.vae, latents)
    return image


# 辅助函数：将Jittor数组转换为PIL图像
def jt_array_to_pil(image_array):
    """将Jittor数组转换为PIL图像"""
    if isinstance(image_array, jt.Var):
        image_array = image_array.numpy()
    
    # 确保值在0-255范围内并转换为uint8
    if image_array.max() <= 1.0:
        image_array = (image_array * 255).clip(0, 255).astype(np.uint8)
    else:
        image_array = image_array.astype(np.uint8)
    
    # 调整维度顺序：CHW -> HWC
    if image_array.ndim == 3 and image_array.shape[0] in [1, 3, 4]:
        image_array = image_array.transpose(1, 2, 0)
    
    return Image.fromarray(image_array)


# generate validation set examples
items = []
cnt = 0
with open(val_path, 'r') as f:
    for line in f:
        cnt += 1
        item = json.loads(line)
        items.append(item)
        
        if len(items) == batch_size or cnt == 1000:
            prompts = ['A Background image of ' + item['text'] for item in items]
            try:
                images = text2image(prompt=prompts, negative_prompt=None)
                
                for i in range(len(images)):
                    # 保存图像
                    img = images[i]
                    if isinstance(img, jt.Var):
                        # 转换为numpy数组
                        img_np = img.numpy()
                        
                        # 调整维度顺序
                        if img_np.ndim == 3 and img_np.shape[0] == 3:
                            img_np = img_np.transpose(1, 2, 0)
                        
                        # 确保值范围正确
                        if img_np.max() <= 1.0:
                            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                        else:
                            img_np = img_np.astype(np.uint8)
                        
                        img_pil = Image.fromarray(img_np)
                    else:
                        img_pil = Image.fromarray(img)
                    
                    save_path = os.path.join(save_dir, items[i]['file_name'])
                    img_pil.save(save_path)
                    
            except Exception as e:
                print(f"Error generating image for batch {cnt//batch_size}: {e}")
                # 保存空白图像作为占位符
                for i in range(len(items)):
                    blank_img = Image.new('RGB', (512, 512), color='white')
                    save_path = os.path.join(save_dir, items[i]['file_name'])
                    blank_img.save(save_path)
            
            items = []
            
            # 进度显示
            if cnt % (batch_size * 10) == 0:
                print(f"Processed {cnt} items")

print(f"Generation completed. Total processed: {cnt}")