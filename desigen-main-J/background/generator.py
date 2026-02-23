import jittor as jt
import numpy as np
from PIL import Image
from .bg_utils import latent2image, set_seeds
import jittor.nn as nn
from diffusers import StableDiffusionPipeline
from einops import rearrange

# 启用CUDA
jt.flags.use_cuda = 1

SEED = 888
RES = 16
DEGRATE = 1
set_seeds(SEED)
# Jittor中默认不启用梯度
# torch.set_grad_enabled(False) 对应 jt.flags.enable_grad = False
jt.flags.enable_grad = False

# Jittor中设备管理是自动的
model = StableDiffusionPipeline.from_pretrained("logs/background", safety_checker=None)
# 注意：StableDiffusionPipeline需要是Jittor版本
model.scheduler.set_timesteps(num_inference_steps=50)
attn_store = []
avg_store = []

def resize_mask(mask, size):
    """调整掩码大小"""
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    # Jittor的插值函数
    mask = nn.interpolate(mask.float(), size=size, mode="nearest")
    return mask.squeeze(0)


def inj_forward(degrate=DEGRATE, no_attn_mask=None):
    """注入前向传播函数"""
    def forward(self, x, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, dim = x.shape
        h = self.heads
        q = self.to_q(x)  # (batch_size, 64*64, 320)
        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if is_cross else x
        k = self.to_k(encoder_hidden_states)  # (batch_size, 77, 320)
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
        if is_cross:
            attn = rearrange(attn, 'b (h w) t -> b t h w', h=int((attn.shape[1])**0.5))
            if degrate != 1 and no_attn_mask is not None:
                cur_mask = resize_mask(no_attn_mask, attn.shape[2:]).bool().squeeze()
                attn[:, :, cur_mask] *= degrate
            # if attn.shape[2] == RES:
            #     attn_store.append(attn)  # ([16, 77, RES, RES])
            attn = rearrange(attn, 'b t h w -> b (h w) t')
        out = jt.einsum("b i j, b j d -> b i d", attn, v)
        out = self.batch_to_head_dim(out)
        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out
    
    return forward


def get_bg_generator(mask_image=None, degrate=0.1):
    """获取背景生成器"""
    global model
    if mask_image is not None:
        print(f'use mask in {mask_image}')
        # 加载并处理掩码图像
        mask_array = np.array(Image.open(mask_image).convert('L'), dtype=np.float32) / 255.0
        mask_image = jt.array(mask_array).unsqueeze(0)
        
        # 注入修改后的前向传播
        for _module in model.unet.modules():
            if _module.__class__.__name__ == "CrossAttention":
                _module.__class__.__call__ = inj_forward(degrate=degrate, no_attn_mask=mask_image)
    return model


def visualize_attn_map(prompt, res=RES):
    """可视化注意力图"""
    global avg_store, attn_store
    if not attn_store:
        return
    
    b, l, _, _ = attn_store[0].shape
    avg = jt.zeros((b, l, res, res))
    
    for attn in attn_store:
        if attn.shape[-1] == res:
            if attn.ndim == 4:
                avg += attn.squeeze(0)
            else:
                avg += attn
    
    avg /= len(attn_store)
    avg_store.append(avg.unsqueeze(0))
    
    # 注意：show_cross_attention函数需要适配Jittor
    # show_cross_attention(avg.mean(0).numpy(), prompt[0], model.tokenizer, name=f'attn/attention_{len(avg_store)}')
    
    return avg


@jt.no_grad()
def text2image(prompt, latent=None, negative_prompt=None, strength=0.8, 
              guidance_scale=7.5, height=512, width=512, no_attn_mask=None, degrate=1):
    """文本到图像生成"""
    global model, attn_store
    
    # 如果需要，注入修改后的前向传播
    if no_attn_mask is not None:
        for _module in model.unet.modules():
            if _module.__class__.__name__ == "CrossAttention":
                _module.__class__.__call__ = inj_forward(degrate=degrate, no_attn_mask=no_attn_mask)
    
    batch_size = len(prompt)
    
    # 初始化潜在变量
    if latent is None:
        latent = jt.randn((1, model.unet.in_channels, height // 8, width // 8))
        t_start = 0
    else:
        # 注意：add_noise函数需要适配Jittor
        # latent, t_start = add_noise(latent, model.scheduler, strength=strength)
        t_start = 0
    
    # 编码提示文本
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    # 将PyTorch tensor转换为Jittor Var
    input_ids = jt.array(text_input.input_ids.numpy())
    text_embeddings = model.text_encoder(input_ids)[0]
    
    max_length = input_ids.shape[-1]
    
    # 准备无条件提示
    if negative_prompt is None:
        negative_prompt = [''] * batch_size
    
    uncond_input = model.tokenizer(
        negative_prompt,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    
    uncond_input_ids = jt.array(uncond_input.input_ids.numpy())
    uncond_embeddings = model.text_encoder(uncond_input_ids)[0]
    
    # 组合条件和无条件嵌入
    context = jt.contrib.concat([uncond_embeddings, text_embeddings])
    
    # 扩展潜在变量
    latents = latent.expand(batch_size, model.unet.in_channels, height // 8, width // 8)
    latents = latents.astype(text_embeddings.dtype)
    
    # 设置时间步
    model.scheduler.set_timesteps(num_inference_steps=50)
    timesteps = model.scheduler.timesteps
    
    # 扩散过程
    for t in timesteps[t_start:]:
        # 准备输入
        input_latents = jt.contrib.concat([latents] * 2)
        input_latents = model.scheduler.scale_model_input(input_latents, t)
        
        # 预测噪声
        noise_pred = model.unet(input_latents, t, encoder_hidden_states=context)["sample"]
        
        # 分离条件和无条件预测
        noise_pred_uncond, noise_prediction_text = jt.chunk(noise_pred, 2, dim=0)
        
        # 应用引导
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        
        # 更新潜在变量
        latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    
    # 解码为图像
    image = latent2image(model.vae, latents)
    
    return image, latent


# 辅助函数：清理注意力存储
def clear_attn_store():
    """清理注意力存储"""
    global attn_store, avg_store
    attn_store.clear()
    avg_store.clear()


# 辅助函数：保存注意力图
def save_attention_maps(output_dir="attn_maps"):
    """保存注意力图"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for i, attn_map in enumerate(avg_store):
        if attn_map.ndim == 4:
            # 取平均并转换为numpy
            avg_attn = attn_map.mean(0).numpy()
            
            # 保存为图像
            for j in range(min(5, avg_attn.shape[0])):  # 保存前5个token
                token_attn = avg_attn[j]
                
                # 归一化
                if token_attn.max() > token_attn.min():
                    token_attn = (token_attn - token_attn.min()) / (token_attn.max() - token_attn.min())
                
                # 转换为0-255范围
                token_attn = (token_attn * 255).astype(np.uint8)
                
                # 创建PIL图像
                img = Image.fromarray(token_attn)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 调整大小以便查看
                img = img.resize((256, 256), Image.NEAREST)
                
                # 保存
                img.save(os.path.join(output_dir, f"attn_{i}_token_{j}.png"))


# 示例使用函数
def generate_with_mask(prompt, mask_path=None, degrate=0.1, **kwargs):
    """使用掩码生成图像"""
    # 获取生成器
    generator = get_bg_generator(mask_image=mask_path, degrate=degrate)
    
    # 生成图像
    image, latent = text2image(prompt=prompt, no_attn_mask=None, degrate=degrate, **kwargs)
    
    return image, latent


if __name__ == "__main__":
    # 测试代码
    test_prompt = ["a beautiful sunset over mountains"]
    
    # 生成不带掩码的图像
    print("Generating image without mask...")
    image, latent = text2image(prompt=test_prompt)
    print(f"Generated image shape: {image.shape if hasattr(image, 'shape') else 'Unknown'}")
    
    # 清理
    clear_attn_store()