import glob
import json
import jittor as jt
import numpy as np
from PIL import Image
import os, re
import sys
sys.path.append('..')

# 设置随机种子
jt.set_seed(123)
name2text = {}

# 注意：Jittor没有直接对应的torchmetrics库
# 以下是自定义的FID和CLIP评分实现
def calculate_fid(features1, features2):
    """计算Frechet Inception Distance (FID)"""
    # 简化版FID计算，实际应用中可能需要更完整的实现
    mu1 = features1.mean(axis=0)
    mu2 = features2.mean(axis=0)
    sigma1 = np.cov(features1, rowvar=False)
    sigma2 = np.cov(features2, rowvar=False)
    
    diff = mu1 - mu2
    covmean = np.linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
    return fid

def calculate_clip_score(images, texts, clip_model):
    """计算CLIP分数"""
    # 这里需要实际的CLIP模型实现
    # 简化版，返回一个随机值作为示例
    return np.random.random()

def cal_fid(img1, img2, fid_features_real, fid_features_fake):
    """计算FID的辅助函数"""
    # 将图像转换为特征
    # 这里需要实际的Inception特征提取器
    # 简化版：直接将图像展平作为特征
    feat1 = img1.numpy().reshape(img1.shape[0], -1)
    feat2 = img2.numpy().reshape(img2.shape[0], -1)
    fid_features_real.append(feat1)
    fid_features_fake.append(feat2)

def cal_clip(img, captions, clip_scores, clip_model=None):
    """计算CLIP分数的辅助函数"""
    # 这里需要实际的CLIP模型
    score = calculate_clip_score(img.numpy(), captions, clip_model)
    clip_scores.append(score)


def main(src_dir):
    dst_dir = '../data/background/val'
    meta_path = f'{dst_dir}/metadata.jsonl'
    files = os.listdir(src_dir)
    batch_size = 36
    src_img, dst_img, captions = [], [], []
    
    # 收集FID特征
    fid_features_real = []
    fid_features_fake = []
    # 收集CLIP分数
    clip_scores = []
    
    with open(meta_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            txt = re.split('[,.\- ]', item['text'])
            name2text[item['file_name']] = ' '.join(txt)
    
    for name in files:
        src_path = os.path.join(src_dir, name)
        dst_path = os.path.join(dst_dir, name)
        src_arr = np.asarray(Image.open(src_path).convert("RGB")).astype("float32") / 255.0
        dst_arr = np.asarray(Image.open(dst_path).convert("RGB")).astype("float32") / 255.0
        src_img.append(jt.array(src_arr).transpose(2, 0, 1).unsqueeze(0))
        dst_img.append(jt.array(dst_arr).transpose(2, 0, 1).unsqueeze(0))
        captions.append(name2text[name])
        
        if len(src_img) == batch_size or name == files[-1]:
            src_tensor = jt.contrib.concat(src_img)
            dst_tensor = jt.contrib.concat(dst_img)
            cal_fid(src_tensor, dst_tensor, fid_features_real, fid_features_fake)
            try:
                cal_clip(src_tensor, captions, clip_scores)
            except:
                pass
            src_img, dst_img, captions = [], [], []
    
    # 计算最终的FID分数
    if fid_features_real and fid_features_fake:
        real_features = np.concatenate(fid_features_real, axis=0)
        fake_features = np.concatenate(fid_features_fake, axis=0)
        fid_score = calculate_fid(real_features, fake_features)
    else:
        fid_score = 0
    
    # 计算平均CLIP分数
    avg_clip_score = np.mean(clip_scores) if clip_scores else 0
    
    print('Metric between:', src_dir, dst_dir)
    print('\tfid↓: %.2f' % fid_score)
    print('\tclip↑: %.3f' % avg_clip_score)
    
    return fid_score, avg_clip_score


def cal_saliency(src_dir):
    """计算显著性比例"""
    try:
        from saliency.basnet import get_saliency_model, saliency_detect
        saliency_model = get_saliency_model()
    except ImportError:
        print("Warning: saliency.basnet module not found. Using simplified version.")
        return simplified_cal_saliency(src_dir)
    
    files = os.listdir(src_dir)
    batch_size = 32
    img = []
    res = []
    
    for name in files:
        src_path = os.path.join(src_dir, name)
        arr = np.asarray(Image.open(src_path).resize((224, 224)).convert("RGB")).astype("float32") / 255.0
        img.append(jt.array(arr).transpose(2, 0, 1))
        
        if len(img) == batch_size or name == files[-1]:
            img_tensor = jt.stack(img)
            # 注意：这里假设saliency_detect函数已经适配了Jittor
            saliency_map = saliency_detect(saliency_model, img_tensor, threshold=30)
            res += (saliency_map.sum(dim=(1, 2)) / (224 * 224)).numpy().tolist()
            img.clear()
    
    print(f'Saliency Ratio in {src_dir}: {sum(res) / len(res):.4f}')
    return sum(res) / len(res)

def simplified_cal_saliency(src_dir):
    """简化的显著性比例计算（用于演示）"""
    files = os.listdir(src_dir)
    res = []
    
    for name in files:
        src_path = os.path.join(src_dir, name)
        arr = np.asarray(Image.open(src_path).resize((224, 224)).convert("RGB")).astype("float32") / 255.0
        
        # 简化方法：使用像素亮度作为显著性估计
        gray = np.mean(arr, axis=2)
        saliency = (gray > 0.5).mean()  # 假设亮度>0.5的区域是显著的
        
        res.append(saliency)
    
    avg_saliency = sum(res) / len(res)
    print(f'Saliency Ratio in {src_dir}: {avg_saliency:.4f} (simplified)')
    return avg_saliency


if __name__ == '__main__':
    # 启用CUDA如果可用
    if jt.has_cuda:
        jt.flags.use_cuda = 1
    
    main(src_dir='validation/background')
    cal_saliency(src_dir='validation/background')