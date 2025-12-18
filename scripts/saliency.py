import os
import sys
from pathlib import Path

import torch
import torchvision
from PIL import Image
from tqdm import tqdm
# Allow running the script directly by ensuring the project root is first on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from saliency.basnet import get_saliency_model, saliency_detect

if __name__ == '__main__':
    BASE = r'data/image'
    SALIENCY = r'data/saliency/'
    os.makedirs(SALIENCY, exist_ok=True)
    img_paths = os.listdir(BASE)
    model = get_saliency_model()
    # Large batches quickly exhaust GPU memory; keep the chunk size modest
    batch_size = 16
    img_list = []
    name_list = []
    with torch.no_grad():
        for img_path in tqdm(img_paths):
            base_path = os.path.join(BASE, img_path)
            img = torchvision.transforms.functional.to_tensor(Image.open(base_path)).unsqueeze(0)
            img_list.append(img)
            name_list.append(img_path)
            if len(img_list) == batch_size or img_path == img_paths[-1]:
                inp = torch.cat(img_list, dim=0).cuda()
                smap = saliency_detect(model, inp, threshold=None)
                for i in range(len(smap)):
                    smap_img = torchvision.transforms.functional.to_pil_image(smap[i].unsqueeze(0))
                    smap_img.save(os.path.join(SALIENCY, name_list[i]))
                img_list, name_list = [], []
