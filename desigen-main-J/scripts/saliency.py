import os
import numpy as np
import jittor as jt
from PIL import Image
from tqdm import tqdm
from saliency.basnet import get_saliency_model, saliency_detect
from multiprocessing import Pool, cpu_count

def load_image(args):
    base_dir, img_path = args
    base_path = os.path.join(base_dir, img_path)
    arr = np.asarray(Image.open(base_path).convert("RGB")).astype("float32") / 255.0
    arr = arr.transpose(2, 0, 1)
    return img_path, arr

if __name__ == '__main__':
    BASE = r'data/image'
    SALIENCY = r'data/saliency/'
    os.makedirs(SALIENCY, exist_ok=True)
    img_paths = os.listdir(BASE)
    if jt.has_cuda:
        jt.flags.use_cuda = 1
    model = get_saliency_model()
    batch_size = 128
    img_list = []
    name_list = []
    num_workers = cpu_count()
    print(f"cpu_count: {num_workers}")
    with Pool(num_workers) as workers:
        with jt.no_grad():
            for img_path, arr in tqdm(workers.imap_unordered(load_image, [(BASE, p) for p in img_paths]), total=len(img_paths)):
                img = jt.array(arr)
                img_list.append(img)
                name_list.append(img_path)
                if len(img_list) == batch_size:
                    inp = jt.stack(img_list, dim=0)
                    smap = saliency_detect(model, inp, threshold=None)
                    for i in range(len(smap)):
                        smap_img = Image.fromarray((smap[i].numpy() * 255).astype("uint8"))
                        smap_img.save(os.path.join(SALIENCY, name_list[i]))
                    img_list, name_list = [], []
            if img_list:
                inp = jt.stack(img_list, dim=0)
                smap = saliency_detect(model, inp, threshold=None)
                for i in range(len(smap)):
                    smap_img = Image.fromarray((smap[i].numpy() * 255).astype("uint8"))
                    smap_img.save(os.path.join(SALIENCY, name_list[i]))
