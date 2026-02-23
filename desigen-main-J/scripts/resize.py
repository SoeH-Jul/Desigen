from PIL import Image
import os
import json
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool, cpu_count


data_path = r'data'
source_img_dir = os.path.join(data_path, 'raw')
target_img_dir = os.path.join(data_path, 'image')
meta_dir = os.path.join(data_path, 'meta')
os.makedirs(target_img_dir, exist_ok=True)
H, W = 512, 512


def get_image_size(meta_path):
    with open(meta_path, "r") as f:
        meta = json.load(f)

    for ele in meta['layout']:
        if ele['type'] == 'background':
            _, _, img_w, img_h = ele['position']
            return img_w, img_h
        
    return None

def resize_image(website):
    img_path = os.path.join(source_img_dir, website)
    website_name = os.path.splitext(website)[0]
    meta_path = os.path.join(meta_dir, website_name + '.json')
    img_w, img_h = get_image_size(meta_path)
    
    img = Image.open(img_path)
    if not img.mode == "RGB":
        img = img.convert("RGB")
    # two modes for resizing: based on width / based on height
    if int(img_w / img.width * img.height) >= img_h:
        new_h = int(img_w / img.width * img.height)
        new_w = img_w
    else:
        new_h = img_h
        new_w = int(img_h / img.height * img.width)
    img = img.resize((new_w, new_h), resample=Image.BILINEAR)
    left = max((new_w - img_w) // 2, 0)
    top = max((new_h - img_h) // 2, 0)
    img = img.crop((left, top, left + img_w, top + img_h))
    img = img.resize((W, H), resample=Image.BILINEAR)
    img.save(os.path.join(target_img_dir, website))
    

if __name__ == '__main__':
    websites = os.listdir(source_img_dir)
    num_workers = cpu_count()
    print(f"cpu_count: {num_workers}")
    with Pool(num_workers) as workers:
        with tqdm(total=len(websites)) as pbar:
            for i in workers.imap(resize_image, websites):
                pbar.update()
