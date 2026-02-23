import os
import logging

from tqdm import tqdm
from PIL import Image
import numpy as np
from einops import rearrange, repeat
import jittor as jt
from jittor.nn import functional as F
from jittor.dataset import DataLoader
from jittor import optim
import wandb
from .dataset import Padding
from .utils import sample, seq_to_bbox
import sys 
sys.path.append("..") 
from saliency.basnet import get_saliency_model, saliency_detect

logger = logging.getLogger(__name__)

def pil_to_jt_tensor(image):
    arr = np.asarray(image).astype("float32") / 255.0
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    arr = arr.transpose(2, 0, 1)
    return jt.array(arr)

def jt_tensor_to_pil(tensor):
    arr = tensor.numpy()
    if arr.ndim == 3:
        arr = arr.transpose(1, 2, 0)
    arr = (arr * 255.0).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)

def to_device(x):
    if jt.flags.use_cuda:
        return x.cuda()
    return x

def module_to_device(module):
    if jt.flags.use_cuda:
        module.cuda()
    return module

def clip_grad_norm_(parameters, max_norm):
    total_norm = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = (p.grad ** 2).sum().sqrt()
        total_norm += float(param_norm)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in parameters:
            if p.grad is None:
                continue
            p.grad *= clip_coef

class MultiStepLR:
    def __init__(self, optimizer, milestones, gamma=0.1):
        self.optimizer = optimizer
        self.milestones = set(milestones)
        self.gamma = gamma
        self.last_epoch = 0

    def step(self):
        self.last_epoch += 1
        if self.last_epoch in self.milestones:
            self.optimizer.lr *= self.gamma

    def get_last_lr(self):
        return [self.optimizer.lr]

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_iters = 0
    final_iters = 0  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_dir = None
    samples_dir = None
    sample_every = 1
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class Eval:
    def __init__(self, encoder, decoder, test_dataset, config):
        self.encoder = encoder
        self.decoder = decoder
        self.test_dataset = test_dataset
        self.bos_token = test_dataset.bos_token
        self.eos_token = test_dataset.eos_token
        self.pad_token = test_dataset.pad_token
        self.config = config
        self.device = config.device
        module_to_device(self.decoder)
        module_to_device(self.encoder)
        self.max_elements_num = test_dataset.max_elements_num
        self.transform = Padding(test_dataset.max_length, test_dataset.vocab_size)

    def calculate_coverage(self, layout):
        L_layout = np.asarray(layout.convert("L"))
        color_white = np.where(L_layout == 255)[0].shape[0]
        return color_white

    def generate(self, image, saliency, category, generated_num=8):
        '''
        generate layout given background image and category
        '''
        with jt.no_grad():
            # extract background features
            img_feature = self.encoder.forward_features(image)
            if len(img_feature.shape) == 4: # resnet
                img_feature = rearrange(img_feature, 'b c h w -> b (h w) c')
            img_feature = repeat(img_feature, 'b p f -> (n b) p f', n=generated_num)
            
            if saliency is not None:
                saliency = (saliency.squeeze().numpy() * 255).astype(np.uint8)
                saliency = pil_to_jt_tensor(Image.fromarray(saliency, mode='L').convert('RGB'))
                saliency = self.encoder.forward_features(jt.unsqueeze(to_device(saliency), 0))
                if len(img_feature.shape) == 4: # resnet
                    saliency = rearrange(saliency, 'b c h w -> b (h w) c')
                saliency = repeat(saliency, 'b p f -> (n b) p f', n=generated_num)
            
            # construct sequence for layout generation
            cclass = self.test_dataset.component_class
            cat2idx = self.test_dataset.json_category_id_to_contiguous_id
            category_idx = []
            for cat in category:
                if cat not in cclass:
                    print(f'Invalid Class: {cat}, all the classes should be in {cclass.keys()}')
                    return
                category_idx.append(cat2idx[cclass[cat]])
            category_idx.append(self.eos_token)
            x_start = jt.full((generated_num, 1), self.bos_token, dtype=jt.int32) # 228 in webui
            category_seq = jt.zeros((generated_num, self.test_dataset.max_length), dtype=jt.int32)
            for i in range(len(category_idx)):
                category_seq[:, 5*i+1] = category_idx[i]
            
            # generate layout
            layouts = sample(self.decoder, x_start, img_feature, saliency, steps=self.test_dataset.max_length,
                        temperature=1.0, sample=True, top_k=5, only_label=True, gt=category_seq).detach().numpy()
            
            return layouts
        

    def eval(self, command):
        model = self.decoder
        raw_model = model.module if hasattr(self.decoder, "module") else model
        if self.config.decoder_path != None:
            raw_model.load_state_dict(jt.load(self.config.decoder_path))
            self.encoder.load_state_dict(jt.load(self.config.encoder_path))
        else:
            print("args model_path is None")
            return
        
        # if command["calculate_harmony"]:
        #     saliency_model = get_saliency_model()
        
        loader = DataLoader(self.test_dataset, shuffle=False,
                            batch_size=self.config.batch_size,
                            num_workers=self.config.num_workers)
        print("dataset length:", len(loader))

        pbar = tqdm(enumerate(loader), total=len(loader))
        results = []
        harmony = []
        total_box_num = 0
        color_white_num = 0
        names = self.test_dataset.component_class
        names = dict(zip(names.values(), names.keys()))

        with jt.no_grad():         
            for it, (x, y, img, saliency) in pbar:
                x_cond = to_device(x)
                img = to_device(img)
                saliency = to_device(saliency)

                if command["name"] == "category_generate":
                    img_feature = self.encoder.forward_features(img)
                    saliency = self.encoder.forward_features(saliency)
                    if len(img_feature.shape) == 4: # resnet
                        img_feature = rearrange(img_feature, 'b c h w -> b (h w) c')
                        saliency = rearrange(saliency, 'b c h w -> b (h w) c')
                    layouts = sample(model, x_cond.clone()[:, :1], img_feature, saliency=saliency, steps=self.test_dataset.max_length,
                                temperature=1.0, sample=True, top_k=5, only_label=True, gt=x_cond).detach().numpy()

                elif command["name"] == "real_image":
                    layouts = x_cond.detach().numpy()
                elif command["name"] == "reconstruction":
                    logits, _ = model(x_cond)
                    probs = F.softmax(logits, dim=-1)
                    _, y = jt.topk(probs, k=1, dim=-1)
                    layouts = jt.concat((x_cond[:, :1], y[:, :, 0]), dim=1).detach().numpy()
                else:
                    raise ValueError(f"{command['name']} dose not exist.")                    

                for i, layout in enumerate(layouts):
                    img_name = self.test_dataset.image["image_path"][i + it * self.config.batch_size].split("/")[-1]
                    if command["save_image"]:
                        # cur_img = T.functional.to_pil_image(img[i])
                        cur_img = Image.open(os.path.join('data/img_512', img_name))
                        layout_img = self.test_dataset.render(layout, cur_img, border=0, W=512, H=512)
                        layout_img.save(os.path.join(self.config.samples_dir, command["name"], img_name))

                    if command["calculate_coverage"]:
                        cur_img = jt_tensor_to_pil(img[i])
                        layout_img = self.test_dataset.render(layout, cur_img)
                        color_white = self.calculate_coverage(layout_img)
                        total_box_num += 1
                        color_white_num += color_white

                    if command["save_pkl"]:
                        box_and_label = self.test_dataset.render_normalized_layout(layout)
                        results.append(box_and_label)
                        
                    if command["calculate_harmony"]:
                        # saliency_map = saliency_detect(saliency_model, img[i].unsqueeze(0), threshold=1)
                        saliency_map = Image.open(os.path.join('../data/saliency', img_name)).resize((224, 224))
                        saliency_map = (np.array(saliency_map) > 1).astype(np.uint8)
                        cats, bbox = seq_to_bbox(layout, self.bos_token, self.eos_token, self.pad_token) # replace layout with x[i]
                        cats = cats - self.test_dataset.size 
                        res, area = 0, 0
                        for ele_id in range(len(cats)):
                            box = bbox[ele_id]
                            res += saliency_map[box[1]:box[1]+box[3], box[0]:box[0]+box[2]].sum()
                            area += (box[2] * box[3]).item()
                        if area != 0:
                            harmony.append(res / area)

        if command["calculate_coverage"]:
            Height = self.test_dataset.H
            Width = self.test_dataset.W
            coverage_rate = color_white_num / (total_box_num * Height * Width)
            print("coverage rate:", coverage_rate)
        
        if command["calculate_harmony"]:
            print("harmony is: ", sum(harmony) / len(harmony))

        if command["save_pkl"]:
            import pickle
            with open(self.config.evaluate_layout_path, 'wb') as fb:
                pickle.dump(results, fb)
        

class Trainer:

    def __init__(self, encoder, decoder, train_dataset, test_dataset, config):
        self.encoder = encoder
        self.decoder = decoder
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.iters = 0
        self.fixed_x = None
        self.fixed_y = None
        self.fixed_img = None
        self.fixed_saliency = None
        self.device=config.device
        module_to_device(self.encoder)
        module_to_device(self.decoder)
        self.bos_token = test_dataset.bos_token
        self.eos_token = test_dataset.eos_token
        self.pad_token = test_dataset.pad_token
        self.max_elements_num = test_dataset.max_elements_num
        self.transform = Padding(test_dataset.max_length, test_dataset.vocab_size)
        # self.evaler = Eval(encoder, decoder, test_dataset, config)

    def save_checkpoint(self, epoch=None):
        encoder = self.encoder.module if hasattr(self.encoder, "module") else self.encoder
        decoder = self.decoder.module if hasattr(self.decoder, "module") else self.decoder
        if epoch is not None:
            ckpt_encoder = os.path.join(self.config.ckpt_dir, 'encoder_%d.pth' % epoch)
            ckpt_decoder = os.path.join(self.config.ckpt_dir, 'decoder_%d.pth' % epoch)
        else:
            ckpt_encoder = os.path.join(self.config.ckpt_dir, 'encoder.pth')
            ckpt_decoder = os.path.join(self.config.ckpt_dir, 'decoder.pth')
        jt.save(encoder.state_dict(), ckpt_encoder)
        jt.save(decoder.state_dict(), ckpt_decoder)


    def train(self):
        encoder, decoder, config = self.encoder, self.decoder, self.config
        raw_decoder = decoder.module if hasattr(self.decoder, "module") else decoder
        raw_encoder = encoder.module if hasattr(self.encoder, "module") else encoder

        if self.config.decoder_path != None:
            raw_decoder.load_state_dict(jt.load(self.config.decoder_path))

        optimizer_encoder = optim.AdamW(raw_encoder.parameters(), lr=self.config.learning_rate * 0.01, betas=self.config.betas)
        optimizer_decoder = raw_decoder.configure_optimizers(config)
        scheduler_encoder = MultiStepLR(optimizer_encoder, milestones=[30, 90, 210], gamma=0.1)
        scheduler_decoder = MultiStepLR(optimizer_decoder, milestones=[30, 90, 210], gamma=0.1)
        pad_token = self.train_dataset.vocab_size - 1

        def run_epoch(split):
            is_train = split == 'train'
            decoder.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)
            fixed_n = config.val_sample_num
            losses_decode = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y, img, saliency) in pbar:

                if epoch == 0 and not is_train and it == 0:
                    self.fixed_x = to_device(x[:min(fixed_n, len(x))])
                    self.fixed_y = to_device(y[:min(fixed_n, len(y))])
                    self.fixed_img = to_device(img[:min(fixed_n, len(y))])
                    self.fixed_saliency = to_device(saliency[:min(fixed_n, len(y))])

                # place data on the correct device
                x = to_device(x)
                y = to_device(y)
                img = to_device(img)
                saliency = to_device(saliency)

                if not is_train:
                    with jt.no_grad():
                        img_feature = raw_encoder.forward_features(img)
                        # saliency = raw_encoder.forward_features(saliency)
                        if len(img_feature.shape) == 4: # resnet
                            img_feature = rearrange(img_feature, 'b c h w -> b (h w) c')
                            # saliency = rearrange(saliency, 'b c h w -> b (h w) c')
                        logits, loss_decoder = decoder(img_feature=img_feature, saliency=None, idx=x, targets=y, pad_token=pad_token)
                        loss_decoder = loss_decoder.mean()  # collapse all losses if they are scattered on multiple gpus
                        losses_decode.append(loss_decoder.item())
                else:
                    img_feature = raw_encoder.forward_features(img)
                    # saliency = raw_encoder.forward_features(saliency)
                    if len(img_feature.shape) == 4: # resnet
                        img_feature = rearrange(img_feature, 'b c h w -> b (h w) c')
                        # saliency = rearrange(saliency, 'b c h w -> b (h w) c')
                    logits, loss_decoder = decoder(img_feature=img_feature, saliency=None, idx=x, targets=y, pad_token=pad_token)
                    loss_decoder = loss_decoder.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses_decode.append(loss_decoder.item())
                    encoder.zero_grad()
                    decoder.zero_grad()
                    loss_decoder.backward()
                    clip_grad_norm_(decoder.parameters(), config.grad_norm_clip)
                    optimizer_decoder.step()
                    optimizer_encoder.step()
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss_decoder.item():.5f}.")
                

            if not is_train:
                test_loss = float(np.mean(losses_decode))
                print("decoder loss: %.3f" % (float(np.mean(losses_decode))))
                if not self.config.debug:
                    wandb.log({
                        "decoder loss": float(np.mean(losses_decode)),
                        "learning rate": scheduler_decoder.get_last_lr()[0]
                    })

                return test_loss

        best_loss = float('inf')
        for epoch in range(config.max_epochs):
            run_epoch('train')
            if config.lr_decay:
                scheduler_encoder.step()
                scheduler_decoder.step()
            if self.test_dataset is not None:
                with jt.no_grad():
                    test_loss = run_epoch('test')

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_dir is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint()

            if epoch % 10 == 9:
                self.save_checkpoint(epoch)

            # sample from the model
            if self.config.samples_dir is not None and (epoch+1) % self.config.sample_every == 0:
                if (epoch+1) == self.config.sample_every:
                    layouts = self.fixed_x.detach().clone().numpy()
                    imgs = self.fixed_img.detach()
                    for i, layout in enumerate(layouts):
                        img = jt_tensor_to_pil(imgs[i])
                        layout = self.train_dataset.render(layout, img)
                        layout.save(os.path.join(self.config.samples_dir, f'input_{i:02d}.png'))
                
                img_feature = encoder.forward_features(self.fixed_img)
                saliency = encoder.forward_features(self.fixed_saliency)
                if len(img_feature.shape) == 4: # resnet
                    img_feature = rearrange(img_feature, 'b c h w -> b (h w) c')
                    saliency = rearrange(saliency, 'b c h w -> b (h w) c')
                layouts = sample(decoder, self.fixed_x[:, :1], img_feature, saliency, steps=self.train_dataset.max_length,
                                 temperature=1.0, sample=True, top_k=5, only_label=True, gt=self.fixed_x).detach().numpy()
                for i, layout in enumerate(layouts):
                    img = jt_tensor_to_pil(imgs[i])
                    layout = self.train_dataset.render(layout, img)
                    layout.save(os.path.join(self.config.samples_dir, 'category_generate', f'{epoch:02d}_{i:02d}.png'))
