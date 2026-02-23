import pickle
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import jittor as jt


def print_scores(score_dict):
    for k, v in score_dict.items():
        if k in ['Alignment', 'Overlap']:
            v = [_v * 100 for _v in v]
        if len(v) > 1:
            mean, std = np.mean(v), np.std(v)
            print(f'\t{k}: {mean:.2f} ({std:.2f})')
        else:
            print(f'\t{k}: {v[0]:.2f}')

def average(scores):
    return sum(scores) / len(scores)

def convert_xywh_to_ltrb(bbox):
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]

def compute_overlap(bbox, mask):
    # Attribute-conditioned Layout GAN
    # 3.6.3 Overlapping Loss

    bbox = jt.where(mask.unsqueeze(-1), bbox, jt.zeros_like(bbox))
    bbox = bbox.permute(2, 0, 1)

    l1, t1, r1, b1 = convert_xywh_to_ltrb(bbox.unsqueeze(-1))
    l2, t2, r2, b2 = convert_xywh_to_ltrb(bbox.unsqueeze(-2))
    a1 = (r1 - l1) * (b1 - t1)

    # intersection
    l_max = jt.maximum(l1, l2)
    r_min = jt.minimum(r1, r2)
    t_max = jt.maximum(t1, t2)
    b_min = jt.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = jt.where(cond, (r_min - l_max) * (b_min - t_max),
                     jt.zeros_like(a1[0]))

    diag_mask = jt.eye(a1.size(1)).astype("bool")
    ai = jt.where(diag_mask, jt.zeros_like(ai), ai)

    ar = ai / (a1 + 1e-6)

    return ar.sum(dim=(1, 2)) / mask.float().sum(-1)


def compute_alignment(bbox, mask):
    # Attribute-conditioned Layout GAN
    # 3.6.4 Alignment Loss

    bbox = bbox.permute(2, 0, 1)
    xl, yt, xr, yb = convert_xywh_to_ltrb(bbox)
    xc, yc = bbox[0], bbox[1]
    X = jt.stack([xl, xc, xr, yt, yc, yb], dim=1)

    X = X.unsqueeze(-1) - X.unsqueeze(-2)
    idx = jt.arange(X.size(2))
    X[:, :, idx, idx] = 1.
    X = X.abs().permute(0, 2, 1, 3)
    X[~mask] = 1.
    X = X.permute(0, 3, 2, 1)
    X[~mask] = 1.
    X = jt.min(X, dim=-1)[0]
    X = jt.min(X, dim=-1)[0]
    X = jt.where(X == 1.0, jt.zeros_like(X), X)

    X = -jt.log(1 - X + 1e-8)

    return X.sum(-1) / mask.float().sum(-1)

def dense_batch(bboxes, labels):
    lengths = [len(l) for l in labels]
    max_len = max(lengths) if lengths else 0
    batch = len(labels)
    bbox_pad = jt.zeros((batch, max_len, 4))
    label_pad = jt.zeros((batch, max_len), dtype=jt.int32)
    mask = jt.zeros((batch, max_len), dtype=jt.bool)
    for i in range(batch):
        n = lengths[i]
        if n == 0:
            continue
        bbox_pad[i, :n, :] = jt.array(bboxes[i], dtype=jt.float32)
        label_pad[i, :n] = jt.array(labels[i], dtype=jt.int32)
        mask[i, :n] = True
    return bbox_pad, label_pad, mask

def main(args):
    # generated layouts
    scores = defaultdict(list)
    for pkl_path in args.pkl_paths:
        alignment, overlap = [], []
        with Path(pkl_path).open('rb') as fb:
            generated_layouts = pickle.load(fb)

        for i in range(0, len(generated_layouts), args.batch_size):
            i_end = min(i + args.batch_size, len(generated_layouts))

            # get batch from data list
            batch_boxes = []
            batch_labels = []
            for b, l in generated_layouts[i:i_end]:
                batch_boxes.append(b)
                batch_labels.append(l)
            bbox, label, mask = dense_batch(batch_boxes, batch_labels)

            alignment += compute_alignment(bbox, mask).tolist()
            overlap += compute_overlap(bbox, mask).tolist()

        alignment = average(alignment)
        overlap = average(overlap)

        scores['Alignment'].append(alignment)
        scores['Overlap'].append(overlap)

    print(f'Dataset: {args.dataset}')
    print_scores(scores)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='dataset name',
                        choices=['rico', 'publaynet', 'webui'])
    parser.add_argument('pkl_paths', type=str, nargs='+',
                        help='generated pickle path')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='input batch size')
    parser.add_argument('--compute_real', action='store_true')
    args = parser.parse_args()

    if jt.has_cuda:
        jt.flags.use_cuda = 1
    else:
        jt.flags.use_cuda = 0
    main(args)
