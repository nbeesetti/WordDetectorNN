import argparse
import torch
import cv2
import numpy as np
from path import Path
from dataloader import DataLoaderImgFile
from eval import evaluate
from net import WordDetectorNet


def save_word_crops(orig_img_bgr, aabbs, out_dir, img_name):
    h, w = orig_img_bgr.shape[:2]
    if not aabbs:
        return
    heights = [aabb.ymax - aabb.ymin for aabb in aabbs]
    median_h = float(np.median(heights))
    threshold = median_h * 0.5
    boxes = []
    for aabb in aabbs:
        yc = (aabb.ymin + aabb.ymax) / 2.0
        boxes.append((yc, aabb.xmin, aabb))
    boxes.sort(key=lambda x: x[0])
    rows = []
    row_centers = []
    for yc, xmin, aabb in boxes:
        placed = False
        for i, rc in enumerate(row_centers):
            if abs(yc - rc) < threshold:
                rows[i].append(aabb)
                row_centers[i] = np.mean([(b.ymin + b.ymax) / 2.0 for b in rows[i]])
                placed = True
                break
        if not placed:
            rows.append([aabb])
            row_centers.append(yc)
    ordered = []
    row_centers, rows = zip(*sorted(zip(row_centers, rows), key=lambda x: x[0]))
    for row in rows:
        row_sorted = sorted(row, key=lambda b: b.xmin)
        ordered.extend(row_sorted)
    out_dir.makedirs_p()
    pad = 5
    for idx, aabb in enumerate(ordered):
        x0 = int(round(aabb.xmin)) - pad
        y0 = int(round(aabb.ymin)) - pad
        x1 = int(round(aabb.xmax)) + pad
        y1 = int(round(aabb.ymax)) + pad
        x0 = max(0, min(x0, w - 1))
        x1 = max(0, min(x1, w - 1))
        y0 = max(0, min(y0, h - 1))
        y1 = max(0, min(y1, h - 1))
        if x1 <= x0 or y1 <= y0:
            continue
        crop = orig_img_bgr[y0:y1, x0:x1]
        out_path = out_dir / f"{idx}.png"
        cv2.imwrite(str(out_path), crop)


def main(device="cpu", save_dir="word_crops", image_name=None):
    device = torch.device(device)
    net = WordDetectorNet()
    net.load_state_dict(torch.load("../model/weights", map_location=device))
    net.eval()
    net.to(device)
    loader = DataLoaderImgFile(Path("../data/test"), net.input_size, device)
    res = evaluate(net, loader, max_aabbs=1000)
    img_folder = Path("../data/test")
    files = [
        p for p in img_folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    base_save = Path("../" + save_dir)
    base_save.makedirs_p()

    def process_one(idx):
        img_path = files[idx]
        img_name = img_path.stem
        orig_img_bgr = cv2.imread(str(img_path))
        aabbs = res.batch_aabbs[idx]
        scale = loader.get_scale_factor(idx)
        scaled_boxes = [aabb.scale(1 / scale, 1 / scale) for aabb in aabbs]
        subdir = base_save / img_name
        save_word_crops(orig_img_bgr, scaled_boxes, subdir, img_name)

    if image_name is None:
        for i in range(len(files)):
            process_one(i)
    else:
        target = Path("../data/test") / image_name
        if target in files:
            idx = files.index(target)
            process_one(idx)
    print("Done")


if __name__ == "__main__":
    file_name = "random.jpg"  # specific name in /test folder or None (runs all)
    main("cpu", "word_crops", file_name)
