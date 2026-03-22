"""CLIP 特征预提取脚本

对 synthetic/frames 中的所有图片提取 CLIP ViT-B-32 特征并缓存。

使用方法：
  python training/extract_features.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BASE_DIR = Path(__file__).parent
FRAMES_DIR = BASE_DIR / "data" / "synthetic" / "frames"
LABELS_PATH = BASE_DIR / "data" / "synthetic" / "labels.json"
FEATURES_DIR = BASE_DIR / "features"


def main():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # 加载 CLIP 模型
    print("加载 CLIP ViT-B-32...")
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"  设备: {device}")

    # 加载标签
    with open(LABELS_PATH, "r") as f:
        data = json.load(f)
    pairs = data["pairs"]

    # 收集所有需要提取特征的文件名（去重）
    all_filenames = set()
    for p in pairs:
        all_filenames.add(p["original"])
        all_filenames.add(p["edited"])
    all_filenames = sorted(all_filenames)
    print(f"需要提取 {len(all_filenames)} 张图片的特征")

    # 检查已有缓存
    cache_path = FEATURES_DIR / "clip_features.npz"
    if cache_path.exists():
        existing = dict(np.load(str(cache_path), allow_pickle=True))
        print(f"  已有缓存 {len(existing)} 张")
    else:
        existing = {}

    # 找出需要新提取的
    to_extract = [f for f in all_filenames if f not in existing]
    print(f"  需要新提取 {len(to_extract)} 张")

    if not to_extract:
        print("所有特征已缓存，无需提取")
        return

    # 批量提取
    batch_size = 64
    features_dict = dict(existing)

    for i in range(0, len(to_extract), batch_size):
        batch_files = to_extract[i:i + batch_size]
        batch_images = []

        for fname in batch_files:
            img_path = FRAMES_DIR / fname
            if not img_path.exists():
                # 用零向量占位
                features_dict[fname] = np.zeros(512, dtype=np.float32)
                continue
            img = Image.open(str(img_path)).convert("RGB")
            batch_images.append((fname, preprocess(img)))

        if not batch_images:
            continue

        names, tensors = zip(*batch_images)
        batch_tensor = torch.stack(tensors).to(device)

        with torch.no_grad():
            feats = model.encode_image(batch_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats = feats.cpu().numpy()

        for name, feat in zip(names, feats):
            features_dict[name] = feat.astype(np.float32)

        done = min(i + batch_size, len(to_extract))
        print(f"  提取进度: {done}/{len(to_extract)}", end="\r")

    # 保存
    np.savez_compressed(str(cache_path), **features_dict)
    print(f"\n特征缓存已保存: {cache_path} ({len(features_dict)} 张)")


if __name__ == "__main__":
    main()
