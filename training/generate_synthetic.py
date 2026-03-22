"""合成编辑对生成脚本

从 Pexels 视频中提取帧，施加不同强度的编辑，自动标注质量分。
输出：training/data/synthetic/frames/ + labels.json

使用方法：
  python training/generate_synthetic.py
"""

import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# 路径
BASE_DIR = Path(__file__).parent
VIDEO_DIR = BASE_DIR / "data" / "pexels_videos"
OUTPUT_DIR = BASE_DIR / "data" / "synthetic"
FRAMES_DIR = OUTPUT_DIR / "frames"
LABELS_PATH = OUTPUT_DIR / "labels.json"

# 每段视频提取帧的间隔（秒）
FRAME_INTERVAL = 3.0
# 每帧生成的编辑变体数
EDITS_PER_FRAME = 15


def extract_frames(video_path: str, interval: float = 3.0) -> list[np.ndarray]:
    """从视频中按间隔提取帧"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total <= 0:
        cap.release()
        return []

    frame_interval = int(fps * interval)
    frames = []
    for i in range(0, total, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)
    cap.release()
    return frames


def apply_edit(img: Image.Image, edit_type: str, params: dict) -> Image.Image:
    """对 PIL 图片应用编辑"""
    result = img.copy()

    if edit_type == "brightness":
        factor = params["factor"]  # 0.5-1.5
        result = ImageEnhance.Brightness(result).enhance(factor)

    elif edit_type == "contrast":
        factor = params["factor"]  # 0.5-2.0
        result = ImageEnhance.Contrast(result).enhance(factor)

    elif edit_type == "saturation":
        factor = params["factor"]  # 0.5-2.0
        result = ImageEnhance.Color(result).enhance(factor)

    elif edit_type == "sharpness":
        factor = params["factor"]  # 0.5-3.0
        result = ImageEnhance.Sharpness(result).enhance(factor)

    elif edit_type == "blur":
        radius = params["radius"]  # 0.5-5.0
        result = result.filter(ImageFilter.GaussianBlur(radius=radius))

    elif edit_type == "color_shift":
        # 色偏：调整 R/G/B 通道
        arr = np.array(result, dtype=np.float32)
        arr[:, :, 0] *= params.get("r_factor", 1.0)  # R
        arr[:, :, 1] *= params.get("g_factor", 1.0)  # G
        arr[:, :, 2] *= params.get("b_factor", 1.0)  # B
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        result = Image.fromarray(arr)

    elif edit_type == "combo":
        # 组合编辑：按顺序应用多个
        for sub_edit in params["edits"]:
            result = apply_edit(result, sub_edit["type"], sub_edit["params"])

    return result


def generate_edit_variants() -> list[dict]:
    """生成编辑变体列表（含质量标签）"""
    variants = []

    # ── 单项编辑（温和 → 好，极端 → 差）──

    # 亮度
    for factor, score in [(1.05, 0.75), (1.10, 0.70), (0.95, 0.75), (0.90, 0.65),
                           (1.25, 0.35), (0.70, 0.25), (1.40, 0.15), (0.55, 0.10)]:
        variants.append({
            "edit_type": "brightness", "params": {"factor": factor},
            "score": score, "label": f"brightness_{factor}",
        })

    # 对比度
    for factor, score in [(1.10, 0.75), (1.20, 0.65), (0.90, 0.70),
                           (1.50, 0.30), (0.60, 0.20), (2.00, 0.10)]:
        variants.append({
            "edit_type": "contrast", "params": {"factor": factor},
            "score": score, "label": f"contrast_{factor}",
        })

    # 饱和度
    for factor, score in [(1.10, 0.75), (1.20, 0.65), (0.90, 0.70),
                           (1.50, 0.30), (0.50, 0.20), (2.00, 0.10)]:
        variants.append({
            "edit_type": "saturation", "params": {"factor": factor},
            "score": score, "label": f"saturation_{factor}",
        })

    # 锐化
    for factor, score in [(1.3, 0.75), (1.5, 0.65), (2.0, 0.40),
                           (3.0, 0.15), (0.5, 0.30)]:
        variants.append({
            "edit_type": "sharpness", "params": {"factor": factor},
            "score": score, "label": f"sharpness_{factor}",
        })

    # 模糊（模拟质量退化）
    for radius, score in [(0.5, 0.40), (1.0, 0.25), (2.0, 0.10)]:
        variants.append({
            "edit_type": "blur", "params": {"radius": radius},
            "score": score, "label": f"blur_{radius}",
        })

    # 色偏
    variants.extend([
        {"edit_type": "color_shift", "params": {"r_factor": 1.1, "g_factor": 1.0, "b_factor": 0.9},
         "score": 0.55, "label": "warm_slight"},
        {"edit_type": "color_shift", "params": {"r_factor": 0.9, "g_factor": 1.0, "b_factor": 1.1},
         "score": 0.55, "label": "cool_slight"},
        {"edit_type": "color_shift", "params": {"r_factor": 1.3, "g_factor": 1.0, "b_factor": 0.7},
         "score": 0.15, "label": "warm_extreme"},
        {"edit_type": "color_shift", "params": {"r_factor": 0.7, "g_factor": 1.0, "b_factor": 1.3},
         "score": 0.15, "label": "cool_extreme"},
    ])

    # ── 组合编辑 ──

    variants.extend([
        # 好的组合
        {"edit_type": "combo", "params": {"edits": [
            {"type": "brightness", "params": {"factor": 1.05}},
            {"type": "contrast", "params": {"factor": 1.05}},
            {"type": "saturation", "params": {"factor": 1.10}},
        ]}, "score": 0.80, "label": "combo_natural_enhance"},

        {"edit_type": "combo", "params": {"edits": [
            {"type": "brightness", "params": {"factor": 0.98}},
            {"type": "contrast", "params": {"factor": 1.10}},
            {"type": "saturation", "params": {"factor": 0.90}},
        ]}, "score": 0.80, "label": "combo_cinematic"},

        {"edit_type": "combo", "params": {"edits": [
            {"type": "brightness", "params": {"factor": 1.08}},
            {"type": "sharpness", "params": {"factor": 1.3}},
        ]}, "score": 0.75, "label": "combo_brighten_sharpen"},

        # 差的组合
        {"edit_type": "combo", "params": {"edits": [
            {"type": "brightness", "params": {"factor": 1.25}},
            {"type": "contrast", "params": {"factor": 1.50}},
            {"type": "sharpness", "params": {"factor": 2.5}},
        ]}, "score": 0.10, "label": "combo_over_processed"},

        {"edit_type": "combo", "params": {"edits": [
            {"type": "saturation", "params": {"factor": 1.50}},
            {"type": "color_shift", "params": {"r_factor": 1.2, "g_factor": 1.0, "b_factor": 0.8}},
        ]}, "score": 0.15, "label": "combo_oversaturated_warm"},
    ])

    return variants


def main():
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    if not videos:
        print(f"错误：{VIDEO_DIR} 中没有找到视频文件")
        sys.exit(1)

    print(f"找到 {len(videos)} 段视频")

    all_variants = generate_edit_variants()
    print(f"编辑变体池：{len(all_variants)} 种")

    labels = []  # 所有三元组 (原始帧, 编辑帧, 质量分)
    total_frames = 0
    total_pairs = 0

    for vi, video_path in enumerate(videos):
        print(f"\n[{vi+1}/{len(videos)}] 处理: {video_path.name}")

        frames = extract_frames(str(video_path), interval=FRAME_INTERVAL)
        if not frames:
            print("  跳过（无法读取）")
            continue

        print(f"  提取 {len(frames)} 帧")
        video_id = video_path.stem  # e.g. "pexels_12345_1920x1080"

        for fi, frame_bgr in enumerate(frames):
            # 保存原始帧
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            orig_filename = f"{video_id}_f{fi}_original.jpg"
            orig_path = FRAMES_DIR / orig_filename
            pil_img.save(str(orig_path), quality=95)
            total_frames += 1

            # 从变体池中随机选取 N 个
            selected = random.sample(all_variants, min(EDITS_PER_FRAME, len(all_variants)))

            for variant in selected:
                try:
                    edited_img = apply_edit(pil_img, variant["edit_type"], variant["params"])
                except Exception as e:
                    continue

                edit_filename = f"{video_id}_f{fi}_{variant['label']}.jpg"
                edit_path = FRAMES_DIR / edit_filename
                edited_img.save(str(edit_path), quality=95)

                labels.append({
                    "original": orig_filename,
                    "edited": edit_filename,
                    "score": variant["score"],
                    "edit_label": variant["label"],
                    "video": video_path.name,
                })
                total_pairs += 1

        if (vi + 1) % 10 == 0:
            print(f"  进度：{vi+1}/{len(videos)} 视频，{total_frames} 帧，{total_pairs} 编辑对")

    # 保存标签
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "total_pairs": total_pairs,
            "total_original_frames": total_frames,
            "edit_variants": len(all_variants),
            "pairs": labels,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"完成！")
    print(f"  原始帧：{total_frames}")
    print(f"  编辑对：{total_pairs}")
    print(f"  帧图片目录：{FRAMES_DIR}")
    print(f"  标签文件：{LABELS_PATH}")


if __name__ == "__main__":
    main()
