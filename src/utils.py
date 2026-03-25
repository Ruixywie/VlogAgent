"""共享工具函数：帧编码、Storyboard 构建、VLM 消息构建"""

import base64
import io
import math
from pathlib import Path

from PIL import Image

from src.models import SegmentMetadata


def encode_image(image_path: str) -> str:
    """将图片文件编码为 base64 字符串"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_storyboard(
    segments: list[SegmentMetadata],
    frames_per_segment: int = 2,
    thumb_width: int = 320,
    thumb_height: int = 180,
    max_cols: int = 4,
) -> list[tuple[str, str]]:
    """
    将多段关键帧拼成 Storyboard 网格图。

    每段取 frames_per_segment 帧（首尾），拼成一张大图。
    相比逐帧发送，token 消耗减少 10-15 倍。

    返回：[(storyboard_描述文字, base64编码), ...]
    每张 Storyboard 最多 max_cols × N 行，如果帧太多则分成多张。
    """
    # 收集所有要拼的帧
    frames_info = []  # [(seg_id, time_range, PIL.Image), ...]
    for seg in segments:
        if not seg.keyframe_paths:
            continue
        # 取首尾帧
        paths_to_use = []
        if len(seg.keyframe_paths) >= 2:
            paths_to_use = [seg.keyframe_paths[0], seg.keyframe_paths[-1]]
        elif len(seg.keyframe_paths) == 1:
            paths_to_use = [seg.keyframe_paths[0]]

        for kf_path in paths_to_use[:frames_per_segment]:
            if Path(kf_path).exists():
                try:
                    img = Image.open(kf_path).convert("RGB")
                    img = img.resize((thumb_width, thumb_height), Image.LANCZOS)
                    frames_info.append((seg.seg_id, seg.time_range, img))
                except Exception:
                    continue

    if not frames_info:
        return []

    # 拼成网格
    n_frames = len(frames_info)
    n_cols = min(n_frames, max_cols)
    n_rows = math.ceil(n_frames / n_cols)

    # 留 20px 给标注
    label_height = 20
    cell_height = thumb_height + label_height
    grid_width = n_cols * thumb_width
    grid_height = n_rows * cell_height

    grid = Image.new("RGB", (grid_width, grid_height), (30, 30, 30))

    # 加载字体（PIL 默认）
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for idx, (seg_id, time_range, img) in enumerate(frames_info):
        col = idx % n_cols
        row = idx // n_cols
        x = col * thumb_width
        y = row * cell_height

        # 贴帧
        grid.paste(img, (x, y))
        # 加标注
        label = f"seg-{seg_id} ({time_range[0]:.1f}s-{time_range[1]:.1f}s)"
        draw.text((x + 4, y + thumb_height + 2), label, fill=(200, 200, 200), font=font)

    # 编码为 base64
    buf = io.BytesIO()
    grid.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    desc = f"Storyboard: {n_frames} 帧 ({n_cols}×{n_rows} 网格)"
    return [(desc, b64)]


def build_frames_content(
    segments: list[SegmentMetadata],
    mode: str = "storyboard",
) -> list[dict]:
    """
    构建 VLM 多模态消息内容。

    mode="storyboard": 拼成网格图（推荐，token 少）
    mode="individual": 逐帧发送（token 多，但细节更清晰）
    """
    if mode == "storyboard":
        storyboards = build_storyboard(segments)
        if storyboards:
            content = []
            for desc, b64 in storyboards:
                content.append({"type": "text", "text": desc})
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "high",  # storyboard 用 high detail 看清细节
                    },
                })
            # 加上各段时间范围的文字标注
            seg_labels = "\n".join(
                f"- seg-{s.seg_id}: {s.time_range[0]:.1f}s - {s.time_range[1]:.1f}s"
                for s in segments
            )
            content.append({"type": "text", "text": f"\n各片段时间范围：\n{seg_labels}"})
            return content

    # fallback: 逐帧发送（每段只取中间帧，最多 1 帧/段）
    content = []
    for seg in segments:
        content.append({
            "type": "text",
            "text": f"--- 片段 {seg.seg_id} ({seg.time_range[0]:.1f}s - {seg.time_range[1]:.1f}s) ---",
        })
        if seg.keyframe_paths:
            mid_idx = len(seg.keyframe_paths) // 2
            kf_path = seg.keyframe_paths[mid_idx]
            if Path(kf_path).exists():
                b64 = encode_image(kf_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "low",
                    },
                })
    return content


def build_comparison_storyboard(
    original_frames: list[tuple[str, str]],
    edited_frames: list[tuple[str, str]],
    thumb_width: int = 320,
    thumb_height: int = 180,
) -> list[tuple[str, str]]:
    """
    构建编辑前后对比 Storyboard。

    每行两张图：左边原始，右边编辑后。
    用于 Critic 评审，比逐段发两张图 token 少很多。

    参数：
        original_frames: [(label, base64), ...]
        edited_frames: [(label, base64), ...]
    返回：[(描述, base64), ...]
    """
    from PIL import ImageDraw, ImageFont

    n_pairs = min(len(original_frames), len(edited_frames))
    if n_pairs == 0:
        return []

    label_height = 25
    cell_height = thumb_height + label_height
    grid_width = thumb_width * 2 + 10  # 中间留 10px 间隔
    grid_height = n_pairs * cell_height

    grid = Image.new("RGB", (grid_width, grid_height), (30, 30, 30))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i in range(n_pairs):
        orig_label, orig_b64 = original_frames[i]
        edit_label, edit_b64 = edited_frames[i]

        y = i * cell_height

        # 解码并贴原始帧
        try:
            orig_img = Image.open(io.BytesIO(base64.b64decode(orig_b64))).convert("RGB")
            orig_img = orig_img.resize((thumb_width, thumb_height), Image.LANCZOS)
            grid.paste(orig_img, (0, y))
        except Exception:
            pass

        # 解码并贴编辑后帧
        try:
            edit_img = Image.open(io.BytesIO(base64.b64decode(edit_b64))).convert("RGB")
            edit_img = edit_img.resize((thumb_width, thumb_height), Image.LANCZOS)
            grid.paste(edit_img, (thumb_width + 10, y))
        except Exception:
            pass

        # 标注
        draw.text((4, y + thumb_height + 2), f"原始 | {orig_label}", fill=(150, 200, 150), font=font)
        draw.text((thumb_width + 14, y + thumb_height + 2), f"编辑后 | {edit_label}", fill=(200, 150, 150), font=font)

    buf = io.BytesIO()
    grid.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return [(f"对比 Storyboard: {n_pairs} 段", b64)]


def build_metrics_text(segments: list[SegmentMetadata]) -> str:
    """构建技术指标文本摘要"""
    lines = []
    for seg in segments:
        lines.append(
            f"片段 {seg.seg_id} ({seg.time_range[0]:.1f}s-{seg.time_range[1]:.1f}s): "
            f"亮度={seg.mean_brightness:.0f}/255, "
            f"色温R/B比={seg.color_temp_est:.2f}, "
            f"清晰度={seg.sharpness_score:.0f}, "
            f"稳定性={seg.stability_score:.2f}(光流方差), "
            f"噪声={seg.noise_level:.1f}"
            + (f", 语音: \"{seg.speech_text[:60]}\"" if seg.has_speech else "")
        )
    return "\n".join(lines)
