"""组合工具：需要多步操作的高级编辑"""

import logging

import cv2
import numpy as np

from src.models import SegmentMetadata
from src.executor.basic_tools import BasicTools

logger = logging.getLogger(__name__)


class CompoundTools:
    def __init__(self, basic_tools: BasicTools):
        self.basic = basic_tools

    def auto_color_harmonize(
        self,
        video_path: str,
        segments: list[SegmentMetadata],
    ) -> str:
        """
        全片色彩统一：
        1. 选取锚点片段（亮度最接近中位数的片段）
        2. 对其他片段做 Reinhard Color Transfer 近似（通过 FFmpeg 调色参数）
        3. 输出统一色彩的完整视频
        """
        if len(segments) <= 1:
            return video_path

        # 1. 选锚点：亮度最接近中位数
        brightnesses = [s.mean_brightness for s in segments]
        median_br = float(np.median(brightnesses))
        anchor_idx = int(np.argmin([abs(b - median_br) for b in brightnesses]))
        anchor = segments[anchor_idx]

        logger.info(f"色彩统一锚点: seg-{anchor.seg_id} (亮度={anchor.mean_brightness:.1f})")

        # 2. 对每个非锚点片段，计算需要的亮度/色温调整
        edited_segments = []
        for seg in segments:
            start, end = seg.time_range
            seg_path = self.basic.trim_segment(video_path, start, end)

            if seg.seg_id == anchor.seg_id:
                edited_segments.append(seg_path)
                continue

            # 亮度差异 → brightness 调整
            br_diff = (anchor.mean_brightness - seg.mean_brightness) / 255.0
            br_diff = max(-0.3, min(0.3, br_diff))

            # 色温差异 → 简单的色温调整
            ct_ratio = anchor.color_temp_est / (seg.color_temp_est + 1e-6)
            # 将比值映射到色温 K 值调整
            target_temp = 6500 * ct_ratio
            target_temp = max(2000, min(10000, target_temp))

            # 先调亮度再调色温
            if abs(br_diff) > 0.02:
                seg_path = self.basic.color_adjust(seg_path, brightness=br_diff)
            if abs(ct_ratio - 1.0) > 0.05:
                seg_path = self.basic.white_balance(seg_path, temperature=target_temp)

            edited_segments.append(seg_path)

        # 3. 拼接
        output_path = self.basic._make_output_path(video_path, "harmonized")
        return self.basic.concat_segments(edited_segments, output_path)

    def get_tool_registry(self) -> dict:
        return {
            "auto_color_harmonize": {
                "func": self.auto_color_harmonize,
                "description": "自动统一全片色彩风格",
                "params": ["segments"],
            },
        }
