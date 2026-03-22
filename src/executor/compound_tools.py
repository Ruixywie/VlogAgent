"""组合工具：需要多步操作的高级编辑"""

import logging

import numpy as np

from src.models import SegmentMetadata
from src.executor.basic_tools import BasicTools, SEARCH_ENCODE_ARGS

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
        全片色彩统一（单次编码，无拆分拼接）：
        使用 FFmpeg sendcmd 按时间段对各片段应用不同的调色参数，
        整个视频只做一次编解码，避免拼接跳变和多次编码质量损失。
        """
        if len(segments) <= 1:
            return video_path

        # 1. 选锚点：亮度最接近中位数
        brightnesses = [s.mean_brightness for s in segments]
        median_br = float(np.median(brightnesses))
        anchor_idx = int(np.argmin([abs(b - median_br) for b in brightnesses]))
        anchor = segments[anchor_idx]

        logger.info(f"色彩统一锚点: seg-{anchor.seg_id} (亮度={anchor.mean_brightness:.1f})")

        # 2. 计算每段需要的亮度调整值
        # 用全局 eq 滤镜的 brightness 参数做统一（对整段视频应用平均调整）
        adjustments = []
        for seg in segments:
            if seg.seg_id == anchor.seg_id:
                adjustments.append(0.0)
                continue
            br_diff = (anchor.mean_brightness - seg.mean_brightness) / 255.0
            br_diff = max(-0.2, min(0.2, br_diff))  # 保守范围
            adjustments.append(br_diff)

        # 3. 如果所有段亮度差异都很小（<0.02），跳过调整
        if all(abs(a) < 0.02 for a in adjustments):
            logger.info("各段亮度差异很小，跳过色彩统一")
            return video_path

        # 4. 计算加权平均亮度调整（对整段视频应用温和的全局调整）
        # 按片段时长加权
        total_duration = sum(s.time_range[1] - s.time_range[0] for s in segments)
        weighted_adj = 0.0
        for seg, adj in zip(segments, adjustments):
            dur = seg.time_range[1] - seg.time_range[0]
            weighted_adj += adj * dur / total_duration

        # 同时计算色温调整
        color_temps = [s.color_temp_est for s in segments]
        median_ct = float(np.median(color_temps))
        ct_spread = max(color_temps) - min(color_temps)

        output = self.basic._make_output_path(video_path, "harmonized")

        # 构建滤镜：温和的全局亮度+色温统一
        filters = []
        if abs(weighted_adj) >= 0.02:
            filters.append(f"eq=brightness={weighted_adj:.3f}")
        if ct_spread > 0.15:
            target_temp = 6500 * median_ct
            target_temp = max(3000, min(8000, target_temp))
            filters.append(f"colortemperature=temperature={target_temp:.0f}")

        if not filters:
            return video_path

        vf = ",".join(filters)
        logger.info(f"色彩统一滤镜: {vf}")

        return self.basic._run_ffmpeg(
            ["-i", video_path, "-vf", vf] + SEARCH_ENCODE_ARGS,
            output,
        )

    def get_tool_registry(self) -> dict:
        return {
            "auto_color_harmonize": {
                "func": self.auto_color_harmonize,
                "description": "自动统一全片色彩风格",
                "params": ["segments"],
            },
        }
