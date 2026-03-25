"""基础工具：FFmpeg / OpenCV 封装

设计原则：
- 搜索阶段直接在原视频上执行，结果仅用于评估
- 最终执行时滤镜合并为一次编码，使用视觉无损参数
- 搜索阶段用 SEARCH_ENCODE_ARGS（速度优先，结果会丢弃）
- 最终输出用 ENCODE_ARGS（质量优先，CRF=10 视觉无损）
"""

import os
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 搜索阶段编码参数（结果仅用于评估，速度优先）
SEARCH_ENCODE_ARGS = ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-c:a", "aac", "-b:a", "192k"]

# 最终输出编码参数（质量优先，CRF=10 视觉无损）
ENCODE_ARGS = ["-c:v", "libx264", "-preset", "slow", "-crf", "10", "-c:a", "aac", "-b:a", "320k"]


class BasicTools:
    def __init__(self, config: dict):
        self.ffmpeg = config.get("ffmpeg_path", "ffmpeg")
        self.ffprobe = config.get("ffprobe_path", "ffprobe")
        self.temp_dir = config.get("temp_dir", "output/temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        # 滤镜收集器：用于合并多个滤镜为一次 FFmpeg 调用
        self._pending_vfilters: list[str] = []
        self._pending_afilters: list[str] = []

    def _run_ffmpeg(self, args: list[str], output_path: str) -> str:
        """执行 FFmpeg 命令"""
        cmd = [self.ffmpeg, "-y"] + args + [output_path]
        logger.info(f"FFmpeg: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg 失败: {result.stderr[-500:]}")
        return output_path

    def _make_output_path(self, input_path: str, suffix: str) -> str:
        stem = Path(input_path).stem
        ext = Path(input_path).suffix
        return os.path.join(self.temp_dir, f"{stem}_{suffix}{ext}")

    # ══════════════════════════════════════════════════════
    # 滤镜链合并模式
    # ══════════════════════════════════════════════════════

    def collect_filter(self, vfilter: str):
        """收集一个视频滤镜，稍后统一应用"""
        self._pending_vfilters.append(vfilter)

    def apply_collected_filters(self, video_path: str, suffix: str = "edited") -> str:
        """一次性应用所有收集的滤镜（单次编码）"""
        if not self._pending_vfilters:
            return video_path

        output = self._make_output_path(video_path, suffix)
        vf = ",".join(self._pending_vfilters)
        args = ["-i", video_path, "-vf", vf] + ENCODE_ARGS

        if self._pending_afilters:
            af = ",".join(self._pending_afilters)
            args = ["-i", video_path, "-vf", vf, "-af", af] + [
                "-c:v", "libx264", "-preset", "slow", "-crf", "16"
            ]

        self._pending_vfilters.clear()
        self._pending_afilters.clear()
        return self._run_ffmpeg(args, output)

    def has_pending_filters(self) -> bool:
        return len(self._pending_vfilters) > 0

    # ══════════════════════════════════════════════════════
    # 单工具调用（搜索阶段在代理副本上用，速度优先）
    # ══════════════════════════════════════════════════════

    def color_adjust(
        self,
        video_path: str,
        brightness: float = 0.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        gamma: float = 1.0,
    ) -> str:
        output = self._make_output_path(video_path, "color")
        vf = f"eq=brightness={brightness}:contrast={contrast}:saturation={saturation}:gamma={gamma}"
        return self._run_ffmpeg(
            ["-i", video_path, "-vf", vf] + SEARCH_ENCODE_ARGS, output,
        )

    def color_correct(
        self, video_path: str,
        brightness: float = 0.0, contrast: float = 1.0,
        saturation: float = 1.0, gamma: float = 1.0,
    ) -> str:
        """技术性色彩校正（曝光校正、白平衡修正）"""
        output = self._make_output_path(video_path, "cc")
        vf = f"eq=brightness={brightness}:contrast={contrast}:saturation={saturation}:gamma={gamma}"
        return self._run_ffmpeg(
            ["-i", video_path, "-vf", vf] + SEARCH_ENCODE_ARGS, output,
        )

    def color_grade(
        self, video_path: str,
        brightness: float = 0.0, contrast: float = 1.0,
        saturation: float = 1.0, gamma: float = 1.0,
    ) -> str:
        """创意调色（风格化色调、情绪渲染）"""
        output = self._make_output_path(video_path, "cg")
        vf = f"eq=brightness={brightness}:contrast={contrast}:saturation={saturation}:gamma={gamma}"
        return self._run_ffmpeg(
            ["-i", video_path, "-vf", vf] + SEARCH_ENCODE_ARGS, output,
        )

    def white_balance(self, video_path: str, temperature: float = 6500) -> str:
        output = self._make_output_path(video_path, "wb")
        vf = f"colortemperature=temperature={temperature}"
        return self._run_ffmpeg(
            ["-i", video_path, "-vf", vf] + SEARCH_ENCODE_ARGS, output,
        )

    def denoise(self, video_path: str, strength: float = 4.0) -> str:
        output = self._make_output_path(video_path, "denoise")
        vf = f"hqdn3d=luma_spatial={strength}"
        return self._run_ffmpeg(
            ["-i", video_path, "-vf", vf] + SEARCH_ENCODE_ARGS, output,
        )

    def sharpen(self, video_path: str, amount: float = 1.0) -> str:
        output = self._make_output_path(video_path, "sharp")
        vf = f"unsharp=5:5:{amount}:5:5:0"
        return self._run_ffmpeg(
            ["-i", video_path, "-vf", vf] + SEARCH_ENCODE_ARGS, output,
        )

    def stabilize(self, video_path: str, smoothing: int = 10) -> str:
        """视频防抖（vidstab 两遍处理）— 此工具无法合并，必须独立执行"""
        output = self._make_output_path(video_path, "stab")
        transforms_path = os.path.join(self.temp_dir, "transforms.trf")

        cmd1 = [
            self.ffmpeg, "-y",
            "-i", video_path,
            "-vf", f"vidstabdetect=result={transforms_path}:shakiness=5",
            "-f", "null", os.devnull,
        ]
        r1 = subprocess.run(cmd1, capture_output=True, text=True)
        if r1.returncode != 0:
            raise RuntimeError(f"vidstabdetect 失败: {r1.stderr[-500:]}")

        vf = f"vidstabtransform=input={transforms_path}:smoothing={smoothing}:interpol=linear,unsharp=5:5:0.8:3:3:0.4"
        return self._run_ffmpeg(
            ["-i", video_path, "-vf", vf] + SEARCH_ENCODE_ARGS, output,
        )

    def apply_lut(self, video_path: str, lut_file: str) -> str:
        output = self._make_output_path(video_path, "lut")
        vf = f"lut3d=file={lut_file}"
        return self._run_ffmpeg(
            ["-i", video_path, "-vf", vf] + SEARCH_ENCODE_ARGS, output,
        )

    def speed_adjust(self, video_path: str, factor: float = 1.0) -> str:
        if factor <= 0:
            raise ValueError("speed factor 必须 > 0")
        output = self._make_output_path(video_path, "speed")
        pts = 1.0 / factor
        vf = f"setpts={pts}*PTS"

        atempo_filters = []
        remaining = factor
        while remaining > 2.0:
            atempo_filters.append("atempo=2.0")
            remaining /= 2.0
        while remaining < 0.5:
            atempo_filters.append("atempo=0.5")
            remaining /= 0.5
        atempo_filters.append(f"atempo={remaining:.4f}")
        af = ",".join(atempo_filters)

        return self._run_ffmpeg(
            ["-i", video_path, "-vf", vf, "-af", af,
             "-c:v", "libx264", "-preset", "fast", "-crf", "20"],
            output,
        )

    # ── 片段裁剪 ──────────────────────────────────────

    def trim_segment(self, video_path: str, start: float, end: float) -> str:
        output = self._make_output_path(video_path, f"trim_{start:.1f}_{end:.1f}")
        return self._run_ffmpeg(
            ["-i", video_path,
             "-ss", f"{start:.3f}", "-to", f"{end:.3f}"]
            + SEARCH_ENCODE_ARGS,
            output,
        )

    # ── 片段拼接 ──────────────────────────────────────

    def concat_segments(self, segment_paths: list[str], output_path: str) -> str:
        """拼接多个视频片段（重新编码以确保无缝）"""
        list_file = os.path.join(self.temp_dir, "concat_list.txt")
        with open(list_file, "w") as f:
            for p in segment_paths:
                f.write(f"file '{os.path.abspath(p)}'\n")

        # 用重新编码拼接，避免 -c copy 导致的拼接点跳变
        return self._run_ffmpeg(
            ["-f", "concat", "-safe", "0", "-i", list_file] + SEARCH_ENCODE_ARGS,
            output_path,
        )

    # ── 获取滤镜字符串（供合并模式使用）────────────────

    @staticmethod
    def get_filter_string(
        tool_name: str,
        params: dict,
        time_range: tuple[float, float] | None = None,
    ) -> str | None:
        """返回工具对应的 FFmpeg 滤镜字符串，用于合并。

        当 time_range 不为 None 时，自动添加 enable='between(t,start,end)'，
        使滤镜仅对指定时间段生效（段级编辑）。
        """
        eq_filter = lambda p: (
            f"eq=brightness={p.get('brightness', 0)}:"
            f"contrast={p.get('contrast', 1)}:"
            f"saturation={p.get('saturation', 1)}:"
            f"gamma={p.get('gamma', 1)}"
        )
        mapping = {
            "color_adjust": eq_filter,          # 旧名向后兼容
            "color_correct": eq_filter,         # 技术校正（阶段 3）
            "color_grade": eq_filter,           # 创意调色（阶段 4）
            "white_balance": lambda p: f"colortemperature=temperature={p.get('temperature', 6500)}",
            "denoise": lambda p: f"hqdn3d=luma_spatial={p.get('strength', 4)}",
            "sharpen": lambda p: f"unsharp=5:5:{p.get('amount', 1)}:5:5:0",
        }
        if tool_name not in mapping:
            return None  # stabilize, speed_adjust, apply_lut 等不可合并

        filter_str = mapping[tool_name](params)

        # 段级编辑：添加时间范围 enable
        if time_range is not None:
            start, end = time_range
            filter_str += f":enable='between(t,{start:.3f},{end:.3f})'"

        return filter_str

    # ── 工具注册表 ────────────────────────────────────

    def get_tool_registry(self) -> dict:
        return {
            "color_adjust": {
                "func": self.color_adjust,
                "description": "调整亮度/对比度/饱和度/Gamma（旧名，建议用 color_correct 或 color_grade）",
                "params": ["brightness", "contrast", "saturation", "gamma"],
                "mergeable": True,
                "stage": "color_grade",
            },
            "color_correct": {
                "func": self.color_correct,
                "description": "技术性色彩校正（曝光、白平衡、灰度一致性）",
                "params": ["brightness", "contrast", "saturation", "gamma"],
                "mergeable": True,
                "stage": "color_correct",
            },
            "color_grade": {
                "func": self.color_grade,
                "description": "创意调色（风格化色调、情绪渲染）",
                "params": ["brightness", "contrast", "saturation", "gamma"],
                "mergeable": True,
                "stage": "color_grade",
            },
            "white_balance": {
                "func": self.white_balance,
                "description": "调整色温（白平衡）",
                "params": ["temperature"],
                "mergeable": True,
                "stage": "color_correct",
            },
            "denoise": {
                "func": self.denoise,
                "description": "视频降噪",
                "params": ["strength"],
                "mergeable": True,
                "stage": "denoise",
            },
            "sharpen": {
                "func": self.sharpen,
                "description": "锐化画面",
                "params": ["amount"],
                "mergeable": True,
                "stage": "sharpen",
            },
            "stabilize": {
                "func": self.stabilize,
                "description": "视频防抖（仅支持全局应用）",
                "params": ["smoothing"],
                "mergeable": False,
                "global_only": True,
                "stage": "stabilize",
            },
            "apply_lut": {
                "func": self.apply_lut,
                "description": "应用 3D LUT 滤镜",
                "params": ["lut_file"],
                "mergeable": False,
            },
            "speed_adjust": {
                "func": self.speed_adjust,
                "description": "视频变速",
                "params": ["factor"],
                "mergeable": False,
            },
        }
