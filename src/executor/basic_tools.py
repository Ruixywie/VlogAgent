"""基础工具：FFmpeg / OpenCV 封装"""

import os
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BasicTools:
    def __init__(self, config: dict):
        self.ffmpeg = config.get("ffmpeg_path", "ffmpeg")
        self.ffprobe = config.get("ffprobe_path", "ffprobe")
        self.temp_dir = config.get("temp_dir", "output/temp")
        os.makedirs(self.temp_dir, exist_ok=True)

    def _run_ffmpeg(self, args: list[str], output_path: str) -> str:
        """执行 FFmpeg 命令"""
        cmd = [self.ffmpeg, "-y"] + args + [output_path]
        logger.info(f"FFmpeg: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg 失败: {result.stderr}")
        return output_path

    def _make_output_path(self, input_path: str, suffix: str) -> str:
        """生成输出文件路径"""
        stem = Path(input_path).stem
        ext = Path(input_path).suffix
        return os.path.join(self.temp_dir, f"{stem}_{suffix}{ext}")

    # ── 色彩调整 ──────────────────────────────────────

    def color_adjust(
        self,
        video_path: str,
        brightness: float = 0.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        gamma: float = 1.0,
    ) -> str:
        """调整亮度/对比度/饱和度/Gamma"""
        output = self._make_output_path(video_path, "color")
        vf = f"eq=brightness={brightness}:contrast={contrast}:saturation={saturation}:gamma={gamma}"
        return self._run_ffmpeg(
            ["-i", video_path, "-vf", vf, "-c:a", "copy"],
            output,
        )

    # ── 白平衡 ────────────────────────────────────────

    def white_balance(self, video_path: str, temperature: float = 6500) -> str:
        """调整色温（白平衡）"""
        output = self._make_output_path(video_path, "wb")
        vf = f"colortemperature=temperature={temperature}"
        return self._run_ffmpeg(
            ["-i", video_path, "-vf", vf, "-c:a", "copy"],
            output,
        )

    # ── 降噪 ──────────────────────────────────────────

    def denoise(self, video_path: str, strength: float = 4.0) -> str:
        """视频降噪（hqdn3d）"""
        output = self._make_output_path(video_path, "denoise")
        vf = f"hqdn3d=luma_spatial={strength}"
        return self._run_ffmpeg(
            ["-i", video_path, "-vf", vf, "-c:a", "copy"],
            output,
        )

    # ── 锐化 ──────────────────────────────────────────

    def sharpen(self, video_path: str, amount: float = 1.0) -> str:
        """锐化（unsharp mask）"""
        output = self._make_output_path(video_path, "sharp")
        # unsharp=lx:ly:la:cx:cy:ca  (luma_size:luma_size:luma_amount:...)
        vf = f"unsharp=5:5:{amount}:5:5:0"
        return self._run_ffmpeg(
            ["-i", video_path, "-vf", vf, "-c:a", "copy"],
            output,
        )

    # ── 稳定 ──────────────────────────────────────────

    def stabilize(self, video_path: str, smoothing: int = 10) -> str:
        """视频防抖（vidstab 两遍处理）"""
        output = self._make_output_path(video_path, "stab")
        transforms_path = os.path.join(self.temp_dir, "transforms.trf")

        # Pass 1: 检测运动
        cmd1 = [
            self.ffmpeg, "-y",
            "-i", video_path,
            "-vf", f"vidstabdetect=result={transforms_path}:shakiness=5",
            "-f", "null", os.devnull,
        ]
        r1 = subprocess.run(cmd1, capture_output=True, text=True)
        if r1.returncode != 0:
            raise RuntimeError(f"vidstabdetect 失败: {r1.stderr}")

        # Pass 2: 应用稳定
        vf = f"vidstabtransform=input={transforms_path}:smoothing={smoothing}:interpol=linear,unsharp=5:5:0.8:3:3:0.4"
        return self._run_ffmpeg(
            ["-i", video_path, "-vf", vf, "-c:a", "copy"],
            output,
        )

    # ── LUT 应用 ──────────────────────────────────────

    def apply_lut(self, video_path: str, lut_file: str) -> str:
        """应用 3D LUT"""
        output = self._make_output_path(video_path, "lut")
        vf = f"lut3d=file={lut_file}"
        return self._run_ffmpeg(
            ["-i", video_path, "-vf", vf, "-c:a", "copy"],
            output,
        )

    # ── 变速 ──────────────────────────────────────────

    def speed_adjust(self, video_path: str, factor: float = 1.0) -> str:
        """视频变速（音视频同步）"""
        if factor <= 0:
            raise ValueError("speed factor 必须 > 0")
        output = self._make_output_path(video_path, "speed")
        pts = 1.0 / factor
        vf = f"setpts={pts}*PTS"

        # 音频变速：atempo 只支持 [0.5, 2.0]，需要链式调用
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
            ["-i", video_path, "-vf", vf, "-af", af],
            output,
        )

    # ── 片段裁剪 ──────────────────────────────────────

    def trim_segment(
        self, video_path: str, start: float, end: float
    ) -> str:
        """裁剪视频片段"""
        output = self._make_output_path(video_path, f"trim_{start:.1f}_{end:.1f}")
        return self._run_ffmpeg(
            [
                "-i", video_path,
                "-ss", f"{start:.3f}",
                "-to", f"{end:.3f}",
                "-c:v", "libx264", "-c:a", "aac",
            ],
            output,
        )

    # ── 片段拼接 ──────────────────────────────────────

    def concat_segments(self, segment_paths: list[str], output_path: str) -> str:
        """拼接多个视频片段"""
        list_file = os.path.join(self.temp_dir, "concat_list.txt")
        with open(list_file, "w") as f:
            for p in segment_paths:
                f.write(f"file '{os.path.abspath(p)}'\n")

        return self._run_ffmpeg(
            ["-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy"],
            output_path,
        )

    # ── 工具注册表 ────────────────────────────────────

    def get_tool_registry(self) -> dict:
        """返回所有可用工具的名称和说明"""
        return {
            "color_adjust": {
                "func": self.color_adjust,
                "description": "调整亮度/对比度/饱和度/Gamma",
                "params": ["brightness", "contrast", "saturation", "gamma"],
            },
            "white_balance": {
                "func": self.white_balance,
                "description": "调整色温（白平衡）",
                "params": ["temperature"],
            },
            "denoise": {
                "func": self.denoise,
                "description": "视频降噪",
                "params": ["strength"],
            },
            "sharpen": {
                "func": self.sharpen,
                "description": "锐化画面",
                "params": ["amount"],
            },
            "stabilize": {
                "func": self.stabilize,
                "description": "视频防抖",
                "params": ["smoothing"],
            },
            "apply_lut": {
                "func": self.apply_lut,
                "description": "应用 3D LUT 滤镜",
                "params": ["lut_file"],
            },
            "speed_adjust": {
                "func": self.speed_adjust,
                "description": "视频变速",
                "params": ["factor"],
            },
        }
