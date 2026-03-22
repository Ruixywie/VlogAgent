"""视频分析模块：场景分割 + 逐段质量检测 + 音频分析 + 关键帧提取"""

import os
import subprocess
import logging
from pathlib import Path

import cv2
import numpy as np

from src.models import SegmentMetadata

logger = logging.getLogger(__name__)


class VideoAnalyzer:
    def __init__(self, config: dict):
        self.scene_threshold = config.get("scene_threshold", 27.0)
        self.min_scene_len = config.get("min_scene_len", 15)
        self.keyframe_interval = config.get("keyframe_interval", 1.0)  # 每N秒提取1帧
        self.max_keyframes_per_segment = config.get("max_keyframes_per_segment", 10)
        self.whisper_model_name = config.get("whisper_model", "base")
        self.ffmpeg_path = config.get("ffmpeg_path", "ffmpeg")
        self._whisper_model = None

    # ── 场景分割 ──────────────────────────────────────────

    def detect_scenes(self, video_path: str) -> list[tuple[float, float]]:
        """用 PySceneDetect 进行场景分割，返回 [(start_sec, end_sec), ...]"""
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector

        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(
                threshold=self.scene_threshold,
                min_scene_len=self.min_scene_len,
            )
        )
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        if not scene_list:
            # 没有检测到场景切换，整段视频作为一个场景
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            duration = total_frames / fps if fps > 0 else 0.0
            return [(0.0, duration)]

        return [(s.get_seconds(), e.get_seconds()) for s, e in scene_list]

    # ── 逐段质量检测 ─────────────────────────────────────

    def analyze_segment_quality(
        self, video_path: str, start: float, end: float
    ) -> dict:
        """对单段视频提取质量指标"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start * fps)
        end_frame = int(end * fps)

        brightnesses = []
        sharpnesses = []
        color_temps = []
        noise_levels = []
        prev_gray = None
        flow_vars = []

        sample_interval = max(1, (end_frame - start_frame) // 30)  # 最多采样30帧

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for i in range(start_frame, end_frame, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break

            # 亮度：Y通道均值
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            brightnesses.append(float(yuv[:, :, 0].mean()))

            # 清晰度：Laplacian 方差
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpnesses.append(float(lap_var))

            # 色温估算：R/B 通道比值
            b, _, r = cv2.split(frame)
            b_mean = float(b.mean()) + 1e-6
            r_mean = float(r.mean())
            color_temps.append(r_mean / b_mean)

            # 噪声估计：高频能量
            noise = float(cv2.Laplacian(gray, cv2.CV_64F).std())
            noise_levels.append(noise)

            # 稳定性：光流方差
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
                flow_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                flow_vars.append(float(flow_mag.var()))
            prev_gray = gray

        cap.release()

        return {
            "mean_brightness": float(np.mean(brightnesses)) if brightnesses else 0.0,
            "sharpness_score": float(np.mean(sharpnesses)) if sharpnesses else 0.0,
            "color_temp_est": float(np.mean(color_temps)) if color_temps else 1.0,
            "stability_score": float(np.mean(flow_vars)) if flow_vars else 0.0,
            "noise_level": float(np.mean(noise_levels)) if noise_levels else 0.0,
        }

    # ── 关键帧提取 ─────────────────────────────────────

    def extract_keyframes(
        self, video_path: str, start: float, end: float, seg_id: int, output_dir: str
    ) -> list[str]:
        """按时间间隔自适应提取关键帧（默认每秒1帧，上限可配置）"""
        os.makedirs(output_dir, exist_ok=True)
        duration = end - start

        # 根据片段时长和间隔计算帧数，至少2帧
        n = max(2, int(duration / self.keyframe_interval))
        n = min(n, self.max_keyframes_per_segment)

        paths = []
        for i in range(n):
            t = start + duration * (i + 0.5) / n
            out_path = os.path.join(output_dir, f"seg{seg_id}_kf{i}.jpg")
            cmd = [
                self.ffmpeg_path, "-y",
                "-ss", f"{t:.3f}",
                "-i", video_path,
                "-frames:v", "1",
                "-q:v", "2",
                out_path,
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            paths.append(out_path)

        return paths

    # ── 音频分析 ──────────────────────────────────────────

    def analyze_audio(self, video_path: str, start: float, end: float) -> dict:
        """Whisper ASR 转录 + 语音检测"""
        import whisper
        import tempfile

        # 提取音频片段
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cmd = [
                self.ffmpeg_path, "-y",
                "-i", video_path,
                "-ss", f"{start:.3f}",
                "-to", f"{end:.3f}",
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                tmp_path,
            ]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                return {"has_speech": False, "speech_text": ""}

            # Whisper 转录
            if self._whisper_model is None:
                self._whisper_model = whisper.load_model(self.whisper_model_name)

            result = self._whisper_model.transcribe(tmp_path, language=None)
            text = result.get("text", "").strip()

            return {
                "has_speech": len(text) > 0,
                "speech_text": text,
            }
        except Exception as e:
            logger.warning(f"音频分析失败: {e}")
            return {"has_speech": False, "speech_text": ""}
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ── 主分析流程 ─────────────────────────────────────

    def analyze(self, video_path: str, output_dir: str = "output/analysis") -> list[SegmentMetadata]:
        """完整视频分析流程"""
        logger.info(f"开始分析视频: {video_path}")

        # 1. 场景分割
        scenes = self.detect_scenes(video_path)
        logger.info(f"检测到 {len(scenes)} 个场景")

        segments = []
        for seg_id, (start, end) in enumerate(scenes):
            logger.info(f"分析片段 {seg_id}: {start:.1f}s - {end:.1f}s")

            # 2. 质量检测
            quality = self.analyze_segment_quality(video_path, start, end)

            # 3. 关键帧提取
            kf_dir = os.path.join(output_dir, "keyframes")
            keyframe_paths = self.extract_keyframes(video_path, start, end, seg_id, kf_dir)

            # 4. 音频分析
            audio = self.analyze_audio(video_path, start, end)

            seg = SegmentMetadata(
                seg_id=seg_id,
                time_range=(start, end),
                mean_brightness=quality["mean_brightness"],
                color_temp_est=quality["color_temp_est"],
                sharpness_score=quality["sharpness_score"],
                stability_score=quality["stability_score"],
                noise_level=quality["noise_level"],
                has_speech=audio["has_speech"],
                speech_text=audio["speech_text"],
                keyframe_paths=keyframe_paths,
            )
            segments.append(seg)

        return segments
