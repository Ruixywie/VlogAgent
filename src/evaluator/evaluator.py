"""Evaluator：多维评分 + 闭环控制"""

import base64
import json
import logging
import os

import cv2
import numpy as np
from openai import OpenAI

from src.models import SegmentMetadata, EvaluationResult

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, config: dict):
        self.weights = config.get("weights", {
            "visual_quality": 0.25,
            "content_fidelity": 0.25,
            "inter_segment_consistency": 0.20,
            "audio_integrity": 0.10,
            "aesthetic": 0.20,
        })
        self.clip_model_name = config.get("clip_model", "ViT-B-32")
        self.clip_pretrained = config.get("clip_pretrained", "openai")
        self.client = OpenAI(
            api_key=config.get("api_key", None),
            base_url=config.get("base_url", None),
        )
        self.model = config.get("model", "gpt-4o")
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None

    def _load_clip(self):
        if self._clip_model is not None:
            return
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.clip_model_name, pretrained=self.clip_pretrained
        )
        self._clip_model = model.eval()
        self._clip_preprocess = preprocess
        self._clip_tokenizer = open_clip.get_tokenizer(self.clip_model_name)

    # ── 1. 视觉质量 ──────────────────────────────────

    def score_visual_quality(self, video_path: str) -> float:
        """清晰度 (Laplacian方差) + 曝光合理性"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_count = min(20, total)
        interval = max(1, total // sample_count)

        sharpness_scores = []
        exposure_scores = []

        for i in range(0, total, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 清晰度
            lap = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_scores.append(min(lap / 500.0, 1.0))  # 归一化

            # 曝光合理性：亮度在 [80, 180] 范围内得分高
            mean_br = gray.mean()
            if 80 <= mean_br <= 180:
                exposure_scores.append(1.0)
            else:
                dist = min(abs(mean_br - 80), abs(mean_br - 180)) / 80.0
                exposure_scores.append(max(0.0, 1.0 - dist))

        cap.release()

        sharpness = float(np.mean(sharpness_scores)) if sharpness_scores else 0.5
        exposure = float(np.mean(exposure_scores)) if exposure_scores else 0.5
        return 0.6 * sharpness + 0.4 * exposure

    # ── 2. 内容保真度 (CLIP) ─────────────────────────

    def score_content_fidelity(
        self, original_path: str, edited_path: str
    ) -> float:
        """CLIP 编辑前后余弦相似度"""
        import torch

        self._load_clip()

        def extract_frames(path, n=5):
            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            interval = max(1, total // n)
            frames = []
            for i in range(0, total, interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    from PIL import Image
                    img = Image.fromarray(frame_rgb)
                    frames.append(self._clip_preprocess(img))
                if len(frames) >= n:
                    break
            cap.release()
            return torch.stack(frames) if frames else None

        orig_frames = extract_frames(original_path)
        edit_frames = extract_frames(edited_path)

        if orig_frames is None or edit_frames is None:
            return 0.5

        with torch.no_grad():
            orig_feat = self._clip_model.encode_image(orig_frames)
            edit_feat = self._clip_model.encode_image(edit_frames)

            # 归一化
            orig_feat = orig_feat / orig_feat.norm(dim=-1, keepdim=True)
            edit_feat = edit_feat / edit_feat.norm(dim=-1, keepdim=True)

            # 平均余弦相似度
            sim = (orig_feat * edit_feat).sum(dim=-1).mean().item()

        return max(0.0, min(1.0, sim))

    # ── 3. 段间一致性 ────────────────────────────────

    def score_inter_segment_consistency(self, video_path: str, scenes: list[tuple[float, float]]) -> float:
        """相邻段 Lab 色彩空间差异"""
        if len(scenes) <= 1:
            return 1.0

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        segment_colors = []
        for start, end in scenes:
            mid_frame = int((start + end) / 2 * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, frame = cap.read()
            if ret:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                mean_lab = lab.mean(axis=(0, 1))
                segment_colors.append(mean_lab)

        cap.release()

        if len(segment_colors) < 2:
            return 1.0

        # 相邻段 Lab 距离
        diffs = []
        for i in range(len(segment_colors) - 1):
            diff = np.linalg.norm(segment_colors[i] - segment_colors[i + 1])
            diffs.append(diff)

        avg_diff = float(np.mean(diffs))
        # 归一化：diff < 10 视为一致，> 50 视为不一致
        score = max(0.0, 1.0 - (avg_diff - 10) / 40.0)
        return min(1.0, score)

    # ── 4. 音频完整性 ────────────────────────────────

    def score_audio_integrity(
        self, original_path: str, edited_path: str, segments: list[SegmentMetadata]
    ) -> float:
        """检查语音区间是否被破坏"""
        speech_segments = [s for s in segments if s.has_speech]
        if not speech_segments:
            return 1.0  # 无语音则满分

        # 简化检查：比较编辑前后音频时长
        import subprocess

        def get_duration(path):
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            try:
                return float(r.stdout.strip())
            except ValueError:
                return 0.0

        orig_dur = get_duration(original_path)
        edit_dur = get_duration(edited_path)

        if orig_dur == 0:
            return 1.0

        ratio = edit_dur / orig_dur
        # 时长变化 < 5% 视为完好
        if abs(ratio - 1.0) < 0.05:
            return 1.0
        # 变化 > 30% 视为严重损坏
        elif abs(ratio - 1.0) > 0.30:
            return 0.3
        else:
            return max(0.3, 1.0 - abs(ratio - 1.0) * 2)

    # ── 5. 整体美学 (MLLM-as-Judge) ─────────────────

    def score_aesthetic(self, original_path: str, edited_path: str) -> float:
        """GPT-4o 对比编辑前后关键帧打分"""

        def extract_frame(path) -> str | None:
            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return None
            _, buf = cv2.imencode(".jpg", frame)
            return base64.b64encode(buf).decode("utf-8")

        orig_b64 = extract_frame(original_path)
        edit_b64 = extract_frame(edited_path)

        if not orig_b64 or not edit_b64:
            return 0.5

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是视频美学评判专家。比较两帧画面（原始 vs 编辑后），"
                        "从色彩、曝光、构图、整体美感四个维度评分。"
                        "返回 JSON: {\"score\": 0.0-1.0, \"reason\": \"...\"}"
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "原始画面:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{orig_b64}", "detail": "low"},
                        },
                        {"type": "text", "text": "编辑后画面:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{edit_b64}", "detail": "low"},
                        },
                    ],
                },
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        try:
            data = json.loads(response.choices[0].message.content)
            score = float(data.get("score", 0.5))
            reason = data.get("reason", "")
            logger.info(f"美学评分: {score:.2f} - {reason}")
            return max(0.0, min(1.0, score))
        except (json.JSONDecodeError, ValueError):
            return 0.5

    # ── 综合评估 ──────────────────────────────────────

    def evaluate(
        self,
        original_path: str,
        edited_path: str,
        segments: list[SegmentMetadata],
    ) -> EvaluationResult:
        """5维综合评估"""
        logger.info("Evaluator: 开始评估...")

        scenes = [s.time_range for s in segments]

        result = EvaluationResult(
            visual_quality=self.score_visual_quality(edited_path),
            content_fidelity=self.score_content_fidelity(original_path, edited_path),
            inter_segment_consistency=self.score_inter_segment_consistency(edited_path, scenes),
            audio_integrity=self.score_audio_integrity(original_path, edited_path, segments),
            aesthetic=self.score_aesthetic(original_path, edited_path),
        )
        result.compute_overall(self.weights)

        logger.info(
            f"评估结果: 视觉={result.visual_quality:.2f} "
            f"保真={result.content_fidelity:.2f} "
            f"一致={result.inter_segment_consistency:.2f} "
            f"音频={result.audio_integrity:.2f} "
            f"美学={result.aesthetic:.2f} "
            f"总分={result.overall_score:.2f}"
        )
        return result
