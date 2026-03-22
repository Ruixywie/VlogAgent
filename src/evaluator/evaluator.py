"""Evaluator：多维评分 + 闭环控制"""

import base64
import json
import logging
import os

import cv2
import numpy as np

from src.models import SegmentMetadata, EvaluationResult, FallbackLLM, extract_json

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, config: dict, llm: FallbackLLM):
        # 参考 PhotoAgent：美学/感知质量为主，技术指标为辅
        self.weights = config.get("weights", {
            "visual_quality": 0.10,
            "content_fidelity": 0.15,
            "inter_segment_consistency": 0.10,
            "audio_integrity": 0.05,
            "aesthetic": 0.60,
        })
        self.clip_model_name = config.get("clip_model", "ViT-B-32")
        self.clip_pretrained = config.get("clip_pretrained", "openai")
        self.llm = llm
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
        if not cap.isOpened():
            logger.warning(f"无法打开视频: {video_path}")
            return 0.5
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return 0.5
        sample_count = min(20, total)
        interval = max(1, total // sample_count)

        sharpness_scores = []
        exposure_scores = []

        for i in range(0, total, interval):
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
            except cv2.error:
                continue

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
            if not cap.isOpened():
                return None
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            interval = max(1, total // n)
            frames = []
            for i in range(0, total, interval):
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        continue
                except cv2.error:
                    continue
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

        # 对齐帧数：取两者的最小值
        min_n = min(len(orig_frames), len(edit_frames))
        orig_frames = orig_frames[:min_n]
        edit_frames = edit_frames[:min_n]

        with torch.no_grad():
            orig_feat = self._clip_model.encode_image(orig_frames)
            edit_feat = self._clip_model.encode_image(edit_frames)

            orig_feat = orig_feat / orig_feat.norm(dim=-1, keepdim=True)
            edit_feat = edit_feat / edit_feat.norm(dim=-1, keepdim=True)

            sim = (orig_feat * edit_feat).sum(dim=-1).mean().item()

        return max(0.0, min(1.0, sim))

    # ── 3. 段间一致性 ────────────────────────────────

    def score_inter_segment_consistency(self, video_path: str, scenes: list[tuple[float, float]]) -> float:
        """相邻段 Lab 色彩空间差异"""
        if len(scenes) <= 1:
            return 1.0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 1.0
        fps = cap.get(cv2.CAP_PROP_FPS)

        segment_colors = []
        for start, end in scenes:
            mid_frame = int((start + end) / 2 * fps)
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
            except cv2.error:
                continue
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
        # 归一化：diff < 20 视为一致，> 60 视为不一致
        # 不同场景内容本身就有色彩差异，阈值不宜太严格
        score = max(0.0, 1.0 - (avg_diff - 20) / 40.0)
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

    def _extract_segment_frames(
        self, path: str, scenes: list[tuple[float, float]]
    ) -> list[tuple[str, str]]:
        """按场景分段提取帧：每段取中间帧，返回 [(seg_label, base64), ...]"""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return []
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            cap.release()
            return []

        frames = []
        for i, (start, end) in enumerate(scenes):
            mid_time = (start + end) / 2
            mid_frame = int(mid_time * fps)
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
            except cv2.error:
                continue
            _, buf = cv2.imencode(".jpg", frame)
            b64 = base64.b64encode(buf).decode("utf-8")
            frames.append((f"片段{i} ({start:.1f}s-{end:.1f}s)", b64))
        cap.release()
        return frames

    def score_aesthetic(
        self, original_path: str, edited_path: str,
        scenes: list[tuple[float, float]] | None = None,
    ) -> float:
        """
        VLM 逐段对比编辑前后画面打分（核心评估维度）。

        按场景分段取帧，确保每个场景都被覆盖，
        VLM 看到的是逐段并排对比（原始 vs 编辑后）。
        """
        if not scenes:
            # 回退：如果没有场景信息，均匀取 4 帧
            cap = cv2.VideoCapture(original_path)
            if not cap.isOpened():
                return 0.5
            fps = cap.get(cv2.CAP_PROP_FPS)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total / fps if fps > 0 else 0
            cap.release()
            n = max(2, min(6, int(duration / 5)))
            scenes = [(duration * i / n, duration * (i + 1) / n) for i in range(n)]

        orig_frames = self._extract_segment_frames(original_path, scenes)
        edit_frames = self._extract_segment_frames(edited_path, scenes)

        if not orig_frames or not edit_frames:
            return 0.5

        # 构建逐段对比消息
        content = []
        n_pairs = min(len(orig_frames), len(edit_frames))
        for i in range(n_pairs):
            orig_label, orig_b64 = orig_frames[i]
            edit_label, edit_b64 = edit_frames[i]

            content.append({"type": "text", "text": f"### {orig_label}"})
            content.append({"type": "text", "text": "原始:"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{orig_b64}", "detail": "low"},
            })
            content.append({"type": "text", "text": "编辑后:"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{edit_b64}", "detail": "low"},
            })

        content.append({
            "type": "text",
            "text": "\n请逐段对比原始和编辑后的画面，做出整体评估。",
        })

        response = self.llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是专业视频美学评审。请仔细逐段对比编辑前后的画面，评估编辑效果。\n\n"
                        "评估维度：\n"
                        "1. 色彩与调色：色彩是否更和谐、是否有色偏或过饱和\n"
                        "2. 曝光与细节：亮度是否合适、暗部/亮部细节是否保留\n"
                        "3. 画面质量：是否出现锯齿、噪点增加、模糊、压缩伪影\n"
                        "4. 段间一致性：各段编辑风格是否统一\n"
                        "5. 整体美感：编辑后是否比原片更好看\n\n"
                        "评分标准（这很重要）：\n"
                        "- 0.5 = 编辑前后无明显差异\n"
                        "- 0.5 以上 = 编辑后更好（越高越好）\n"
                        "- 0.5 以下 = 编辑后变差（越低越差）\n"
                        "- 如果原片已经很好，编辑没有实质提升，应给 0.5 左右\n"
                        "- 如果编辑引入了伪影/锯齿/过度处理，必须给低分\n"
                        "- 如果某些段变好但其他段变差，需要综合权衡\n\n"
                        "只返回 JSON: {\"score\": 0.0-1.0, \"reason\": \"...\"}"
                    ),
                },
                {"role": "user", "content": content},
            ],
            temperature=0.3,
            max_tokens=300,  # 只需返回 JSON + 简短理由
        )

        try:
            data = extract_json(response.choices[0].message.content)
            if data and isinstance(data, dict):
                score = float(data.get("score", 0.5))
                reason = data.get("reason", "")
                logger.info(f"美学评分: {score:.2f} - {reason}")
                return max(0.0, min(1.0, score))
            return 0.5
        except (TypeError, ValueError):
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
            aesthetic=self.score_aesthetic(original_path, edited_path, scenes=scenes),
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

    def evaluate_baseline(
        self, video_path: str, segments: list[SegmentMetadata]
    ) -> EvaluationResult:
        """原始视频基线评估（不调 VLM，美学基准为 0.5）"""
        logger.info("Evaluator: 基线评估（无 VLM）...")

        scenes = [s.time_range for s in segments]

        result = EvaluationResult(
            visual_quality=self.score_visual_quality(video_path),
            content_fidelity=1.0,         # 自己和自己比，满分
            inter_segment_consistency=self.score_inter_segment_consistency(video_path, scenes),
            audio_integrity=1.0,          # 未编辑，满分
            aesthetic=0.5,                # 基准线：无变化
        )
        result.compute_overall(self.weights)
        logger.info(f"基线分数: {result.overall_score:.3f}")
        return result
