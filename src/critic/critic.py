"""Critic Agent：结构化评审 + 路由决策

职责：
- evaluate()：逐段对比编辑前后画面 → 因果分析 → 结构化反馈 + 路由
- VLM 自主决定路由（accept/refine/redirect）和改进建议

内部组合 Evaluator 获取技术指标，VLM 负责逐段视觉评审和决策。
"""

import base64
import logging

import cv2
import numpy as np

from src.models import (
    SegmentMetadata, EditAction, EditPlan, EvaluationResult,
    CriticFeedback, SegmentCritic, FallbackLLM, extract_json,
)
from src.evaluator.evaluator import Evaluator
from src.utils import build_comparison_storyboard

logger = logging.getLogger(__name__)

CRITIC_PROMPT = """你是专业视频编辑质量评审。请逐段对比编辑前后的画面，给出结构化评审和改进建议。

## 评审要求

对每个片段判断：
- verdict: "improved"（变好了）/ "unchanged"（没变化）/ "degraded"（变差了）
- reason: 简要说明原因（如"提亮自然，暗部细节改善"或"稳定化过度导致模糊"）

## 建议要求

给出具体的改进建议（suggestions），你可以自由发挥，例如：
- 某段的操作过度了，建议减小力度或去掉
- 某段的操作有效，可以保持
- 整体风格方向的调整建议
- 哪些段还有改进空间，可以尝试什么

注意：建议应该是方向性的指导，不要写具体的参数数值。

## 评分标准
- 0.5 = 编辑前后无差异
- >0.5 = 编辑后更好
- <0.5 = 编辑后变差
- 原片已经很好，编辑没有实质提升 → 约 0.5
- 编辑引入了伪影/锯齿/过度处理 → 必须给低分

只返回 JSON：
{
  "overall_score": 0.0-1.0,
  "segment_feedback": {
    "seg-0": {"verdict": "improved", "reason": "..."},
    "seg-1": {"verdict": "degraded", "reason": "..."}
  },
  "global_issues": ["..."],
  "global_positives": ["..."],
  "suggestions": ["..."]
}"""


class Critic:
    def __init__(self, config: dict, llm: FallbackLLM):
        self.llm = llm
        self.weights = config.get("weights", {
            "visual_quality": 0.10,
            "content_fidelity": 0.15,
            "inter_segment_consistency": 0.10,
            "audio_integrity": 0.05,
            "aesthetic": 0.60,
        })
        self.accept_threshold = config.get("accept_threshold", 0.01)
        self.redirect_after_refines = config.get("redirect_after_refines", 2)
        self.good_enough_score = config.get("good_enough_score", 0.85)

        # 组合 Evaluator 获取技术指标
        self._evaluator = Evaluator(config, llm)

    def evaluate(
        self,
        original_path: str,
        edited_path: str,
        segments: list[SegmentMetadata],
        plan: EditPlan,
        prev_score: float,
        refine_count: int,
    ) -> CriticFeedback:
        """完整评审流程：技术指标 + VLM 自主评审和决策"""
        logger.info("Critic: 开始评审...")

        scenes = [s.time_range for s in segments]

        # 1. 技术指标（复用 Evaluator）
        tech_scores = {
            "visual_quality": self._evaluator.score_visual_quality(edited_path),
            "content_fidelity": self._evaluator.score_content_fidelity(original_path, edited_path),
            "inter_segment_consistency": self._evaluator.score_inter_segment_consistency(edited_path, scenes),
            "audio_integrity": self._evaluator.score_audio_integrity(original_path, edited_path, segments),
        }

        # 2. VLM 自主评审 + 路由决策（一次调用完成）
        vlm_result = self._vlm_structured_review(original_path, edited_path, segments, plan)

        # 3. 计算综合分数（技术指标 + VLM 美学分加权）
        aesthetic_score = vlm_result.get("overall_score", 0.5)
        overall_score = (
            tech_scores["visual_quality"] * self.weights.get("visual_quality", 0.10)
            + tech_scores["content_fidelity"] * self.weights.get("content_fidelity", 0.15)
            + tech_scores["inter_segment_consistency"] * self.weights.get("inter_segment_consistency", 0.10)
            + tech_scores["audio_integrity"] * self.weights.get("audio_integrity", 0.05)
            + aesthetic_score * self.weights.get("aesthetic", 0.60)
        )

        # 4. 构建 segment_feedback
        segment_feedback = {}
        for seg_id, fb in vlm_result.get("segment_feedback", {}).items():
            if isinstance(fb, dict):
                segment_feedback[seg_id] = SegmentCritic(
                    segment_id=seg_id,
                    verdict=fb.get("verdict", "unchanged"),
                    reason=fb.get("reason", ""),
                    action_feedback=fb.get("action_feedback", {}),
                )

        # 5. 路由决策：基于分数与基线的比对
        route, route_reason = self._compute_route(
            overall_score, prev_score, segment_feedback, refine_count
        )

        feedback = CriticFeedback(
            overall_score=overall_score,
            segment_feedback=segment_feedback,
            global_issues=vlm_result.get("global_issues", []),
            global_positives=vlm_result.get("global_positives", []),
            suggestions=vlm_result.get("suggestions", []),
            route=route,
            route_reason=route_reason,
        )

        logger.info(
            f"Critic 评审: score={overall_score:.3f} "
            f"(视觉={tech_scores['visual_quality']:.2f} "
            f"保真={tech_scores['content_fidelity']:.2f} "
            f"一致={tech_scores['inter_segment_consistency']:.2f} "
            f"美学={aesthetic_score:.2f}) "
            f"route={route} ({route_reason})"
        )
        return feedback

    def _vlm_structured_review(
        self,
        original_path: str,
        edited_path: str,
        segments: list[SegmentMetadata],
        plan: EditPlan,
    ) -> dict:
        """VLM 自主评审 + 路由决策（一次调用）"""
        scenes = [s.time_range for s in segments]

        orig_frames = self._extract_segment_frames(original_path, scenes)
        edit_frames = self._extract_segment_frames(edited_path, scenes)

        if not orig_frames or not edit_frames:
            return {"overall_score": 0.5, "segment_feedback": {}, "suggestions": []}

        # 构建对比 Storyboard
        comparison = build_comparison_storyboard(orig_frames, edit_frames)

        content = []
        actions_desc = "\n".join(
            f"  - [{a.tool_name}] {a.action_description} → {a.target_segment}"
            for a in plan.actions
        )
        content.append({"type": "text", "text": f"## 执行的编辑动作\n{actions_desc}\n"})

        if comparison:
            desc, b64 = comparison[0]
            content.append({"type": "text", "text": "## 编辑前后对比（左:原始 右:编辑后）"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
            })
        else:
            content.append({"type": "text", "text": "（无法生成对比图，请基于动作列表评估）"})

        content.append({
            "type": "text",
            "text": "\n请逐段评审编辑效果，给出改进建议，并做出路由决策。",
        })

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": CRITIC_PROMPT},
                    {"role": "user", "content": content},
                ],
                temperature=0.3,
                max_tokens=600,
            )
            data = extract_json(response.choices[0].message.content)
            if data and isinstance(data, dict):
                return data
        except Exception as e:
            logger.warning(f"Critic VLM 评审失败: {e}")

        return {"overall_score": 0.5, "segment_feedback": {}, "suggestions": []}

    def _compute_route(
        self,
        overall_score: float,
        prev_score: float,
        segment_feedback: dict,
        refine_count: int,
    ) -> tuple[str, str]:
        """基于分数比对的路由决策"""
        improvement = overall_score - prev_score

        # 分数明确提升 → accept
        if improvement > self.accept_threshold:
            return "accept", f"分数提升 +{improvement:.3f}"

        # 分数已经足够高 → accept
        if overall_score > self.good_enough_score:
            return "accept", f"分数已达 {overall_score:.2f}"

        # 连续 refine 太多次 → redirect
        if refine_count >= self.redirect_after_refines:
            return "redirect", f"连续 {refine_count} 轮 refine 无提升，需要换策略方向"

        # 分析 degraded 段数量
        degraded = [seg_id for seg_id, sc in segment_feedback.items()
                    if (sc.verdict if hasattr(sc, "verdict") else "") == "degraded"]

        # 大部分段 degraded → redirect
        if degraded and len(degraded) > len(segment_feedback) // 2:
            return "redirect", f"多数段变差 ({', '.join(degraded)})，策略方向可能有误"

        # 其他情况 → refine
        return "refine", f"分数未达标 (improvement={improvement:.3f})，需要调整"

    def _extract_segment_frames(
        self, path: str, scenes: list[tuple[float, float]]
    ) -> list[tuple[str, str]]:
        """按场景分段提取中间帧，返回 [(label, base64), ...]"""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return []
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            cap.release()
            return []

        frames = []
        for i, (start, end) in enumerate(scenes):
            mid_frame = int((start + end) / 2 * fps)
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
            except cv2.error:
                continue
            _, buf = cv2.imencode(".jpg", frame)
            b64 = base64.b64encode(buf).decode("utf-8")
            frames.append((f"seg-{i} ({start:.1f}s-{end:.1f}s)", b64))
        cap.release()
        return frames

    # ── 逐阶段轻量评审 ─────────────────────────────

    STAGE_CRITIC_PROMPT = """你是专业视频编辑质量评审。请评审当前阶段的编辑效果。

## 评审要求
对每个片段判断 verdict: "improved"/"unchanged"/"degraded" + reason。

## 路由决策（你来决定）
- "accept": 编辑整体有效，这个阶段的操作可以保留
- "refine": 有问题需要调整（某些段变差了，或效果不明显），给出具体建议
- "skip": 这个操作对画面没有帮助甚至有害，应该跳过整个阶段

## 注意
- 如果有任何段 degraded，应该倾向 refine 或 skip，不应该 accept
- 如果所有段都 unchanged，说明操作没效果，应该 skip
- 只有大多数段 improved 且没有 degraded 才 accept

只返回 JSON：
{
  "overall_score": 0.0-1.0,
  "segment_feedback": {"seg-0": {"verdict": "...", "reason": "..."}},
  "suggestions": ["..."],
  "route": "accept 或 refine 或 skip",
  "route_reason": "决策理由"
}"""

    def evaluate_stage(
        self,
        original_path: str,
        edited_path: str,
        segments: list[SegmentMetadata],
        stage_name: str,
        action: EditAction,
        prev_score: float,
        refine_count: int,
    ) -> CriticFeedback:
        """评审单个阶段的编辑效果——VLM 自主决定路由"""
        logger.info(f"Critic: 评审阶段 {stage_name}...")

        scenes = [s.time_range for s in segments]

        # 技术指标
        visual = self._evaluator.score_visual_quality(edited_path)
        fidelity = self._evaluator.score_content_fidelity(original_path, edited_path)

        # VLM 评审（带路由决策）
        orig_frames = self._extract_segment_frames(original_path, scenes)
        edit_frames = self._extract_segment_frames(edited_path, scenes)

        vlm_result = {"overall_score": 0.5, "segment_feedback": {},
                      "suggestions": [], "route": "refine", "route_reason": ""}

        if orig_frames and edit_frames:
            comparison = build_comparison_storyboard(orig_frames, edit_frames)
            content = []
            content.append({"type": "text", "text": (
                f"## 当前阶段: {stage_name}\n"
                f"## 执行的操作: [{action.tool_name}] {action.action_description} → {action.target_segment}\n"
            )})
            if comparison:
                desc, b64 = comparison[0]
                content.append({"type": "text", "text": "## 编辑前后对比（左:原始 右:编辑后）"})
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
                })
            content.append({"type": "text", "text": "\n请评审并做出路由决策。"})

            try:
                response = self.llm.chat(
                    messages=[
                        {"role": "system", "content": self.STAGE_CRITIC_PROMPT},
                        {"role": "user", "content": content},
                    ],
                    temperature=0.3,
                    max_tokens=500,
                )
                data = extract_json(response.choices[0].message.content)
                if data and isinstance(data, dict):
                    vlm_result = data
            except Exception as e:
                logger.warning(f"Critic VLM 评审失败: {e}")

        aesthetic = vlm_result.get("overall_score", 0.5)
        overall_score = visual * 0.15 + fidelity * 0.20 + aesthetic * 0.65

        # 构建反馈
        segment_feedback = {}
        for seg_id, fb in vlm_result.get("segment_feedback", {}).items():
            if isinstance(fb, dict):
                segment_feedback[seg_id] = SegmentCritic(
                    segment_id=seg_id,
                    verdict=fb.get("verdict", "unchanged"),
                    reason=fb.get("reason", ""),
                )

        # 路由：优先用 VLM 决策，兜底用规则
        route = vlm_result.get("route", "")
        route_reason = vlm_result.get("route_reason", "")

        if route not in ("accept", "refine", "skip"):
            # VLM 没给有效路由，用规则兜底
            if refine_count >= 2:
                route, route_reason = "skip", f"阶段 {stage_name} 重试 {refine_count} 次，跳过"
            else:
                route, route_reason = "refine", "VLM 未给出有效路由"

        feedback = CriticFeedback(
            overall_score=overall_score,
            segment_feedback=segment_feedback,
            global_issues=vlm_result.get("global_issues", []),
            suggestions=vlm_result.get("suggestions", []),
            route=route,
            route_reason=route_reason,
        )

        logger.info(
            f"Critic 阶段 {stage_name}: score={overall_score:.3f} "
            f"(视觉={visual:.2f} 保真={fidelity:.2f} 美学={aesthetic:.2f}) "
            f"route={route}"
        )
        return feedback
