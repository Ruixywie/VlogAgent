"""Director Agent：制定整体风格策略 + 判断是否继续编辑

职责：
- strategize()：VLM 看全片关键帧 → 输出 StyleBrief（风格策略）
- should_continue()：纯规则判断是否继续外层循环（不调 VLM）
- revise_brief()：redirect 时根据 Critic 反馈调整策略（调 VLM）
"""

import logging

from src.models import (
    SegmentMetadata, StyleBrief, StageDecision, CriticFeedback, FallbackLLM, extract_json,
)
from src.utils import build_frames_content, build_metrics_text

logger = logging.getLogger(__name__)

STRATEGIZE_PROMPT = """你是一位资深视频导演。请观察以下视频的关键帧序列和技术指标，制定整体编辑风格策略。

你需要决定：
1. 整体风格方向（如"清新自然""电影感""复古胶片"等）
2. 色彩方向（如"保持暖色调""整体偏冷""增强对比"等）
3. 编辑优先级（最需要改善什么）
4. 约束条件（哪些操作不应该做，基于画面实际情况判断）
5. 目标情绪（这段视频应该传达什么感觉）
6. 分阶段规划（按照专业后期工作流的 5 个阶段逐一决策）：
   - stabilize（防抖）：scope 选 skip/global/per_segment，加方向描述
   - denoise（降噪）：scope 选 skip/global/per_segment，加方向描述
   - color_correct（技术校正：曝光、白平衡、灰度一致性）：scope 选 skip/global/per_segment，加方向描述
   - color_grade（创意调色：风格化色调、情绪渲染）：scope 选 skip/global/per_segment，加方向描述
   - sharpen（锐化）：scope 选 skip/global/per_segment，加方向描述

注意：
- 如果原片某方面质量已经很好，该阶段应该 skip
- per_segment 时需指明哪些段需要处理（target_segments）
- color_correct 只做技术性纠正（如欠曝校正），color_grade 做创意性调色（如电影感色调）

只返回 JSON，不要输出其他文字：
{
  "overall_style": "...", "color_direction": "...", "priority": "...",
  "constraints": ["..."], "target_mood": "...",
  "stages": [
    {"stage": "stabilize", "scope": "skip", "direction": ""},
    {"stage": "denoise", "scope": "global", "direction": "轻度降噪，保留纹理"},
    {"stage": "color_correct", "scope": "per_segment", "direction": "校正seg-1欠曝", "target_segments": ["seg-1"]},
    {"stage": "color_grade", "scope": "global", "direction": "偏暖电影感色调"},
    {"stage": "sharpen", "scope": "skip", "direction": ""}
  ]
}"""

REVISE_PROMPT = """你是一位资深视频导演。之前的编辑风格策略执行后效果不佳，请根据评审反馈调整策略。

## 当前风格策略
{current_brief}

## 评审反馈
全局问题：{global_issues}
改进建议：{suggestions}
路由原因：{route_reason}

请重新制定风格策略。注意避免之前导致失败的方向。

只返回 JSON：
{{"overall_style": "...", "color_direction": "...", "priority": "...", "constraints": ["...", "..."], "target_mood": "..."}}"""


class Director:
    def __init__(self, llm: FallbackLLM):
        self.llm = llm

    def strategize(self, segments: list[SegmentMetadata]) -> StyleBrief:
        """VLM 看全片关键帧 → 制定风格策略"""
        logger.info("Director: 制定风格策略...")

        content = build_frames_content(segments)
        metrics = build_metrics_text(segments)
        content.append({
            "type": "text",
            "text": f"\n## 技术指标\n{metrics}",
        })

        response = self.llm.chat(
            messages=[
                {"role": "system", "content": STRATEGIZE_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0.5,
            max_tokens=1200,  # stages 数组需要更多 token
        )

        raw = response.choices[0].message.content
        data = extract_json(raw)

        if data and isinstance(data, dict):
            # 解析 stages
            stages = []
            for sd in data.get("stages", []):
                if isinstance(sd, dict):
                    stages.append(StageDecision(
                        stage=sd.get("stage", ""),
                        scope=sd.get("scope", "skip"),
                        direction=sd.get("direction", ""),
                        target_segments=sd.get("target_segments", []),
                    ))

            brief = StyleBrief(
                overall_style=data.get("overall_style", ""),
                color_direction=data.get("color_direction", ""),
                priority=data.get("priority", ""),
                constraints=data.get("constraints", []),
                target_mood=data.get("target_mood", ""),
                stages=stages,
            )
        else:
            logger.warning(f"Director 输出无法解析: {raw[:200]}")
            brief = StyleBrief(overall_style="自然", priority="避免过度处理")

        # 如果 VLM 没有返回 stages，生成保守的默认阶段规划
        if not brief.stages:
            logger.warning("Director 未输出 stages，使用保守默认规划")
            brief.stages = [
                StageDecision(stage="stabilize", scope="skip", direction="默认跳过，避免画质损失"),
                StageDecision(stage="denoise", scope="skip", direction="默认跳过"),
                StageDecision(stage="color_correct", scope="global", direction="轻微校正"),
                StageDecision(stage="color_grade", scope="global", direction=brief.color_direction or "保持自然"),
                StageDecision(stage="sharpen", scope="skip", direction="默认跳过，避免伪影"),
            ]

        logger.info(
            f"Director 策略: 风格={brief.overall_style}, "
            f"色彩={brief.color_direction}, "
            f"优先={brief.priority}, "
            f"约束={brief.constraints}"
        )
        for sd in brief.stages:
            logger.info(f"  阶段 {sd.stage}: {sd.scope} — {sd.direction}")
        return brief

    def should_continue(
        self,
        critic_feedback: CriticFeedback | None,
        prev_score: float,
        iteration: int,
        max_iterations: int,
    ) -> bool:
        """纯规则判断是否继续外层循环（不调 VLM）"""
        if critic_feedback is None:
            return False

        # 达到最大轮数
        if iteration >= max_iterations:
            logger.info("Director: 达到最大外层迭代次数，终止")
            return False

        # 分数已经足够高
        if critic_feedback.overall_score > 0.85:
            logger.info(f"Director: 分数已达 {critic_feedback.overall_score:.2f} > 0.85，终止")
            return False

        # 没有发现新问题
        has_degraded = any(
            sc.verdict == "degraded"
            for sc in critic_feedback.segment_feedback.values()
            if isinstance(sc, dict) and sc.get("verdict") == "degraded"
        ) if isinstance(critic_feedback.segment_feedback, dict) else False

        # 兼容 segment_feedback 可能是 SegmentCritic 对象或 dict
        if not has_degraded:
            for val in critic_feedback.segment_feedback.values():
                if hasattr(val, "verdict") and val.verdict == "degraded":
                    has_degraded = True
                    break

        if not has_degraded and not critic_feedback.global_issues:
            logger.info("Director: 无 degraded 段且无全局问题，终止")
            return False

        logger.info("Director: 仍有改进空间，继续外层循环")
        return True

    def revise_brief(
        self,
        current_brief: StyleBrief,
        critic_feedback: CriticFeedback,
        segments: list[SegmentMetadata],
    ) -> StyleBrief:
        """redirect 时：根据 Critic 反馈调整风格策略"""
        logger.info("Director: 根据 Critic 反馈修订风格策略...")

        brief_text = (
            f"风格={current_brief.overall_style}, "
            f"色彩={current_brief.color_direction}, "
            f"优先={current_brief.priority}, "
            f"约束={current_brief.constraints}"
        )

        prompt = REVISE_PROMPT.format(
            current_brief=brief_text,
            global_issues=", ".join(critic_feedback.global_issues) or "无",
            suggestions=", ".join(critic_feedback.suggestions) or "无",
            route_reason=critic_feedback.route_reason,
        )

        response = self.llm.chat(
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=800,
        )

        raw = response.choices[0].message.content
        data = extract_json(raw)

        if data and isinstance(data, dict):
            brief = StyleBrief(
                overall_style=data.get("overall_style", current_brief.overall_style),
                color_direction=data.get("color_direction", current_brief.color_direction),
                priority=data.get("priority", current_brief.priority),
                constraints=data.get("constraints", current_brief.constraints),
                target_mood=data.get("target_mood", current_brief.target_mood),
            )
        else:
            logger.warning("Director 修订输出无法解析，保持原策略")
            brief = current_brief

        logger.info(f"Director 修订策略: 风格={brief.overall_style}, 优先={brief.priority}")
        return brief
