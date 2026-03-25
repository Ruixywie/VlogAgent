"""Perceiver：两阶段 VLM 观察 + 编辑指令生成

阶段 1（观察）：VLM 观看完整帧序列，描述每段画面内容、光线、构图、情绪、问题
阶段 2（建议）：基于观察描述 + 技术指标，生成候选编辑指令
"""

import logging
from pathlib import Path

from src.models import (
    SegmentMetadata, EditAction, StyleBrief, StageDecision, CriticFeedback,
    FallbackLLM, extract_json, TOOL_TO_STAGE,
)
from src.utils import encode_image, build_frames_content, build_metrics_text

logger = logging.getLogger(__name__)

# ── 阶段 1：画面观察 ────────────────────────────────────

OBSERVE_PROMPT = """你是一位专业摄影师和视频导演。请仔细观察以下视频的关键帧序列，对每个片段进行详细的视觉分析。

对每个片段，请描述：
1. **画面内容**：场景中有什么（人物、物体、环境）
2. **光线与曝光**：光线方向、是否过曝/欠曝、阴影细节
3. **色彩表现**：色调（冷/暖）、饱和度是否合适、是否有色偏
4. **构图与运镜**：构图方式、镜头是否稳定、运动方向
5. **情绪与风格**：画面传达的情绪、适合的后期风格方向
6. **存在的问题**：噪点、模糊、抖动、曝光不均、色彩不一致等

同时请评估全片的整体风格一致性：各片段之间的色彩/光线/风格是否协调。

以自然语言详细描述，不要返回 JSON。"""

# ── 阶段 2：编辑建议 ────────────────────────────────────

SUGGEST_PROMPT = """你是一位专业的视频后期制作专家。基于你刚才对视频画面的观察分析，现在需要给出具体的编辑方案。

## 技术指标参考
{metrics_text}

## 可用编辑工具（按专业后期阶段分类）

### 阶段 1 — 稳定化
- stabilize: 视频防抖(smoothing[5,30])，仅支持全局应用

### 阶段 2 — 降噪
- denoise: 视频降噪(strength[1,10])

### 阶段 3 — 技术校正
- color_correct: 技术性色彩校正 — brightness[-0.3,0.3], contrast[0.5,2.0], saturation[0.5,2.0], gamma[0.5,2.0]
  用于：曝光校正、灰度一致性调整（让画面"看起来正常"）
- white_balance: 调整色温(temperature[2000,10000] K值)

### 阶段 4 — 创意调色
- color_grade: 创意调色 — brightness[-0.3,0.3], contrast[0.5,2.0], saturation[0.5,2.0], gamma[0.5,2.0]
  用于：风格化色调、情绪渲染、电影感调色（让画面"看起来好看"）
- auto_color_harmonize: 统一全片色彩风格（compound工具，无参数）

### 阶段 5 — 锐化
- sharpen: 锐化(amount[0.5,3.0])

## 要求
1. 基于你对画面内容的理解来决定编辑方案，而不是机械地根据数值调整
2. 生成 8-15 条候选编辑指令
3. 每条指令包含：action_description、target_segment（"seg-N" 或 "global"）、tool_type（"basic"/"compound"）、tool_name、parameters、stage（所属阶段）
4. 参数要合理保守，避免过度处理
5. 区分 color_correct（技术校正）和 color_grade（创意调色），不要混用
6. 以 JSON 格式返回：{{"actions": [...]}}，不要输出其他文字"""


class Perceiver:
    def __init__(self, llm: FallbackLLM):
        self.llm = llm

    # 帧编码和指标构建已提取到 src/utils.py，以下为兼容别名
    @staticmethod
    def _encode_image(image_path: str) -> str:
        return encode_image(image_path)

    @staticmethod
    def _build_frames_content(segments: list[SegmentMetadata]) -> list[dict]:
        return build_frames_content(segments)

    @staticmethod
    def _build_metrics_text(segments: list[SegmentMetadata]) -> str:
        return build_metrics_text(segments)

    # ── 阶段 1：观察 ────────────────────────────────────

    def observe(self, segments: list[SegmentMetadata]) -> str:
        """VLM 观看帧序列，输出画面描述"""
        logger.info("Perceiver 阶段1: VLM 观察画面...")

        content = self._build_frames_content(segments)

        response = self.llm.chat(
            messages=[
                {"role": "system", "content": OBSERVE_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0.5,
            max_tokens=1500,  # 限制观察描述长度
        )

        observation = response.choices[0].message.content
        logger.info(f"Perceiver 阶段1 完成: 观察描述 {len(observation)} 字")
        return observation

    # ── 阶段 2：建议 ────────────────────────────────────

    def suggest(
        self,
        observation: str,
        segments: list[SegmentMetadata],
        style_brief: StyleBrief | None = None,
        critic_feedback: CriticFeedback | None = None,
    ) -> list[EditAction]:
        """基于观察描述 + 技术指标 + Director约束 + Critic反馈，生成编辑指令"""
        logger.info("Editor: 生成候选编辑方案...")

        metrics_text = self._build_metrics_text(segments)
        system_prompt = SUGGEST_PROMPT.format(metrics_text=metrics_text)

        # 注入 Director 风格约束
        user_parts = [f"## 你的画面观察\n\n{observation}\n"]

        if style_brief:
            constraints_text = "\n".join(f"  - {c}" for c in style_brief.constraints) if style_brief.constraints else "  无"
            user_parts.append(
                f"## 导演风格指令\n"
                f"- 整体风格：{style_brief.overall_style}\n"
                f"- 色彩方向：{style_brief.color_direction}\n"
                f"- 编辑优先级：{style_brief.priority}\n"
                f"- 目标情绪：{style_brief.target_mood}\n"
                f"- 约束条件（必须遵守）：\n{constraints_text}\n"
            )

            # 注入分阶段规划（Stage-Aware Planning）
            if style_brief.stages:
                stage_lines = []
                for sd in style_brief.stages:
                    if sd.scope == "skip":
                        stage_lines.append(f"  - {sd.stage}: 跳过")
                    else:
                        targets = f"（目标：{', '.join(sd.target_segments)}）" if sd.target_segments else ""
                        stage_lines.append(f"  - {sd.stage} [{sd.scope}]: {sd.direction}{targets}")
                user_parts.append(
                    "## 分阶段规划（严格按此顺序和范围生成候选）\n"
                    + "\n".join(stage_lines) + "\n"
                    + "请按照以上阶段顺序生成候选，scope=skip 的阶段不要生成任何候选。\n"
                )

        # 注入 Critic 反馈（段级反馈，不写死参数）
        if critic_feedback and critic_feedback.segment_feedback:
            degraded_parts = []
            improved_parts = []
            for seg_id, sc in critic_feedback.segment_feedback.items():
                verdict = sc.verdict if hasattr(sc, "verdict") else sc.get("verdict", "?")
                reason = sc.reason if hasattr(sc, "reason") else sc.get("reason", "")
                if verdict == "degraded":
                    degraded_parts.append(f"  - {seg_id}: 变差了 — {reason}")
                elif verdict == "improved":
                    improved_parts.append(f"  - {seg_id}: 改善了 — {reason}")

            if degraded_parts or improved_parts:
                feedback_text = "## 上轮评审反馈（必须参考）\n"
                if degraded_parts:
                    feedback_text += "变差的片段（避免对这些片段重复同类操作，可以尝试更轻的参数或换其他工具）：\n"
                    feedback_text += "\n".join(degraded_parts) + "\n"
                if improved_parts:
                    feedback_text += "改善的片段（这些操作有效，可以保持）：\n"
                    feedback_text += "\n".join(improved_parts) + "\n"
                if critic_feedback.global_issues:
                    feedback_text += "全局问题：" + "，".join(critic_feedback.global_issues) + "\n"
                user_parts.append(feedback_text)

        user_parts.append("请基于以上信息，给出编辑方案。")

        response = self.llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "\n".join(user_parts)},
            ],
            temperature=0.7,
            max_tokens=1500,
        )

        raw = response.choices[0].message.content
        data = extract_json(raw)
        if data is None:
            logger.error(f"Perceiver 阶段2 返回无法解析: {raw[:200]}")
            return []

        action_list = data if isinstance(data, list) else data.get("actions", [])

        actions = []
        for item in action_list:
            # VLM 有时返回 list 而非 string，统一处理
            target = item.get("target_segment", "global")
            if isinstance(target, list):
                target = target[0] if target else "global"
            params = item.get("parameters", {})
            if not isinstance(params, dict):
                params = {}
            tool_name = str(item.get("tool_name", ""))
            stage = str(item.get("stage", ""))
            # VLM 未返回 stage 时从工具名推导
            if not stage:
                stage = TOOL_TO_STAGE.get(tool_name, "") or ""
            action = EditAction(
                action_description=str(item.get("action_description", "")),
                target_segment=str(target),
                tool_type=str(item.get("tool_type", "basic")),
                tool_name=tool_name,
                parameters=params,
                stage=stage,
            )
            actions.append(action)

        logger.info(f"Perceiver 阶段2 完成: {len(actions)} 条候选指令")
        return actions

    # ── 主入口 ──────────────────────────────────────────

    def perceive(
        self,
        video_path: str,
        segments: list[SegmentMetadata],
        style_brief: StyleBrief | None = None,
        critic_feedback: CriticFeedback | None = None,
    ) -> tuple[str, list[EditAction]]:
        """两阶段感知：观察 → 建议，返回 (观察描述, 候选指令列表)"""
        observation = self.observe(segments)
        actions = self.suggest(observation, segments, style_brief, critic_feedback)
        return observation, actions

    # ── 逐阶段生成 ─────────────────────────────────

    # 每阶段对应的工具说明
    STAGE_TOOL_DESC = {
        "stabilize": "- stabilize: 视频防抖(smoothing[5,30])，仅支持全局应用",
        "denoise": "- denoise: 视频降噪(strength[1,10])",
        "color_correct": (
            "- color_correct: 技术性色彩校正 — brightness[-0.3,0.3], contrast[0.5,2.0], saturation[0.5,2.0], gamma[0.5,2.0]\n"
            "  用于：曝光校正、灰度一致性（让画面'看起来正常'）\n"
            "- white_balance: 调整色温(temperature[2000,10000] K值)"
        ),
        "color_grade": (
            "- color_grade: 创意调色 — brightness[-0.3,0.3], contrast[0.5,2.0], saturation[0.5,2.0], gamma[0.5,2.0]\n"
            "  用于：风格化色调、情绪渲染（让画面'看起来好看'）\n"
            "- auto_color_harmonize: 统一全片色彩风格（compound工具，无参数）"
        ),
        "sharpen": "- sharpen: 锐化(amount[0.5,3.0])",
    }

    def suggest_for_stage(
        self,
        observation: str,
        segments: list[SegmentMetadata],
        stage_decision: StageDecision,
        style_brief: StyleBrief | None = None,
        critic_feedback: CriticFeedback | None = None,
    ) -> list[EditAction]:
        """为单个阶段生成 2-3 条候选编辑动作"""
        stage = stage_decision.stage
        logger.info(f"Editor: 为阶段 {stage} 生成候选...")

        # 构建阶段专属 prompt
        tool_desc = self.STAGE_TOOL_DESC.get(stage, "")
        scope_desc = f"范围: {stage_decision.scope}"
        if stage_decision.target_segments:
            scope_desc += f"（目标段: {', '.join(stage_decision.target_segments)}）"

        metrics_text = self._build_metrics_text(segments)

        system_prompt = (
            f"你是专业视频后期制作专家，当前处理 **{stage}** 阶段。\n\n"
            f"## 技术指标\n{metrics_text}\n\n"
            f"## 当前阶段可用工具\n{tool_desc}\n\n"
            f"## 要求\n"
            f"1. 只使用当前阶段的工具，不要使用其他阶段的工具\n"
            f"2. 生成 2-3 条候选编辑指令，参数各不相同（用于对比选择最优）\n"
            f"3. 参数要合理保守，避免过度处理\n"
            f"4. 以 JSON 格式返回：{{\"actions\": [...]}}\n"
            f"5. 每条包含：action_description, target_segment, tool_type, tool_name, parameters, stage"
        )

        user_parts = [
            f"## 画面观察\n{observation[:500]}\n",  # 截取观察描述避免过长
            f"## 阶段指令\n- 阶段: {stage}\n- {scope_desc}\n- 方向: {stage_decision.direction}\n",
        ]

        # 注入 Critic 反馈（仅与当前阶段相关的部分）
        if critic_feedback and critic_feedback.suggestions:
            user_parts.append(
                f"## 上轮反馈\n" +
                "\n".join(f"- {s}" for s in critic_feedback.suggestions) + "\n"
            )

        user_parts.append("请生成 2-3 条不同参数的候选方案。")

        response = self.llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "\n".join(user_parts)},
            ],
            temperature=0.7,
            max_tokens=400,
        )

        raw = response.choices[0].message.content
        data = extract_json(raw)
        if data is None:
            logger.warning(f"Editor 阶段 {stage} 输出无法解析: {raw[:200]}")
            return []

        action_list = data if isinstance(data, list) else data.get("actions", [])

        actions = []
        for item in action_list:
            target = item.get("target_segment", "global")
            if isinstance(target, list):
                target = target[0] if target else "global"
            params = item.get("parameters", {})
            if not isinstance(params, dict):
                params = {}
            tool_name = str(item.get("tool_name", ""))
            action = EditAction(
                action_description=str(item.get("action_description", "")),
                target_segment=str(target),
                tool_type=str(item.get("tool_type", "basic")),
                tool_name=tool_name,
                parameters=params,
                stage=stage,  # 强制设为当前阶段
            )
            actions.append(action)

        logger.info(f"Editor 阶段 {stage}: {len(actions)} 条候选")
        return actions
