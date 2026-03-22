"""Perceiver：两阶段 VLM 观察 + 编辑指令生成

阶段 1（观察）：VLM 观看完整帧序列，描述每段画面内容、光线、构图、情绪、问题
阶段 2（建议）：基于观察描述 + 技术指标，生成候选编辑指令
"""

import base64
import logging
from pathlib import Path

from src.models import SegmentMetadata, EditAction, FallbackLLM, extract_json

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

## 可用编辑工具
- color_adjust: 调整亮度(brightness[-0.3,0.3])、对比度(contrast[0.5,2.0])、饱和度(saturation[0.5,2.0])、Gamma(gamma[0.5,2.0])
- white_balance: 调整色温(temperature[2000,10000] K值)
- denoise: 视频降噪(strength[1,10])
- sharpen: 锐化(amount[0.5,3.0])
- stabilize: 视频防抖(smoothing[5,30])
- speed_adjust: 变速(factor[0.25,4.0], 1.0=原速)
- auto_color_harmonize: 统一全片色彩风格（compound工具，无参数）

## 要求
1. 基于你对画面内容的理解来决定编辑方案，而不是机械地根据数值调整
   - 例如：日落场景的暖色偏高是风格需要，不应该"纠正"
   - 例如：运动场景的轻微抖动可能是手持风格，不一定需要防抖
2. 生成 8-15 条候选编辑指令
3. 每条指令包含：action_description（语义描述，解释为什么要这样做）、target_segment（"seg-N" 或 "global"）、tool_type（"basic"/"compound"）、tool_name、parameters
4. 参数要合理保守，避免过度处理
5. 以 JSON 格式返回：{{"actions": [...]}}，不要输出其他文字"""


class Perceiver:
    def __init__(self, llm: FallbackLLM):
        self.llm = llm

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _build_frames_content(self, segments: list[SegmentMetadata]) -> list[dict]:
        """构建带标注的帧序列（告诉 VLM 每帧属于哪个片段）"""
        content = []
        for seg in segments:
            # 片段标题
            content.append({
                "type": "text",
                "text": f"--- 片段 {seg.seg_id} ({seg.time_range[0]:.1f}s - {seg.time_range[1]:.1f}s) ---",
            })
            # 该片段的关键帧
            for kf_path in seg.keyframe_paths:
                if Path(kf_path).exists():
                    b64 = self._encode_image(kf_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": "low",
                        },
                    })
        return content

    def _build_metrics_text(self, segments: list[SegmentMetadata]) -> str:
        """构建技术指标文本"""
        lines = []
        for seg in segments:
            lines.append(
                f"片段 {seg.seg_id} ({seg.time_range[0]:.1f}s-{seg.time_range[1]:.1f}s): "
                f"亮度={seg.mean_brightness:.0f}/255, "
                f"色温R/B比={seg.color_temp_est:.2f}, "
                f"清晰度={seg.sharpness_score:.0f}, "
                f"稳定性={seg.stability_score:.2f}(光流方差), "
                f"噪声={seg.noise_level:.1f}"
                + (f", 语音: \"{seg.speech_text[:60]}\"" if seg.has_speech else "")
            )
        return "\n".join(lines)

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
        self, observation: str, segments: list[SegmentMetadata]
    ) -> list[EditAction]:
        """基于观察描述 + 技术指标，生成编辑指令"""
        logger.info("Perceiver 阶段2: 生成编辑建议...")

        metrics_text = self._build_metrics_text(segments)
        system_prompt = SUGGEST_PROMPT.format(metrics_text=metrics_text)

        response = self.llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"## 你的画面观察\n\n{observation}\n\n"
                        f"请基于以上观察，给出编辑方案。"
                    ),
                },
            ],
            temperature=0.7,
            max_tokens=1500,  # 限制建议长度
        )

        raw = response.choices[0].message.content
        data = extract_json(raw)
        if data is None:
            logger.error(f"Perceiver 阶段2 返回无法解析: {raw[:200]}")
            return []

        action_list = data if isinstance(data, list) else data.get("actions", [])

        actions = []
        for item in action_list:
            action = EditAction(
                action_description=item.get("action_description", ""),
                target_segment=item.get("target_segment", "global"),
                tool_type=item.get("tool_type", "basic"),
                tool_name=item.get("tool_name", ""),
                parameters=item.get("parameters", {}),
            )
            actions.append(action)

        logger.info(f"Perceiver 阶段2 完成: {len(actions)} 条候选指令")
        return actions

    # ── 主入口 ──────────────────────────────────────────

    def perceive(
        self, video_path: str, segments: list[SegmentMetadata]
    ) -> tuple[str, list[EditAction]]:
        """两阶段感知：观察 → 建议，返回 (观察描述, 候选指令列表)"""
        observation = self.observe(segments)
        actions = self.suggest(observation, segments)
        return observation, actions
