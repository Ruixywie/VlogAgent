"""Perceiver：VLM 观察 + 生成候选编辑指令"""

import base64
import json
import logging
from pathlib import Path

from openai import OpenAI

from src.models import SegmentMetadata, EditAction

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个专业的视频后期制作专家。你将收到一段视频的分析数据（场景分割、质量指标、关键帧），
需要根据这些信息生成一组候选编辑指令，来提升视频的整体质量和美感。

可用的编辑工具：
- color_adjust: 调整亮度(brightness)、对比度(contrast)、饱和度(saturation)、Gamma
- white_balance: 调整色温(temperature)
- denoise: 视频降噪(strength)
- sharpen: 锐化(amount)
- stabilize: 视频防抖(smoothing)
- speed_adjust: 变速(factor)
- auto_color_harmonize: 统一全片色彩风格（compound工具）

输出要求：
1. 生成 8-15 条候选编辑指令
2. 每条指令包含：action_description（语义描述）、target_segment（"seg-N" 或 "global"）、tool_type（"basic"/"compound"）、tool_name、parameters
3. 优先处理：曝光问题 > 色彩一致性 > 降噪 > 锐化 > 防抖 > 变速
4. 参数要合理，不要过度处理
5. 以 JSON 数组格式返回"""


class Perceiver:
    def __init__(self, config: dict):
        self.client = OpenAI(
            api_key=config.get("api_key", None),
            base_url=config.get("base_url", None),
        )
        self.model = config.get("model", "gpt-4o")
        self.max_candidates = config.get("max_candidates", 15)
        self.min_candidates = config.get("min_candidates", 8)

    def _encode_image(self, image_path: str) -> str:
        """将图片编码为 base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _build_storyboard_message(
        self, segments: list[SegmentMetadata]
    ) -> list[dict]:
        """构建包含关键帧的多模态消息"""
        content = []

        # 文本部分：视频分析摘要
        analysis_text = "## 视频分析数据\n\n"
        for seg in segments:
            analysis_text += (
                f"### 片段 {seg.seg_id} ({seg.time_range[0]:.1f}s - {seg.time_range[1]:.1f}s)\n"
                f"- 亮度: {seg.mean_brightness:.1f}/255\n"
                f"- 色温估计: {seg.color_temp_est:.2f} (R/B比)\n"
                f"- 清晰度: {seg.sharpness_score:.1f}\n"
                f"- 稳定性: {seg.stability_score:.2f} (光流方差，越小越稳)\n"
                f"- 噪声: {seg.noise_level:.1f}\n"
                f"- 语音: {'有' if seg.has_speech else '无'}"
            )
            if seg.speech_text:
                analysis_text += f" - \"{seg.speech_text[:100]}\""
            analysis_text += "\n\n"

        content.append({"type": "text", "text": analysis_text})

        # 图片部分：关键帧 Storyboard
        for seg in segments:
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

    def perceive(
        self, video_path: str, segments: list[SegmentMetadata]
    ) -> list[EditAction]:
        """观察视频分析数据 + 关键帧，生成候选编辑指令"""
        logger.info(f"Perceiver: 生成候选编辑指令 (共 {len(segments)} 个片段)")

        content = self._build_storyboard_message(segments)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
        )

        # 解析返回的 JSON
        raw = response.choices[0].message.content
        try:
            data = json.loads(raw)
            # 支持 {"actions": [...]} 或直接 [...]
            action_list = data if isinstance(data, list) else data.get("actions", [])
        except json.JSONDecodeError:
            logger.error(f"Perceiver 返回无法解析的 JSON: {raw[:200]}")
            return []

        # 转换为 EditAction
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

        logger.info(f"Perceiver: 生成了 {len(actions)} 条候选指令")
        return actions
