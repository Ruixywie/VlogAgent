"""VlogAgent 核心数据模型 + 通用工具"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class FallbackLLM:
    """带自动降级的 LLM 客户端。

    按配置的模型优先级列表调用，当前模型报错（配额用尽、认证失败等）
    自动切换到下一个模型。所有模型共享同一 API Key 和 base_url。
    """

    def __init__(self, config: dict):
        self.client = OpenAI(
            api_key=config.get("api_key", None),
            base_url=config.get("base_url", None),
        )
        # 模型优先级列表
        self.models = config.get("models", [config.get("model", "gpt-4o")])
        if isinstance(self.models, str):
            self.models = [self.models]
        self._current_idx = 0

    @property
    def current_model(self) -> str:
        return self.models[self._current_idx]

    def chat(self, **kwargs) -> object:
        """调用 chat completions，失败时自动切换模型。

        传入的参数与 openai.chat.completions.create 一致，
        但不需要传 model 参数（自动填充当前模型）。
        """
        last_error = None
        start_idx = self._current_idx

        for _ in range(len(self.models)):
            model = self.models[self._current_idx]
            kwargs["model"] = model
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as e:
                error_str = str(e).lower()
                # 判断是否为配额/认证类错误（应该切换模型）
                switchable = any(keyword in error_str for keyword in [
                    "quota", "limit", "exceeded", "insufficient",
                    "429", "401", "403", "billing", "credits",
                ])
                if switchable and self._current_idx < len(self.models) - 1:
                    old_model = model
                    self._current_idx += 1
                    new_model = self.models[self._current_idx]
                    logger.warning(
                        f"模型 {old_model} 不可用 ({type(e).__name__}), "
                        f"切换到 {new_model}"
                    )
                    last_error = e
                    continue
                else:
                    raise  # 非配额错误或已是最后一个模型，直接抛出

        raise last_error  # 所有模型都失败


def extract_json(text: str) -> dict | list | None:
    """从 LLM 输出中提取 JSON（兼容 markdown 代码块和前后多余文字）"""
    # 尝试提取 ```json ... ``` 代码块
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()

    # 尝试找到第一个 { 或 [ 开始的 JSON
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        if start == -1:
            continue
        # 从后往前找匹配的结束符
        end = text.rfind(end_char)
        if end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


@dataclass
class SegmentMetadata:
    """视频片段元数据"""
    seg_id: int
    time_range: tuple[float, float]          # (start_sec, end_sec)
    mean_brightness: float = 0.0             # Y通道均值 [0, 255]
    color_temp_est: float = 0.0              # R/B 通道比值估算色温
    sharpness_score: float = 0.0             # Laplacian 方差
    stability_score: float = 0.0             # 光流场方差（越小越稳）
    noise_level: float = 0.0                 # 噪声估计
    has_speech: bool = False
    speech_text: str = ""
    semantic_description: str = ""           # VLM 生成的语义描述
    emotion_tag: str = ""                    # 情感标签
    keyframe_paths: list[str] = field(default_factory=list)  # 关键帧图片路径


@dataclass
class EditAction:
    """单条编辑动作"""
    action_description: str                  # 语义指令（如 "提升亮度使画面更通透"）
    target_segment: str = "global"           # "seg-0" / "seg-1" / "global"
    tool_type: str = "basic"                 # "basic" / "ai" / "compound"
    tool_name: str = ""                      # "color_adjust" / "stabilize" / ...
    parameters: dict = field(default_factory=dict)  # 工具参数


@dataclass
class EditPlan:
    """编辑方案（有序动作序列）"""
    actions: list[EditAction] = field(default_factory=list)
    estimated_score: float = 0.0


@dataclass
class EvaluationResult:
    """评估结果"""
    visual_quality: float = 0.0              # 视觉质量分 [0, 1]
    content_fidelity: float = 0.0            # 内容保真度 [0, 1]
    inter_segment_consistency: float = 0.0   # 段间一致性 [0, 1]
    audio_integrity: float = 0.0             # 音频完整性 [0, 1]
    aesthetic: float = 0.0                   # 整体美学 [0, 1]
    overall_score: float = 0.0               # 加权总分

    def compute_overall(self, weights: dict) -> float:
        self.overall_score = (
            self.visual_quality * weights.get("visual_quality", 0.25)
            + self.content_fidelity * weights.get("content_fidelity", 0.25)
            + self.inter_segment_consistency * weights.get("inter_segment_consistency", 0.20)
            + self.audio_integrity * weights.get("audio_integrity", 0.10)
            + self.aesthetic * weights.get("aesthetic", 0.20)
        )
        return self.overall_score
