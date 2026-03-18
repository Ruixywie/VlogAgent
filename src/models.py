"""VlogAgent 核心数据模型"""

from dataclasses import dataclass, field
from typing import Optional


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
