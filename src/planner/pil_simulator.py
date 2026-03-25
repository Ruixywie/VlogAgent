"""PIL 视觉预筛选：在关键帧上快速模拟 FFmpeg 滤镜效果

用于逐阶段候选筛选——在执行昂贵的 FFmpeg 操作之前，
先在关键帧上用 PIL 模拟效果，配合 CLIP+MLP 评分选出最优候选。

支持的模拟（对齐 basic_tools.py 的 FFmpeg 参数）：
- color_correct / color_grade / color_adjust: eq 滤镜 → PIL ImageEnhance
- white_balance: colortemperature → R/B 通道缩放
- denoise: hqdn3d → GaussianBlur 近似
- sharpen: unsharp → PIL ImageEnhance.Sharpness

不支持（返回 None）：
- stabilize: 需要运动补偿
- speed_adjust: 时间轴操作
- auto_color_harmonize: 需要全片统计
"""

import logging

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from src.models import EditAction

logger = logging.getLogger(__name__)

# 可 PIL 模拟的工具集合
SIMULATABLE_TOOLS = {
    "color_correct", "color_grade", "color_adjust",
    "white_balance", "denoise", "sharpen",
}


class PILSimulator:
    """在关键帧上快速模拟编辑效果（毫秒级）"""

    @staticmethod
    def can_simulate(tool_name: str) -> bool:
        return tool_name in SIMULATABLE_TOOLS

    @staticmethod
    def simulate(frame_bgr: np.ndarray, action: EditAction) -> np.ndarray | None:
        """在单帧上模拟 EditAction 的效果。

        参数：
            frame_bgr: OpenCV BGR 格式的帧
            action: 编辑动作

        返回：
            模拟后的 BGR 帧，不支持的工具返回 None
        """
        if not PILSimulator.can_simulate(action.tool_name):
            return None

        params = action.parameters if isinstance(action.parameters, dict) else {}

        # BGR → RGB → PIL
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        try:
            if action.tool_name in ("color_correct", "color_grade", "color_adjust"):
                pil_img = PILSimulator._simulate_eq(pil_img, params)
            elif action.tool_name == "white_balance":
                pil_img = PILSimulator._simulate_colortemp(pil_img, params)
            elif action.tool_name == "denoise":
                pil_img = PILSimulator._simulate_denoise(pil_img, params)
            elif action.tool_name == "sharpen":
                pil_img = PILSimulator._simulate_sharpen(pil_img, params)
        except Exception as e:
            logger.warning(f"PIL 模拟失败 [{action.tool_name}]: {e}")
            return None

        # PIL → RGB → BGR
        result_rgb = np.array(pil_img)
        return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _simulate_eq(img: Image.Image, params: dict) -> Image.Image:
        """模拟 FFmpeg eq 滤镜（brightness/contrast/saturation/gamma）"""
        brightness = params.get("brightness", 0)
        contrast = params.get("contrast", 1)
        saturation = params.get("saturation", 1)
        gamma = params.get("gamma", 1)

        # brightness: FFmpeg 加法偏移 → PIL 乘法近似
        if brightness != 0:
            factor = 1.0 + brightness * 2  # brightness=0.1 → factor=1.2
            img = ImageEnhance.Brightness(img).enhance(factor)

        # contrast: 直接对应
        if contrast != 1:
            img = ImageEnhance.Contrast(img).enhance(contrast)

        # saturation: 直接对应
        if saturation != 1:
            img = ImageEnhance.Color(img).enhance(saturation)

        # gamma: numpy 幂次变换
        if gamma != 1:
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = np.power(np.clip(arr, 1e-6, 1.0), 1.0 / gamma)
            img = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))

        return img

    @staticmethod
    def _simulate_colortemp(img: Image.Image, params: dict) -> Image.Image:
        """模拟 FFmpeg colortemperature 滤镜"""
        temperature = params.get("temperature", 6500)
        shift = (temperature - 6500) / 6500  # 归一化到约 [-1, 1]

        arr = np.array(img, dtype=np.float32)
        arr[:, :, 0] *= (1 + shift * 0.15)   # R
        arr[:, :, 2] *= (1 - shift * 0.15)   # B
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    @staticmethod
    def _simulate_denoise(img: Image.Image, params: dict) -> Image.Image:
        """模拟 FFmpeg hqdn3d 降噪（用高斯模糊近似）"""
        strength = params.get("strength", 4)
        radius = max(0.3, strength * 0.3)  # strength=4 → radius=1.2
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    @staticmethod
    def _simulate_sharpen(img: Image.Image, params: dict) -> Image.Image:
        """模拟 FFmpeg unsharp 锐化"""
        amount = params.get("amount", 1.0)
        # PIL Sharpness: 0=模糊, 1=原始, 2=锐化
        factor = 1.0 + amount  # amount=1.0 → factor=2.0
        return ImageEnhance.Sharpness(img).enhance(factor)
