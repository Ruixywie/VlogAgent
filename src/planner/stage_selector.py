"""逐阶段候选选择器：PIL 模拟 + CLIP+MLP 打分

替代 MCTS 树搜索。对单阶段的 2-3 条候选：
1. PIL 在关键帧上模拟编辑效果（毫秒级）
2. CLIP+MLP 对 (原始帧, 模拟帧) 打分
3. 返回最高分候选

保留了 MCTS "先模拟再执行"的核心思想，
但用视觉感知评分替代了无意义的硬编码规则。
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from src.models import EditAction, SegmentMetadata, normalize_seg_id
from src.planner.pil_simulator import PILSimulator
from src.planner.mcts import LocalScorer  # 复用 CLIP+MLP 评分

logger = logging.getLogger(__name__)


class StageSelector:
    """逐阶段视觉预筛选"""

    def __init__(self, config: dict = None):
        config = config or {}
        self.scorer = LocalScorer(config.get("scorer_model_path"))
        # 关键帧缓存（整个 run 过程只加载一次）
        self._keyframe_cache: dict[int, np.ndarray] = {}  # seg_id → BGR frame
        self._feature_cache: dict[int, object] = {}       # seg_id → CLIP feature

    def preload_keyframes(self, segments: list[SegmentMetadata]):
        """预加载每段中间帧的 BGR 像素和 CLIP 特征（一次性）"""
        if self._keyframe_cache:
            return  # 已加载

        for seg in segments:
            if not seg.keyframe_paths:
                continue
            # 取中间帧
            mid_idx = len(seg.keyframe_paths) // 2
            kf_path = seg.keyframe_paths[mid_idx]
            if Path(kf_path).exists():
                frame = cv2.imread(kf_path)
                if frame is not None:
                    self._keyframe_cache[seg.seg_id] = frame
                    self._feature_cache[seg.seg_id] = self.scorer.extract_frame_feature(frame)

        logger.info(f"StageSelector: 预加载 {len(self._keyframe_cache)} 段关键帧")

    def select_best(
        self,
        candidates: list[EditAction],
        segments: list[SegmentMetadata],
    ) -> tuple[EditAction | None, float]:
        """对候选列表打分，返回 (最优候选, 分数)。

        候选为空返回 (None, 0.0)。
        """
        if not candidates:
            return None, 0.0

        self.preload_keyframes(segments)

        best_action = None
        best_score = -1.0
        seg_map = {seg.seg_id: seg for seg in segments}

        for action in candidates:
            score = self._score_action(action, segments, seg_map)
            logger.debug(f"  预筛选: [{action.tool_name}] → {action.target_segment} = {score:.3f}")
            if score > best_score:
                best_score = score
                best_action = action

        if best_action:
            logger.info(
                f"  预筛选最优: [{best_action.tool_name}] "
                f"{best_action.target_segment} score={best_score:.3f}"
            )

        return best_action, best_score

    def _score_action(
        self,
        action: EditAction,
        segments: list[SegmentMetadata],
        seg_map: dict,
    ) -> float:
        """对单条候选打分"""
        target = normalize_seg_id(action.target_segment)

        # 确定要评估的段
        if target == "global":
            eval_seg_ids = list(self._keyframe_cache.keys())
        else:
            # 解析 seg_id 数字
            try:
                seg_num = int(target.replace("seg-", ""))
                eval_seg_ids = [seg_num] if seg_num in self._keyframe_cache else []
            except ValueError:
                eval_seg_ids = list(self._keyframe_cache.keys())

        if not eval_seg_ids:
            return 0.5  # 无法评估，返回基准分

        # 如果不可模拟（stabilize 等），返回基准分
        if not PILSimulator.can_simulate(action.tool_name):
            return 0.5

        scores = []
        for seg_id in eval_seg_ids:
            frame = self._keyframe_cache.get(seg_id)
            orig_feat = self._feature_cache.get(seg_id)
            if frame is None or orig_feat is None:
                continue

            # PIL 模拟
            simulated = PILSimulator.simulate(frame, action)
            if simulated is None:
                scores.append(0.5)
                continue

            # CLIP+MLP 打分
            sim_feat = self.scorer.extract_frame_feature(simulated)
            score = self.scorer.score_edit(orig_feat, sim_feat)
            scores.append(score)

        return float(np.mean(scores)) if scores else 0.5
