"""VlogAgent 主循环：Perceiver → Planner → Executor → Evaluator 闭环"""

import logging
import os
import shutil

import yaml

from src.models import SegmentMetadata, EditPlan, EvaluationResult
from src.perceiver.video_analyzer import VideoAnalyzer
from src.perceiver.perceiver import Perceiver
from src.planner.mcts import MCTSPlanner
from src.executor.basic_tools import BasicTools
from src.executor.tool_selector import ToolSelector
from src.executor.compound_tools import CompoundTools
from src.evaluator.evaluator import Evaluator

logger = logging.getLogger(__name__)


class VlogAgent:
    def __init__(self, config_path: str = "configs/default.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # OpenAI 配置
        openai_cfg = self.config.get("openai", {})
        api_key = openai_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        base_url = openai_cfg.get("base_url")
        model = openai_cfg.get("model", "gpt-4o")

        llm_config = {"api_key": api_key, "base_url": base_url, "model": model}

        # 初始化各组件
        perceiver_cfg = {**self.config.get("perceiver", {}), **llm_config}
        perceiver_cfg["ffmpeg_path"] = self.config.get("executor", {}).get("ffmpeg_path", "ffmpeg")
        self.analyzer = VideoAnalyzer(perceiver_cfg)
        self.perceiver = Perceiver(llm_config)

        planner_cfg = {**self.config.get("planner", {}), **llm_config}
        self.planner = MCTSPlanner(planner_cfg)

        executor_cfg = self.config.get("executor", {})
        self.basic_tools = BasicTools(executor_cfg)
        self.tool_selector = ToolSelector(llm_config)
        self.compound_tools = CompoundTools(self.basic_tools)

        evaluator_cfg = {**self.config.get("evaluator", {}), **llm_config}
        self.evaluator = Evaluator(evaluator_cfg)

        self.max_iterations = self.config.get("agent", {}).get("max_iterations", 3)
        self.min_improvement = self.config.get("evaluator", {}).get("min_improvement", 0.01)
        self.max_no_improve = self.config.get("evaluator", {}).get("max_no_improve_rounds", 2)

    def _execute_plan(
        self,
        video_path: str,
        plan: EditPlan,
        segments: list[SegmentMetadata],
    ) -> str:
        """执行一个编辑方案，返回编辑后的视频路径"""
        current_video = video_path

        for action in plan.actions:
            # 用 ToolSelector 确保 action 有具体工具和参数
            action = self.tool_selector.resolve_action(action)

            logger.info(
                f"执行: [{action.tool_name}] {action.action_description} "
                f"→ {action.target_segment} | params={action.parameters}"
            )

            try:
                if action.tool_type == "compound":
                    if action.tool_name == "auto_color_harmonize":
                        current_video = self.compound_tools.auto_color_harmonize(
                            current_video, segments
                        )
                    else:
                        logger.warning(f"未知 compound 工具: {action.tool_name}")
                else:
                    # basic tool
                    registry = self.basic_tools.get_tool_registry()
                    if action.tool_name in registry:
                        tool_func = registry[action.tool_name]["func"]

                        if action.target_segment == "global":
                            current_video = tool_func(current_video, **action.parameters)
                        else:
                            # 对特定片段操作：先裁剪 → 操作 → 重新拼接
                            # MVP 阶段简化：直接对全视频应用（避免复杂的段级操作）
                            logger.info(
                                f"  (MVP简化) 对全视频应用 {action.tool_name}，"
                                f"目标段: {action.target_segment}"
                            )
                            current_video = tool_func(current_video, **action.parameters)
                    else:
                        logger.warning(f"未知工具: {action.tool_name}")
            except Exception as e:
                logger.error(f"执行失败 [{action.tool_name}]: {e}")
                # 跳过失败的动作，继续下一个
                continue

        return current_video

    def run(self, video_path: str, output_dir: str = "output") -> str:
        """
        主循环：
        1. 视频分析
        2. Perceiver → 候选指令
        3. Planner → MCTS 搜索
        4. Executor → 执行 Top-K
        5. Evaluator → 评分 + 选最优
        6. 闭环迭代
        """
        os.makedirs(output_dir, exist_ok=True)
        original_path = video_path
        analysis_dir = os.path.join(output_dir, "analysis")

        logger.info(f"========== VlogAgent 开始处理: {video_path} ==========")

        # 1. 视频分析
        logger.info("[Step 1] 视频分析...")
        segments = self.analyzer.analyze(video_path, output_dir=analysis_dir)
        logger.info(f"  检测到 {len(segments)} 个片段")

        prev_score = 0.0
        no_improve_count = 0
        best_video = video_path

        for iteration in range(self.max_iterations):
            logger.info(f"\n========== 闭环迭代 {iteration + 1}/{self.max_iterations} ==========")

            # 2. Perceiver：生成候选编辑指令
            logger.info("[Step 2] Perceiver: 生成候选指令...")
            candidates = self.perceiver.perceive(best_video, segments)
            if not candidates:
                logger.warning("Perceiver 未返回候选指令，终止")
                break
            logger.info(f"  生成 {len(candidates)} 条候选指令")

            # 3. Planner：MCTS 搜索
            logger.info("[Step 3] Planner: MCTS 搜索...")
            top_k_plans = self.planner.search(candidates, segments)
            if not top_k_plans:
                logger.warning("Planner 未找到可行方案，终止")
                break
            logger.info(f"  找到 {len(top_k_plans)} 个方案")

            # 4. Executor：执行 Top-K 方案
            logger.info("[Step 4] Executor: 执行方案...")
            results = []
            for i, plan in enumerate(top_k_plans):
                logger.info(f"  执行方案 {i+1}/{len(top_k_plans)}")
                try:
                    edited = self._execute_plan(best_video, plan, segments)
                    results.append((edited, plan))
                except Exception as e:
                    logger.error(f"  方案 {i+1} 执行失败: {e}")

            if not results:
                logger.warning("所有方案执行失败，终止")
                break

            # 5. Evaluator：评分
            logger.info("[Step 5] Evaluator: 评分...")
            best_result = None
            best_score = -1.0

            for edited_path, plan in results:
                eval_result = self.evaluator.evaluate(
                    original_path, edited_path, segments
                )
                if eval_result.overall_score > best_score:
                    best_score = eval_result.overall_score
                    best_result = (edited_path, eval_result)

            if best_result is None:
                break

            edited_path, eval_result = best_result
            logger.info(f"  本轮最佳分数: {best_score:.3f} (上轮: {prev_score:.3f})")

            # 6. 闭环判断
            if best_score > prev_score + self.min_improvement:
                logger.info(f"  ✓ 有提升 (+{best_score - prev_score:.3f})，采用本轮结果")
                best_video = edited_path
                prev_score = best_score
                no_improve_count = 0
            else:
                no_improve_count += 1
                logger.info(f"  ✗ 无明显提升 (连续 {no_improve_count} 轮)")
                if no_improve_count >= self.max_no_improve:
                    logger.info("  达到最大无提升轮数，终止闭环")
                    break

        # 复制最终结果到输出目录
        final_output = os.path.join(output_dir, f"final_{os.path.basename(video_path)}")
        if best_video != video_path:
            shutil.copy2(best_video, final_output)
            logger.info(f"\n========== 完成！输出: {final_output} ==========")
            return final_output
        else:
            logger.info("\n========== 完成！未进行有效编辑，返回原视频 ==========")
            return video_path
