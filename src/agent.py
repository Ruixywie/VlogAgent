"""VlogAgent 主循环：Perceiver → Planner → Executor → Evaluator 闭环

核心设计：
- 搜索阶段直接在原视频上执行和评估（搜索结果作为探索用，不保留）
- 每轮采纳的方案累积到 action_chain 中，下一轮在累积结果上继续探索
- 搜索结束后，在原始视频上一次性执行完整 action_chain（滤镜合并，单次编码）
"""

import logging
import os
import shutil

import yaml

from src.models import SegmentMetadata, EditPlan, EditAction, EvaluationResult, FallbackLLM
from src.perceiver.video_analyzer import VideoAnalyzer
from src.perceiver.perceiver import Perceiver
from src.planner.mcts import MCTSPlanner
from src.executor.basic_tools import BasicTools, ENCODE_ARGS
from src.executor.tool_selector import ToolSelector
from src.executor.compound_tools import CompoundTools
from src.evaluator.evaluator import Evaluator
from src.run_logger import RunLogger

logger = logging.getLogger(__name__)


class VlogAgent:
    def __init__(self, config_path: str = "configs/default.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # LLM 配置（带自动降级）
        openai_cfg = self.config.get("openai", {})
        api_key = openai_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        base_url = openai_cfg.get("base_url")
        models = openai_cfg.get("models", [openai_cfg.get("model", "gpt-4o")])

        llm_config = {"api_key": api_key, "base_url": base_url, "models": models}
        # 创建全局共享的 FallbackLLM 实例
        self.llm = FallbackLLM(llm_config)

        # 初始化各组件（共享同一个 FallbackLLM 实例）
        perceiver_cfg = {**self.config.get("perceiver", {})}
        perceiver_cfg["ffmpeg_path"] = self.config.get("executor", {}).get("ffmpeg_path", "ffmpeg")
        self.analyzer = VideoAnalyzer(perceiver_cfg)
        self.perceiver = Perceiver(self.llm)

        planner_cfg = self.config.get("planner", {})
        self.planner = MCTSPlanner(planner_cfg)

        executor_cfg = self.config.get("executor", {})
        self.basic_tools = BasicTools(executor_cfg)
        self.tool_selector = ToolSelector(self.llm)
        self.compound_tools = CompoundTools(self.basic_tools)

        evaluator_cfg = self.config.get("evaluator", {})
        self.evaluator = Evaluator(evaluator_cfg, self.llm)

        self.max_iterations = self.config.get("agent", {}).get("max_iterations", 3)
        self.min_improvement = self.config.get("evaluator", {}).get("min_improvement", 0.01)
        self.max_no_improve = self.config.get("evaluator", {}).get("max_no_improve_rounds", 2)
        self.ffmpeg = executor_cfg.get("ffmpeg_path", "ffmpeg")

    # ══════════════════════════════════════════════════════
    # 搜索阶段：在原视频上执行方案（结果仅用于评估，不保留）
    # ══════════════════════════════════════════════════════

    def _execute_plan_for_search(
        self,
        video_path: str,
        plan: EditPlan,
        segments: list[SegmentMetadata],
    ) -> tuple[str, list[dict]]:
        """执行方案用于搜索评估，返回 (结果路径, 执行日志)"""
        current_video = video_path
        actions_log = []

        for action in plan.actions:
            action = self.tool_selector.resolve_action(action)

            log_entry = {
                "tool_name": action.tool_name,
                "description": action.action_description,
                "parameters": action.parameters,
                "target": action.target_segment,
                "success": False,
                "error": "",
            }

            try:
                if action.tool_type == "compound":
                    if action.tool_name == "auto_color_harmonize":
                        current_video = self.compound_tools.auto_color_harmonize(
                            current_video, segments
                        )
                        log_entry["success"] = True
                    else:
                        log_entry["error"] = f"未知 compound 工具: {action.tool_name}"
                else:
                    registry = self.basic_tools.get_tool_registry()
                    if action.tool_name in registry:
                        tool_func = registry[action.tool_name]["func"]
                        current_video = tool_func(current_video, **action.parameters)
                        log_entry["success"] = True
                    else:
                        log_entry["error"] = f"未知工具: {action.tool_name}"
            except Exception as e:
                log_entry["error"] = str(e)
                logger.error(f"执行失败 [{action.tool_name}]: {e}")

            actions_log.append(log_entry)

        return current_video, actions_log

    # ══════════════════════════════════════════════════════
    # 最终执行：在原始视频上一次性应用完整动作链
    # ══════════════════════════════════════════════════════

    def _execute_final(
        self,
        original_path: str,
        action_chain: list[EditAction],
        segments: list[SegmentMetadata],
        output_path: str,
    ) -> str:
        """
        在原始视频上执行完整的方案链条（高质量单次编码）。

        策略：
        - 不可合并的动作（stabilize等）按顺序独立执行
        - 所有可合并滤镜收集起来，最后一次 FFmpeg 调用完成
        """
        logger.info(f"在原始视频上执行最终方案（{len(action_chain)} 个动作）...")

        non_mergeable_actions = []
        merged_filters = []

        for action in action_chain:
            filter_str = BasicTools.get_filter_string(action.tool_name, action.parameters)
            if filter_str is not None:
                merged_filters.append(filter_str)
                logger.info(f"  [合并] {action.tool_name}: {filter_str}")
            elif action.tool_type == "compound" and action.tool_name == "auto_color_harmonize":
                non_mergeable_actions.append(("compound", action))
                logger.info(f"  [独立] {action.tool_name}")
            elif action.tool_name in self.basic_tools.get_tool_registry():
                non_mergeable_actions.append(("basic", action))
                logger.info(f"  [独立] {action.tool_name}")

        current_video = original_path

        # 1. 先执行不可合并的动作
        for action_type, action in non_mergeable_actions:
            if action_type == "compound":
                current_video = self.compound_tools.auto_color_harmonize(
                    current_video, segments
                )
            else:
                registry = self.basic_tools.get_tool_registry()
                tool_func = registry[action.tool_name]["func"]
                current_video = tool_func(current_video, **action.parameters)

        # 2. 所有可合并滤镜一次性应用
        if merged_filters:
            vf = ",".join(merged_filters)
            logger.info(f"  合并滤镜链: {vf}")
            self.basic_tools._run_ffmpeg(
                ["-i", current_video, "-vf", vf] + ENCODE_ARGS,
                output_path,
            )
            return output_path
        elif current_video != original_path:
            shutil.copy2(current_video, output_path)
            return output_path
        else:
            return current_video

    # ══════════════════════════════════════════════════════
    # 主循环
    # ══════════════════════════════════════════════════════

    def run(self, video_path: str, output_dir: str = "output") -> str:
        os.makedirs(output_dir, exist_ok=True)
        original_path = video_path
        analysis_dir = os.path.join(output_dir, "analysis")

        # 初始化决策日志
        run_log = RunLogger(output_dir)
        run_log.log_input(video_path, {
            "model": self.config.get("openai", {}).get("model", "?"),
            "max_iterations": self.max_iterations,
            "mcts_simulations": self.config.get("planner", {}).get("mcts_simulations", "?"),
            "mcts_depth": self.config.get("planner", {}).get("mcts_depth", "?"),
        })

        logger.info(f"VlogAgent 开始处理: {video_path}")

        # ── Step 1: 视频分析 ────────────────────────────
        run_log.step_start("analysis")
        logger.info("[Step 1] 视频分析...")
        segments = self.analyzer.analyze(video_path, output_dir=analysis_dir)
        run_log.log_analysis(segments)
        logger.info(f"  检测到 {len(segments)} 个片段 ({run_log.step_end('analysis'):.1f}s)")

        # ── 评估原始视频基线分数（不调 VLM，只算技术指标）────
        run_log.step_start("baseline")
        logger.info("[Baseline] 评估原始视频基线...")
        baseline_result = self.evaluator.evaluate_baseline(original_path, segments)
        prev_score = baseline_result.overall_score
        logger.info(
            f"  原始视频基线分数: {prev_score:.3f} "
            f"(视觉={baseline_result.visual_quality:.2f} "
            f"一致={baseline_result.inter_segment_consistency:.2f} "
            f"美学=0.50[基线]) "
            f"({run_log.step_end('baseline'):.1f}s)"
        )

        no_improve_count = 0
        current_video = video_path       # 搜索阶段的当前视频（累积编辑结果）
        action_chain: list[EditAction] = []  # 累积的完整动作链
        total_iterations = 0

        for iteration in range(1, self.max_iterations + 1):
            total_iterations = iteration
            logger.info(f"\n===== 闭环迭代 {iteration}/{self.max_iterations} =====")

            # ── Step 2: Perceiver（观察当前累积编辑后的视频）
            run_log.step_start("perceiver")
            logger.info("[Step 2] Perceiver...")
            observation, candidates = self.perceiver.perceive(current_video, segments)
            if not candidates:
                logger.warning("Perceiver 未返回候选指令，终止")
                break
            run_log.log_perceiver(iteration, observation, candidates)
            logger.info(f"  {len(candidates)} 条候选 ({run_log.step_end('perceiver'):.1f}s)")

            # ── Step 3: Planner (MCTS) ───────────────────
            run_log.step_start("planner")
            logger.info("[Step 3] Planner MCTS...")
            top_k_plans = self.planner.search(candidates, segments)
            if not top_k_plans:
                logger.warning("Planner 未找到方案，终止")
                break
            run_log.log_planner(iteration, top_k_plans)
            logger.info(f"  {len(top_k_plans)} 个方案 ({run_log.step_end('planner'):.1f}s)")

            # ── Step 4: Executor（在当前视频上执行 Top-K 方案）
            run_log.step_start("executor")
            logger.info("[Step 4] Executor...")
            results = []
            for i, plan in enumerate(top_k_plans):
                logger.info(f"  执行方案 {i+1}/{len(top_k_plans)}")
                try:
                    edited, actions_log = self._execute_plan_for_search(
                        current_video, plan, segments
                    )
                    results.append((edited, plan, actions_log))
                    run_log.log_executor(iteration, i + 1, actions_log)
                except Exception as e:
                    logger.error(f"  方案 {i+1} 失败: {e}")
            logger.info(f"  执行完成 ({run_log.step_end('executor'):.1f}s)")

            if not results:
                logger.warning("所有方案执行失败，终止")
                break

            # ── Step 5: Evaluator ────────────────────────
            run_log.step_start("evaluator")
            logger.info("[Step 5] Evaluator...")
            eval_results = []
            best_result = None
            best_score = -1.0
            best_plan_this_round = None

            for idx, (edited_path, plan, _) in enumerate(results):
                eval_result = self.evaluator.evaluate(
                    original_path, edited_path, segments
                )
                eval_results.append({"plan_idx": idx + 1, "eval": eval_result})
                if eval_result.overall_score > best_score:
                    best_score = eval_result.overall_score
                    best_result = (edited_path, eval_result)
                    best_plan_this_round = plan

            run_log.log_evaluation(iteration, eval_results)
            logger.info(f"  评分完成 ({run_log.step_end('evaluator'):.1f}s)")

            if best_result is None:
                break

            edited_path, eval_result = best_result
            logger.info(f"  本轮最佳: {best_score:.3f} (上轮: {prev_score:.3f})")

            # ── Step 6: 闭环决策 ─────────────────────────
            if best_score > prev_score + self.min_improvement:
                run_log.log_decision(iteration, best_score, prev_score, accepted=True)
                logger.info(f"  采纳 (+{best_score - prev_score:.3f})")

                # 更新当前视频为本轮编辑结果（下轮在此基础上继续探索）
                current_video = edited_path
                prev_score = best_score
                no_improve_count = 0

                # 将本轮方案的动作追加到链条中
                for action in best_plan_this_round.actions:
                    resolved = self.tool_selector.resolve_action(action)
                    action_chain.append(resolved)

                logger.info(f"  动作链累积: {len(action_chain)} 个动作")
            else:
                no_improve_count += 1
                run_log.log_decision(iteration, best_score, prev_score, accepted=False)
                logger.info(f"  回退 (连续 {no_improve_count} 轮无提升)")
                if no_improve_count >= self.max_no_improve:
                    logger.info("  达到上限，终止闭环")
                    break

        # ══════════════════════════════════════════════════
        # 最终执行：在原始视频上一次性应用完整动作链（高质量单次编码）
        # ══════════════════════════════════════════════════
        final_output = os.path.join(output_dir, f"final_{os.path.basename(video_path)}")

        if action_chain:
            run_log.step_start("final")
            logger.info(f"\n===== 在原始视频上执行完整动作链 ({len(action_chain)} 个动作) =====")
            self._execute_final(original_path, action_chain, segments, final_output)
            logger.info(f"  最终编码完成 ({run_log.step_end('final'):.1f}s)")
            run_log.log_finish(final_output, total_iterations, prev_score)
            logger.info(f"完成！输出: {final_output}")
            return final_output
        else:
            run_log.log_finish(video_path, total_iterations, prev_score)
            logger.info("完成！未进行有效编辑，返回原视频")
            return video_path
