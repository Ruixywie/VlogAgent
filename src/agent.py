"""VlogAgent 主循环：Director-Editor-Critic 三Agent双层循环

架构：
- Director（导演）：制定整体风格策略 + 判断是否继续
- Editor（编辑）：在 Director 约束 + Critic 反馈下生成方案（Perceiver + MCTS）
- Critic（评审）：结构化评审 + 路由决策（accept/refine/redirect）

双层循环：
- 外层（Director 驱动）：每轮产出一个被 accept 的方案，累积到 action_chain
- 内层（Editor-Critic）：把一个方案打磨到 Critic 认可

最终执行：在原始视频上一次性应用完整 action_chain（滤镜合并，单次编码）
"""

import logging
import os
import shutil

import yaml

from src.models import (
    SegmentMetadata, EditPlan, EditAction, EvaluationResult,
    StyleBrief, CriticFeedback, FallbackLLM, normalize_seg_id,
    STAGE_ORDER, get_stage,
)
from src.perceiver.video_analyzer import VideoAnalyzer
from src.perceiver.perceiver import Perceiver
from src.director.director import Director
from src.critic.critic import Critic
from src.planner.stage_selector import StageSelector
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
        self.llm = FallbackLLM(llm_config)

        # 视频分析（不变）
        perceiver_cfg = {**self.config.get("perceiver", {})}
        perceiver_cfg["ffmpeg_path"] = self.config.get("executor", {}).get("ffmpeg_path", "ffmpeg")
        self.analyzer = VideoAnalyzer(perceiver_cfg)

        # 三个 Agent + 视觉预筛选
        self.director = Director(self.llm)
        self.editor = Perceiver(self.llm)       # Editor 的候选生成器
        self.stage_selector = StageSelector(self.config.get("planner", {}))
        self.critic = Critic({
            **self.config.get("evaluator", {}),
            **self.config.get("critic", {}),
        }, self.llm)

        # 执行器（不变）
        executor_cfg = self.config.get("executor", {})
        self.basic_tools = BasicTools(executor_cfg)
        self.tool_selector = ToolSelector(self.llm)
        self.compound_tools = CompoundTools(self.basic_tools)

        # Evaluator 保留用于 baseline
        evaluator_cfg = self.config.get("evaluator", {})
        self.evaluator = Evaluator(evaluator_cfg, self.llm)

        # 循环参数
        agent_cfg = self.config.get("agent", {})
        self.max_outer = agent_cfg.get("max_outer_iterations", 3)
        self.max_inner = agent_cfg.get("max_inner_iterations", 3)
        self.ffmpeg = executor_cfg.get("ffmpeg_path", "ffmpeg")

    # ══════════════════════════════════════════════════════
    # 搜索阶段执行（不变）
    # ══════════════════════════════════════════════════════

    def _execute_plan_for_search(
        self,
        video_path: str,
        plan: EditPlan,
        segments: list[SegmentMetadata],
    ) -> tuple[str, list[dict]]:
        """执行方案用于搜索评估"""
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
    # 最终执行（不变）
    # ══════════════════════════════════════════════════════

    def _get_time_range(
        self, target_segment: str, segments: list[SegmentMetadata]
    ) -> tuple[float, float] | None:
        """根据 target_segment 获取时间范围，global 返回 None"""
        target = normalize_seg_id(target_segment)
        if target == "global":
            return None
        # 构建映射
        seg_map = {f"seg-{s.seg_id}": s.time_range for s in segments}
        return seg_map.get(target, None)

    @staticmethod
    def _sort_by_stage(action_chain: list[EditAction]) -> list[EditAction]:
        """按专业后期阶段顺序排列动作链"""
        stage_index = {s: i for i, s in enumerate(STAGE_ORDER)}

        def sort_key(action):
            stage = get_stage(action)
            if stage and stage in stage_index:
                return stage_index[stage]
            return len(STAGE_ORDER)  # 未知阶段排到最后

        return sorted(action_chain, key=sort_key)

    def _execute_final(
        self,
        original_path: str,
        action_chain: list[EditAction],
        segments: list[SegmentMetadata],
        output_path: str,
    ) -> str:
        """在原始视频上执行完整动作链（阶段排序 + 段级感知 + 单次编码）

        按专业后期阶段顺序排列：stabilize → denoise → color_correct → color_grade → sharpen
        可合并滤镜：根据 target_segment 添加 enable='between(t,start,end)'
        不可合并工具（stabilize 等）：仅 global 模式允许执行
        """
        # 按专业阶段顺序排列
        action_chain = self._sort_by_stage(action_chain)
        logger.info(
            f"在原始视频上执行最终方案（{len(action_chain)} 个动作，"
            f"阶段顺序: {[f'{a.tool_name}({get_stage(a)})' for a in action_chain]}）"
        )

        non_mergeable_actions = []
        merged_filters = []
        registry = self.basic_tools.get_tool_registry()

        for action in action_chain:
            target = normalize_seg_id(action.target_segment)
            time_range = self._get_time_range(target, segments)

            # 可合并滤镜：支持段级 enable
            filter_str = BasicTools.get_filter_string(
                action.tool_name, action.parameters, time_range=time_range
            )
            if filter_str is not None:
                label = f"{action.tool_name} → {target}"
                if time_range:
                    label += f" ({time_range[0]:.1f}s-{time_range[1]:.1f}s)"
                merged_filters.append(filter_str)
                logger.info(f"  [合并] {label}: {filter_str}")
            elif action.tool_type == "compound" and action.tool_name == "auto_color_harmonize":
                non_mergeable_actions.append(("compound", action))
                logger.info(f"  [独立] {action.tool_name}")
            elif action.tool_name in registry:
                tool_info = registry[action.tool_name]
                # global_only 工具：只有 global 才执行，段级的跳过
                if tool_info.get("global_only", False) and target != "global":
                    logger.info(
                        f"  [跳过] {action.tool_name} → {target} "
                        f"(仅支持全局应用，段级操作已跳过)"
                    )
                    continue
                non_mergeable_actions.append(("basic", action))
                logger.info(f"  [独立] {action.tool_name}")

        current_video = original_path

        # 先执行不可合并的动作
        for action_type, action in non_mergeable_actions:
            if action_type == "compound":
                current_video = self.compound_tools.auto_color_harmonize(
                    current_video, segments
                )
            else:
                tool_func = registry[action.tool_name]["func"]
                current_video = tool_func(current_video, **action.parameters)

        # 所有可合并滤镜一次性应用（带段级 enable）
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
    # 主循环：Director-Editor-Critic 双层循环
    # ══════════════════════════════════════════════════════

    def run(self, video_path: str, output_dir: str = "output") -> str:
        os.makedirs(output_dir, exist_ok=True)
        original_path = video_path
        analysis_dir = os.path.join(output_dir, "analysis")

        run_log = RunLogger(output_dir)
        run_log.log_input(video_path, {
            "model": self.config.get("openai", {}).get("models", ["?"])[0],
            "max_outer": self.max_outer,
            "max_inner": self.max_inner,
            "mcts_sims": self.config.get("planner", {}).get("mcts_simulations", "?"),
        })

        logger.info(f"VlogAgent 开始处理: {video_path}")

        # ── Step 1: 视频分析 ────────────────────────────
        run_log.step_start("analysis")
        logger.info("[Step 1] 视频分析...")
        segments = self.analyzer.analyze(video_path, output_dir=analysis_dir)
        run_log.log_analysis(segments)
        logger.info(f"  检测到 {len(segments)} 个片段 ({run_log.step_end('analysis'):.1f}s)")

        # ── Baseline 评估 ───────────────────────────────
        run_log.step_start("baseline")
        logger.info("[Baseline] 评估原始视频...")
        baseline_result = self.evaluator.evaluate_baseline(original_path, segments)
        prev_score = baseline_result.overall_score
        logger.info(f"  基线分数: {prev_score:.3f} ({run_log.step_end('baseline'):.1f}s)")

        # ── Step 2: Director 制定风格策略 ────────────────
        run_log.step_start("director")
        logger.info("[Step 2] Director 制定风格策略...")
        style_brief = self.director.strategize(segments)
        run_log.log_director(style_brief)
        logger.info(f"  风格策略完成 ({run_log.step_end('director'):.1f}s)")

        action_chain: list[EditAction] = []
        current_video = video_path
        total_outer = 0
        no_outer_improve = 0

        # 跨轮次保留 Critic 反馈（不在外层循环中重置）
        last_critic_feedback = None

        # ═══════════ 外层循环（Director 驱动）═══════════
        for outer_iter in range(1, self.max_outer + 1):
            total_outer = outer_iter
            logger.info(f"\n{'='*20} 外层迭代 {outer_iter}/{self.max_outer} {'='*20}")

            # 内层循环的 critic_feedback 从上一轮的反馈继承（不丢失历史信息）
            critic_feedback = last_critic_feedback
            refine_count = 0
            inner_accepted = False

            # ═══════ 逐阶段 Editor-Critic 循环 ═══════
            max_stage_refine = self.config.get("planner", {}).get("max_stage_refine", 2)

            if style_brief.stages:
                # 一次性观察画面（所有阶段共用）
                run_log.step_start("observe")
                logger.info("  [Editor] 观察画面...")
                observation = self.editor.observe(segments)
                logger.info(f"  观察完成 ({run_log.step_end('observe'):.1f}s)")

                stage_accepted_actions = []

                for stage_decision in style_brief.stages:
                    if stage_decision.scope == "skip":
                        logger.info(f"  阶段 {stage_decision.stage}: skip")
                        continue

                    logger.info(f"  ── 阶段 {stage_decision.stage} ({stage_decision.scope}) ──")
                    stage_refine_count = 0

                    for stage_attempt in range(max_stage_refine + 1):
                        # ① Editor 为该阶段生成 2-3 条候选
                        run_log.step_start("editor_stage")
                        candidates = self.editor.suggest_for_stage(
                            observation, segments, stage_decision,
                            style_brief, critic_feedback,
                        )
                        if not candidates:
                            logger.warning(f"  阶段 {stage_decision.stage}: 无候选，跳过")
                            break

                        # ② PIL 模拟 + CLIP+MLP 选最优
                        best_action, best_score = self.stage_selector.select_best(
                            candidates, segments
                        )
                        if best_action is None:
                            logger.warning(f"  阶段 {stage_decision.stage}: 预筛选失败，跳过")
                            break

                        logger.info(
                            f"  [{best_action.tool_name}] 预筛选分={best_score:.3f} "
                            f"({run_log.step_end('editor_stage'):.1f}s)"
                        )

                        # ③ FFmpeg 执行最优候选
                        run_log.step_start("executor_stage")
                        plan = EditPlan(actions=[best_action])
                        try:
                            edited_path, actions_log = self._execute_plan_for_search(
                                current_video, plan, segments
                            )
                            run_log.log_executor(outer_iter, stage_attempt + 1, actions_log)
                        except Exception as e:
                            logger.error(f"  阶段 {stage_decision.stage} 执行失败: {e}")
                            break
                        logger.info(f"  执行完成 ({run_log.step_end('executor_stage'):.1f}s)")

                        # ④ Critic 评审该阶段
                        run_log.step_start("critic_stage")
                        critic_feedback = self.critic.evaluate_stage(
                            original_path, edited_path, segments,
                            stage_decision.stage, best_action,
                            prev_score, stage_refine_count,
                        )
                        run_log.log_critic(outer_iter, stage_attempt + 1, critic_feedback)
                        run_log.log_route_decision(
                            outer_iter, stage_attempt + 1,
                            critic_feedback.route, critic_feedback.route_reason,
                        )
                        logger.info(
                            f"  Critic: {critic_feedback.route} "
                            f"({critic_feedback.route_reason}) "
                            f"({run_log.step_end('critic_stage'):.1f}s)"
                        )

                        if critic_feedback.route == "accept":
                            current_video = edited_path
                            prev_score = critic_feedback.overall_score
                            resolved = self.tool_selector.resolve_action(best_action)
                            stage_accepted_actions.append(resolved)
                            logger.info(f"  阶段 {stage_decision.stage}: Accept!")
                            break
                        else:
                            stage_refine_count += 1
                            logger.info(f"  阶段 {stage_decision.stage}: Refine (第 {stage_refine_count} 次)")

                # 所有阶段处理完毕
                if stage_accepted_actions:
                    action_chain.extend(stage_accepted_actions)
                    inner_accepted = True
                    no_outer_improve = 0
                    last_critic_feedback = critic_feedback
                    logger.info(f"  逐阶段完成! 动作链: {len(action_chain)} 个动作")
                else:
                    logger.info("  所有阶段均跳过或失败")

            else:
                # 向后兼容：stages 为空时简单处理
                logger.warning("  Director 未输出 stages，跳过内层循环")

            # ── 外层决策：Director 判断是否继续 ──
            if not inner_accepted:
                no_outer_improve += 1

            if not self.director.should_continue(
                critic_feedback, prev_score, outer_iter, self.max_outer
            ):
                logger.info(f"Director: 终止外层循环")
                break

        # ═══════════ Step 7: 最终执行 ═══════════════════
        final_output = os.path.join(output_dir, f"final_{os.path.basename(video_path)}")

        if action_chain:
            run_log.step_start("final")
            logger.info(f"\n{'='*20} 最终执行 ({len(action_chain)} 个动作) {'='*20}")
            self._execute_final(original_path, action_chain, segments, final_output)
            logger.info(f"  最终编码完成 ({run_log.step_end('final'):.1f}s)")
            run_log.log_finish(final_output, total_outer, prev_score)
            logger.info(f"完成！输出: {final_output}")
            return final_output
        else:
            run_log.log_finish(video_path, total_outer, prev_score)
            logger.info("完成！未进行有效编辑，返回原视频")
            return video_path
