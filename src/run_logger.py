"""运行决策日志：记录关键步骤到 Markdown 文件，方便回溯"""

import os
import time
from datetime import datetime


class RunLogger:
    def __init__(self, output_dir: str = "output"):
        log_dir = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"run_log_{timestamp}.md")
        self._start_time = time.time()
        self._step_times: dict[str, float] = {}

        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(f"# VlogAgent 运行日志\n\n")
            f.write(f"**开始时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    def _elapsed(self) -> str:
        return f"{time.time() - self._start_time:.1f}s"

    def _write(self, text: str):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text)

    def log_input(self, video_path: str, config_summary: dict):
        self._write(
            f"**输入视频**: `{video_path}`\n\n"
            f"**配置**: model=`{config_summary.get('model', '?')}`, "
            f"outer={config_summary.get('max_outer', '?')}, "
            f"inner={config_summary.get('max_inner', '?')}, "
            f"mcts_sims={config_summary.get('mcts_sims', '?')}\n\n"
            f"**架构**: Director-Editor-Critic 三Agent双层循环\n\n"
            f"---\n\n"
        )

    def step_start(self, step_name: str):
        self._step_times[step_name] = time.time()

    def step_end(self, step_name: str) -> float:
        duration = time.time() - self._step_times.get(step_name, time.time())
        return duration

    # ── 各阶段日志 ────────────────────────────────────

    def log_analysis(self, segments: list):
        self._write(f"## Step 1: 视频分析 [{self._elapsed()}]\n\n")
        self._write(f"检测到 **{len(segments)}** 个片段：\n\n")
        self._write("| 片段 | 时间范围 | 亮度 | 色温 | 清晰度 | 稳定性 | 噪声 | 语音 |\n")
        self._write("|------|----------|------|------|--------|--------|------|------|\n")
        for s in segments:
            speech = "有" if s.has_speech else "无"
            self._write(
                f"| seg-{s.seg_id} "
                f"| {s.time_range[0]:.1f}s-{s.time_range[1]:.1f}s "
                f"| {s.mean_brightness:.0f} "
                f"| {s.color_temp_est:.2f} "
                f"| {s.sharpness_score:.0f} "
                f"| {s.stability_score:.2f} "
                f"| {s.noise_level:.1f} "
                f"| {speech} |\n"
            )
        self._write("\n")

    def log_perceiver(self, iteration: int, observation: str, candidates: list):
        self._write(f"## 迭代 {iteration}: Perceiver [{self._elapsed()}]\n\n")
        self._write(f"### 阶段1: VLM 画面观察\n\n")
        self._write(f"{observation}\n\n")
        self._write(f"### 阶段2: 候选编辑指令 ({len(candidates)} 条)\n\n")
        for i, a in enumerate(candidates):
            self._write(f"{i+1}. `[{a.tool_name}]` {a.action_description} → {a.target_segment}\n")
        self._write("\n")

    def log_planner(self, iteration: int, plans: list):
        self._write(f"## 迭代 {iteration}: Planner (MCTS) [{self._elapsed()}]\n\n")
        self._write(f"搜索出 **{len(plans)}** 个方案：\n\n")
        for i, plan in enumerate(plans):
            tools = " → ".join(f"`{a.tool_name}`" for a in plan.actions)
            self._write(f"- **方案 {i+1}** (预估分={plan.estimated_score:.3f}): {tools}\n")
        self._write("\n")

    def log_executor(self, iteration: int, plan_idx: int, actions_log: list[dict]):
        self._write(f"### 执行方案 {plan_idx}\n\n")
        for log in actions_log:
            status = "OK" if log["success"] else "FAIL"
            self._write(
                f"- [{status}] `{log['tool_name']}` "
                f"{log['description']} | params={log['parameters']}"
            )
            if not log["success"]:
                self._write(f" | error: {log['error']}")
            self._write("\n")
        self._write("\n")

    def log_evaluation(self, iteration: int, results: list[dict]):
        self._write(f"## 迭代 {iteration}: Evaluator [{self._elapsed()}]\n\n")
        self._write("| 方案 | 视觉 | 保真 | 一致 | 音频 | 美学 | **总分** |\n")
        self._write("|------|------|------|------|------|------|----------|\n")
        for r in results:
            e = r["eval"]
            self._write(
                f"| {r['plan_idx']} "
                f"| {e.visual_quality:.2f} "
                f"| {e.content_fidelity:.2f} "
                f"| {e.inter_segment_consistency:.2f} "
                f"| {e.audio_integrity:.2f} "
                f"| {e.aesthetic:.2f} "
                f"| **{e.overall_score:.3f}** |\n"
            )
        self._write("\n")

    def log_decision(self, iteration: int, best_score: float, prev_score: float, accepted: bool):
        delta = best_score - prev_score
        if accepted:
            self._write(f"> **决策**: 采纳 (分数 {prev_score:.3f} → {best_score:.3f}, +{delta:.3f})\n\n")
        else:
            self._write(f"> **决策**: 回退 (分数 {best_score:.3f} ≤ {prev_score:.3f}, 无提升)\n\n")
        self._write("---\n\n")

    def log_director(self, style_brief, is_revision: bool = False):
        label = "Director 修订策略" if is_revision else "Director 风格策略"
        self._write(f"## {label} [{self._elapsed()}]\n\n")
        self._write(
            f"- **风格**: {style_brief.overall_style}\n"
            f"- **色彩方向**: {style_brief.color_direction}\n"
            f"- **优先级**: {style_brief.priority}\n"
            f"- **目标情绪**: {style_brief.target_mood}\n"
            f"- **约束**: {', '.join(style_brief.constraints) if style_brief.constraints else '无'}\n"
        )
        if style_brief.stages:
            self._write("\n**分阶段规划**:\n\n")
            for sd in style_brief.stages:
                targets = f" → {', '.join(sd.target_segments)}" if sd.target_segments else ""
                self._write(f"| {sd.stage} | {sd.scope} | {sd.direction}{targets} |\n")
            self._write("\n")
        else:
            self._write("- **分阶段规划**: 未输出（VLM 未返回 stages）\n\n")

    def log_critic(self, outer_iter: int, inner_iter: int, feedback):
        self._write(f"## 外层{outer_iter}-内层{inner_iter}: Critic 评审 [{self._elapsed()}]\n\n")
        self._write(f"**综合分数**: {feedback.overall_score:.3f}\n\n")

        if feedback.segment_feedback:
            self._write("| 片段 | 判定 | 原因 |\n|------|------|------|\n")
            for seg_id, sc in feedback.segment_feedback.items():
                verdict = sc.verdict if hasattr(sc, "verdict") else sc.get("verdict", "?")
                reason = sc.reason if hasattr(sc, "reason") else sc.get("reason", "")
                self._write(f"| {seg_id} | {verdict} | {reason[:80]} |\n")
            self._write("\n")

        if feedback.global_issues:
            self._write(f"**全局问题**: {', '.join(feedback.global_issues)}\n\n")
        if feedback.suggestions:
            self._write(f"**改进建议**: {', '.join(feedback.suggestions)}\n\n")

    def log_route_decision(self, outer_iter: int, inner_iter: int, route: str, reason: str):
        emoji = {"accept": "✓", "refine": "↻", "redirect": "↺"}.get(route, "?")
        self._write(f"> **路由 {emoji}**: {route} — {reason}\n\n---\n\n")

    def log_finish(self, output_path: str, total_iterations: int, final_score: float):
        elapsed = time.time() - self._start_time
        self._write(
            f"## 完成\n\n"
            f"- **总耗时**: {elapsed:.1f}s ({elapsed/60:.1f}min)\n"
            f"- **迭代次数**: {total_iterations}\n"
            f"- **最终分数**: {final_score:.3f}\n"
            f"- **输出文件**: `{output_path}`\n"
        )
        print(f"\n决策日志已保存: {self.log_path}")
