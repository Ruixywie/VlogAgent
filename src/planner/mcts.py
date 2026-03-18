"""MCTS Planner：蒙特卡洛树搜索最优编辑方案"""

import math
import json
import logging
import random
from copy import deepcopy

from openai import OpenAI

from src.models import EditAction, EditPlan, SegmentMetadata

logger = logging.getLogger(__name__)


class MCTSNode:
    """MCTS 搜索树节点"""

    def __init__(self, action: EditAction | None = None, parent: "MCTSNode | None" = None):
        self.action = action
        self.parent = parent
        self.children: list[MCTSNode] = []
        self.visits: int = 0
        self.total_reward: float = 0.0
        self.untried_actions: list[EditAction] = []

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.visits if self.visits > 0 else 0.0

    def ucb_score(self, c: float = 1.4) -> float:
        if self.visits == 0:
            return float("inf")
        exploit = self.avg_reward
        explore = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore

    def best_child(self, c: float = 1.4) -> "MCTSNode":
        return max(self.children, key=lambda n: n.ucb_score(c))

    def get_action_sequence(self) -> list[EditAction]:
        """从根到当前节点的动作序列"""
        actions = []
        node = self
        while node.parent is not None:
            if node.action is not None:
                actions.append(node.action)
            node = node.parent
        actions.reverse()
        return actions


class MCTSPlanner:
    def __init__(self, config: dict):
        self.depth = config.get("mcts_depth", 3)
        self.simulations = config.get("mcts_simulations", 30)
        self.ucb_c = config.get("ucb_c", 1.4)
        self.top_k = config.get("top_k", 3)
        self.client = OpenAI(
            api_key=config.get("api_key", None),
            base_url=config.get("base_url", None),
        )
        self.model = config.get("model", "gpt-4o")

    def search(
        self,
        candidates: list[EditAction],
        segments: list[SegmentMetadata],
    ) -> list[EditPlan]:
        """MCTS 搜索，返回 Top-K 最优编辑方案"""
        if not candidates:
            return []

        logger.info(f"MCTS: 从 {len(candidates)} 条候选中搜索 (depth={self.depth}, sims={self.simulations})")

        root = MCTSNode()
        root.untried_actions = list(candidates)

        for sim_i in range(self.simulations):
            # 1. Selection
            node = self._select(root)

            # 2. Expansion
            if not node.is_fully_expanded and len(node.get_action_sequence()) < self.depth:
                node = self._expand(node, candidates)

            # 3. Simulation (LLM 启发式 Rollout)
            reward = self._simulate(node, candidates, segments)

            # 4. Backpropagation
            self._backpropagate(node, reward)

        # 提取 Top-K 方案
        return self._extract_top_k(root)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """UCB 选择"""
        while node.children and node.is_fully_expanded:
            node = node.best_child(self.ucb_c)
        return node

    def _expand(self, node: MCTSNode, candidates: list[EditAction]) -> MCTSNode:
        """扩展一个未尝试的动作"""
        if not node.untried_actions:
            return node

        action = node.untried_actions.pop(random.randrange(len(node.untried_actions)))
        child = MCTSNode(action=action, parent=node)

        # 已选动作不再进入子节点的候选
        used = {a.action_description for a in node.get_action_sequence()}
        used.add(action.action_description)
        child.untried_actions = [a for a in candidates if a.action_description not in used]

        node.children.append(child)
        return child

    def _simulate(
        self, node: MCTSNode, candidates: list[EditAction], segments: list[SegmentMetadata]
    ) -> float:
        """LLM 启发式 Rollout：让 LLM 预估动作序列的效果分数"""
        action_seq = node.get_action_sequence()
        if not action_seq:
            return 0.0

        # 构建 prompt 让 LLM 评估
        actions_desc = "\n".join(
            f"  {i+1}. [{a.tool_name}] {a.action_description} → {a.target_segment}"
            for i, a in enumerate(action_seq)
        )
        seg_summary = "\n".join(
            f"  seg-{s.seg_id}: 亮度={s.mean_brightness:.0f}, 色温={s.color_temp_est:.2f}, "
            f"清晰度={s.sharpness_score:.0f}, 稳定性={s.stability_score:.2f}"
            for s in segments
        )

        prompt = f"""根据以下视频片段信息和编辑动作序列，预估编辑后视频的整体质量提升。

视频片段:
{seg_summary}

编辑动作序列:
{actions_desc}

请评估：
1. 这些动作是否合理？是否有冲突或冗余？
2. 预估编辑后相对原始视频的质量提升

返回 JSON: {{"score": 0.0-1.0, "reason": "..."}}
score=0.5 表示无变化，>0.5 表示提升，<0.5 表示变差。"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是视频质量评估专家。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            score = float(data.get("score", 0.5))
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"MCTS simulation LLM 调用失败: {e}")
            return 0.5

    def _backpropagate(self, node: MCTSNode, reward: float):
        """回传奖励"""
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def _extract_top_k(self, root: MCTSNode) -> list[EditPlan]:
        """从搜索树中提取 Top-K 最优方案"""
        all_plans = []
        self._collect_plans(root, all_plans)

        # 按平均奖励排序
        all_plans.sort(key=lambda p: p.estimated_score, reverse=True)
        top_k = all_plans[:self.top_k]

        for i, plan in enumerate(top_k):
            logger.info(
                f"MCTS Top-{i+1}: score={plan.estimated_score:.3f}, "
                f"actions={[a.tool_name for a in plan.actions]}"
            )

        return top_k

    def _collect_plans(self, node: MCTSNode, plans: list[EditPlan]):
        """递归收集所有叶节点方案"""
        if not node.children:
            actions = node.get_action_sequence()
            if actions:
                plans.append(EditPlan(
                    actions=actions,
                    estimated_score=node.avg_reward,
                ))
        for child in node.children:
            self._collect_plans(child, plans)
