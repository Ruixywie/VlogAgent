"""工具选择器：LLM 驱动的语义指令 → 工具调用翻译"""

import json
import logging

from openai import OpenAI

from src.models import EditAction

logger = logging.getLogger(__name__)

TOOL_SELECTION_PROMPT = """你是视频编辑工具路由器。将语义编辑指令翻译为具体的工具调用参数。

可用工具及参数范围：
- color_adjust: brightness[-0.3,0.3], contrast[0.5,2.0], saturation[0.5,2.0], gamma[0.5,2.0]
- white_balance: temperature[2000,10000] (色温K值)
- denoise: strength[1,10]
- sharpen: amount[0.5,3.0]
- stabilize: smoothing[5,30]
- speed_adjust: factor[0.25,4.0] (1.0=原速)
- auto_color_harmonize: (无参数，compound工具)

输入：一条编辑指令的语义描述
输出：JSON 格式 {"tool_name": "...", "parameters": {...}}

注意：参数必须在范围内，偏保守，避免过度处理。"""


class ToolSelector:
    def __init__(self, config: dict):
        self.client = OpenAI(
            api_key=config.get("api_key", None),
            base_url=config.get("base_url", None),
        )
        self.model = config.get("model", "gpt-4o")

    def resolve_action(self, action: EditAction) -> EditAction:
        """如果 action 缺少具体工具名/参数，用 LLM 填充"""
        if action.tool_name and action.parameters:
            return action  # 已经有具体工具和参数

        prompt = (
            f"编辑指令: {action.action_description}\n"
            f"目标片段: {action.target_segment}\n"
            f"工具类型: {action.tool_type}\n"
        )
        if action.tool_name:
            prompt += f"建议工具: {action.tool_name}\n"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": TOOL_SELECTION_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        try:
            data = json.loads(raw)
            action.tool_name = data.get("tool_name", action.tool_name)
            action.parameters = data.get("parameters", action.parameters)
        except json.JSONDecodeError:
            logger.warning(f"ToolSelector 解析失败: {raw[:200]}")

        return action
