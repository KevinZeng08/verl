"""GSM8K calculator tool for Agentic RL training."""

import logging

from verl.tools.function_tool import function_tool

logger = logging.getLogger(__name__)


@function_tool("calc_gsm8k_reward")
def calc_gsm8k_reward(expression: str) -> str:
    """Evaluate a mathematical expression and return the numerical result.

    Use this tool to verify your calculations. Pass a Python arithmetic expression.

    Args:
        expression: A Python arithmetic expression, e.g. '(3 + 5) * 2' or '100 / 4 - 3'.
    """
    # logger.warning(f"calc_gsm8k_reward called with expression={expression!r}")
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        # logger.warning(f"calc_gsm8k_reward result={result}")
        return str(result)
    except Exception as e:
        # logger.warning(f"calc_gsm8k_reward error: {e}")
        return f"Error: {e}"
