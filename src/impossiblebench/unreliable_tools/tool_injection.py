"""Unreliable tool wrappers with simple error injection."""

from typing import Callable

from inspect_ai.tool import ToolError, tool
from inspect_ai.tool._tools._execute import code_viewer
from inspect_ai.util import sandbox as sandbox_env

from .tool_unreliability import ToolReliability


def _pre_error(tool_reliability: ToolReliability | None) -> ToolError | None:
    if not tool_reliability:
        return None
    if tool_reliability.error in ("permission_denied", "tool_disabled"):
        return tool_reliability.tool_error()
    return None


def _post_error(tool_reliability: ToolReliability | None) -> ToolError | None:
    if tool_reliability and tool_reliability.error == "timeout":
        return tool_reliability.tool_error()
    return None


def _effective_timeout(tool_reliability: ToolReliability | None, default_timeout: int) -> int:
    if not tool_reliability:
        return default_timeout
    return tool_reliability.apply_timeout(default_timeout)


@tool(viewer=code_viewer("bash", "cmd"), name="bash")
def unreliable_bash(
    tool_reliability: ToolReliability | None = None,
    timeout: int | None = None,
    user: str | None = None,
    sandbox: str | None = None,
    attempt_num: int = 1,
) -> Callable:
    async def execute(cmd: str):
        """
        Use this function to execute bash commands.

        Args:
          cmd (str): The bash command to execute.

        Returns:
          The output of the command.
        """
        err = _pre_error(tool_reliability)
        if err:
            return err

        effective_timeout = _effective_timeout(tool_reliability, timeout or 60)
        result = await sandbox_env(sandbox).exec(
            cmd=["bash", "--login", "-c", cmd],
            timeout=effective_timeout,
            user=user,
        )

        err = _post_error(tool_reliability)
        if err:
            return err

        output = ""
        if result.stderr:
            output = f"{result.stderr}\n"
        return f"{output}{result.stdout}"

    return execute


@tool(viewer=code_viewer("python", "code"), name="python")
def unreliable_python(
    tool_reliability: ToolReliability | None = None,
    timeout: int | None = None,
    user: str | None = None,
    sandbox: str | None = None,
    attempt_num: int = 1,
) -> Callable:

    async def execute(code: str):
        """
        Use the python function to execute Python code.

        The Python tool executes single-run Python scripts. Important notes:
        1. Each execution is independent - no state is preserved between runs
        2. You must explicitly use print() statements to see any output
        3. Simply writing expressions (like in notebooks) will not display results
        4. The script cannot accept interactive input during execution
        5. Return statements alone won't produce visible output
        6. All variables and imports are cleared between executions
        7. Standard output (via print()) is the only way to see results

        Args:
          code (str): The python code to execute.

        Returns:
          The output of the Python code.
        """
        err = _pre_error(tool_reliability)
        if err:
            return err

        effective_timeout = _effective_timeout(tool_reliability, timeout or 60)
        result = await sandbox_env(sandbox).exec(
            cmd=["bash", "--login", "-c", "python3 -"],
            input=code,
            timeout=effective_timeout,
            user=user,
        )

        err = _post_error(tool_reliability)
        if err:
            return err

        output = ""
        if result.stderr:
            output = f"{result.stderr}\n"
        return f"{output}{result.stdout}"

    return execute


@tool(name="text_editor")
def unreliable_text_editor(
    tool_reliability: ToolReliability | None = None,
    timeout: int | None = None,
    user: str | None = None,
    attempt_num: int = 1,
) -> Callable:
    from inspect_ai.tool import text_editor

    editor = text_editor(timeout=_effective_timeout(tool_reliability, timeout or 180), user=user)

    async def execute(*args, **kwargs):
        err = _pre_error(tool_reliability)
        if err:
            return err

        result = await editor(*args, **kwargs)

        err = _post_error(tool_reliability)
        if err:
            return err

        return result

    return execute
