import inspect
import random
from typing import Literal

from inspect_ai.tool import (
    Tool,
    ToolCall,
    ToolCallContent,
    ToolCallView,
    ToolCallViewer,
    ToolDef,
    ToolError,
    ToolResult,
    tool,
)
from inspect_ai.tool._json_rpc_helpers import exec_scalar_request
from inspect_ai.tool._sandbox_tools_utils._runtime_helpers import (
    SandboxJSONRPCTransport,
    SandboxToolsServerErrorMapper,
)
from inspect_ai.tool._sandbox_tools_utils.sandbox import (
    SANDBOX_TOOLS_CLI,
    sandbox_with_injected_tools,
)
from inspect_ai.tool._tools._text_editor import TextEditorParams, TextEditorResult
from inspect_ai.util import sandbox as sandbox_env


# custom viewer for bash and python code blocks
def code_viewer(language: str, code_param: str) -> ToolCallViewer:
    def viewer(tool_call: ToolCall) -> ToolCallView:
        code = tool_call.arguments.get(code_param, None)
        code = str(code or tool_call.function).strip()
        call = ToolCallContent(
            title=language,
            format="markdown",
            content=f"```{language}\n" + code + "\n```\n",
        )
        return ToolCallView(call=call)

    return viewer


@tool(viewer=code_viewer("bash", "cmd"))
def bash(
    timeout: int | None = None,
    user: str | None = None,
    sandbox: str | None = None,
    random_seed: int = 42,
    failure_rate: float = 0.0,
) -> Tool:
    """Bash shell command execution tool.

    Execute bash shell commands using a sandbox environment (e.g. "docker").

    Args:
      timeout: Timeout (in seconds) for command.
      user: User to execute commands as.
      sandbox: Optional sandbox environment name.

    Returns:
      String with command output (stdout) or command error (stderr).
    """

    async def execute(cmd: str) -> str | ToolError:
        """
        Use this function to execute bash commands.

        Args:
          cmd (str): The bash command to execute.

        Returns:
          The output of the command.
        """
        rng = random.Random(random_seed)
        if failure_rate == 1.0:
            return ToolError("Tool permanently unavailable.")

        if 0.0 < failure_rate < 1.0 and rng.random() < failure_rate:
            return ToolError("Tool temporarily unavailable, please try again later.")

        # execute the command
        result = await sandbox_env(sandbox).exec(
            cmd=["bash", "--login", "-c", cmd], timeout=timeout, user=user
        )
        # return output (including stderr if any)
        output = ""
        if result.stderr:
            output = f"{result.stderr}\n"
        return f"{output}{result.stdout}"

    return execute


@tool(viewer=code_viewer("python", "code"))
def python(
    timeout: int | None = None,
    user: str | None = None,
    sandbox: str | None = None,
    random_seed: int = 42,
    failure_rate: float = 0.0,
) -> Tool:
    """Python code execution tool.

    Execute Python code using a sandbox environment (e.g. "docker").

    Args:
      timeout: Timeout (in seconds) for command.
      user: User to execute commands as.
      sandbox: Optional sandbox environment name.

    Returns:
      String with command output (stdout) or command error (stderr).
    """

    async def execute(code: str) -> str | ToolError:
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
        rng = random.Random(random_seed)
        if failure_rate == 1.0:
            return ToolError("Tool permanently unavailable.")

        if 0.0 < failure_rate < 1.0 and rng.random() < failure_rate:
            return ToolError("Tool temporarily unavailable, please try again later.")

        result = await sandbox_env(sandbox).exec(
            cmd=["bash", "--login", "-c", "python3 -"],
            input=code,
            timeout=timeout,
            user=user,
        )
        # return output (including stderr if any)
        output = ""
        if result.stderr:
            output = f"{result.stderr}\n"
        return f"{output}{result.stdout}"

    return execute


@tool
def think(
    description: str | None = None,
    thought_description: str | None = None,
) -> Tool:
    """Think tool for extra thinking.

    Tool that provides models with the ability to include an additional thinking step as part of getting to its final answer.

    Note that the `think()` tool is not a substitute for reasoning and extended thinking, but rather an an alternate way of letting models express thinking that is better suited to some tool use scenarios. Please see the documentation on using the [think tool](https://inspect.aisi.org.uk/tools-standard.html#sec-think) before using it in your evaluations.

    Args:
        description: Override the default description of the think tool.
        thought_description: Override the default description of the thought parameter.
    """

    async def execute(thought: str) -> str:
        """Use the tool to think about something.

        The will not obtain new information or change the environment, but just append the thought to the log. Use it when complex reasoning or some cache memory is needed.

        Args:
            thought: A thought to think about.
        """
        return ""

    return ToolDef(
        execute,
        name="think",
        description=description,
        parameters=(dict(thought=thought_description) if thought_description else None),
        viewer=think_tool_viewer(),
    ).as_tool()


def think_tool_viewer() -> ToolCallViewer:
    def viewer(tool_call: ToolCall) -> ToolCallView:
        call = ToolCallContent(
            format="markdown", content=tool_call.arguments.get("thought", "")
        )
        return ToolCallView(call=call)

    return viewer


@tool()
def text_editor(
    timeout: int | None = None,
    user: str | None = None,
    random_seed: int = 42,
    failure_rate: float = 0.0,
) -> Tool:
    """Custom editing tool for viewing, creating and editing files.

    Perform text editor operations using a sandbox environment (e.g. "docker").

    IMPORTANT: This tool does not currently support Subtask isolation. This means
    that a change made to a file by on Subtask will be visible to another Subtask.

    Args:
      timeout: Timeout (in seconds) for command. Defaults to 180 if not provided.
      user: User to execute commands as.

    Returns:
      String with command output (stdout) or command error (stderr).
    """
    timeout = timeout or 180

    async def execute(
        command: Literal["view", "create", "str_replace", "insert", "undo_edit"],
        path: str,
        file_text: str | None = None,
        insert_line: int | None = None,
        new_str: str | None = None,
        old_str: str | None = None,
        view_range: list[int] | None = None,
    ) -> ToolResult | ToolError:
        """
        Use this function to execute text editing commands.

        Args:
          command: The command to execute.
          path: Path to file or directory, e.g. `/repo/file.py` or `../repo`.
          file_text: Required parameter of `create` command, with the content of the file to be created.
          insert_line: Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.
          new_str: Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.
          old_str: Required parameter of `str_replace` command containing the string in `path` to replace.
          view_range: Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.

        Returns:
          The output of the command.
        """

        rng = random.Random(random_seed)
        if failure_rate == 1.0:
            return ToolError("Tool permanently unavailable.")

        if 0.0 < failure_rate < 1.0 and rng.random() < failure_rate:
            return ToolError("Tool temporarily unavailable, please try again later.")

        sandbox = await sandbox_with_injected_tools()

        # Create a dictionary of the parameters
        params = {
            k: v
            for k, v in locals().items()
            if k in inspect.signature(execute).parameters
        }

        return await exec_scalar_request(
            method="text_editor",
            params=params,
            result_type=TextEditorResult,
            transport=SandboxJSONRPCTransport(sandbox, SANDBOX_TOOLS_CLI),
            server_error_mapper=SandboxToolsServerErrorMapper(),
            timeout=timeout,
            user=user,
        )

    return execute
