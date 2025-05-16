from ast import arguments
import asyncio
import os
import shutil # For shutil.which
import sys
import logging
import json
from pathlib import Path
from typing import Any, Dict, Optional, Type

from langchain_core.tools import BaseTool, ToolException
from pydantic import Field, BaseModel, model_validator, PrivateAttr # Using Pydantic v2 imports

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import (
    Implementation as MCPImplementation,
    ClientCapabilities, # Though not directly passed to ClientSession constructor
    InitializeResult,
    CallToolResult,
    LATEST_PROTOCOL_VERSION,
    TextContent, # For parsing results
    ErrorData # For parsing error results
)

# --- Custom Error ---
class McpPythonToolError(ToolException):
    """Custom exception for MCP Python tool errors."""
    pass

# --- Input Schema ---
class PythonInput(BaseModel):
    code: str = Field(description="The Python code snippet to execute.")
    timeout: Optional[float] = Field(default=60.0, description="Optional timeout in seconds for the Python code execution.")

# --- Main Tool Class (Async Version) ---
class MCPPythonExecutorTool(BaseTool, BaseModel): # Inherit from BaseModel for Pydantic features
    name: str = "MCPPythonExecutor"
    description: str = (
        "Execute any arbitrary Python script or code snippet. "
        "This tool runs your Python code in a secure subprocess and returns the output (stdout), errors (stderr), and exit code. "
        "You can use it for calculations, text processing, web requests, file operations, or any valid Python logic.\n\n"
        "**Important Note:** Avoid passing very large scripts or data blobs directly in the `code` field. "
        "Instead, use the Filesystem tool to save large data to a file and have your Python code read from that file. "
        "This approach helps prevent potential performance issues and errors due to large payload sizes.\n\n"
        "**Input Format:**\n"
        "- Provide a JSON object with a single required key: `code`, containing your Python code as a string.\n"
        "- You may also optionally provide a `timeout` (float, seconds) to limit execution time.\n\n"
        "**Examples:**\n"
        "```json\n"
        "{\n  \"code\": \"print('Hello, world!')\"\n}\n"
        "```\n"
        "```json\n"
        "{\n  \"code\": \"import math; print(math.sqrt(16))\",\n  \"timeout\": 10.0\n}\n"
        "```\n"
        "```json\n"
        "{\n  \"code\": \"with open('data.txt', 'r') as f: print(f.read())\"\n}\n"
        "```\n"
        "In the last example, assume you have used the Filesystem tool to save your large data to a file named `data.txt`. "
        "Your Python code can then read from this file instead of having the large data passed directly in the `code` field.\n\n"
        "**Package Installation:**\n"
        "To install new Python packages, always use the `install_python_package` tool. This guarantees installation in the correct Python environment used by the executor. Do NOT use the bash tool or raw pip commands for installing packages.\n"
        "**Example:**\n"
        "```json\n"
        "{\n  \"package\": \"requests\"\n}\n"
        "```\n"
        "Returns a string containing the script's output, error messages (if any), and the exit code."
    )
    args_schema: Type[BaseModel] = PythonInput
    return_direct: bool = False
    handle_tool_error: bool = True

    # Configuration for the MCP server process
    python_executable_for_server: Path = Field(
        default_factory=lambda: Path(shutil.which("python3") or shutil.which("python") or sys.executable or "python")
    )
    server_script_path: Path = Field(
        default_factory=lambda: (Path(__file__).resolve().parent.parent.parent / "mcp_servers" / "python_server.py").resolve()
    )
    session_init_timeout: float = 30.0
    tool_call_timeout: float = 60.0 # Default timeout for the Python code execution via MCP

    # Internal state
    _session: Optional[ClientSession] = PrivateAttr(default=None)
    _lifecycle_task: Optional[asyncio.Task] = PrivateAttr(default=None)
    _session_ready_event: Optional[asyncio.Event] = PrivateAttr(default=None)
    _shutdown_event: Optional[asyncio.Event] = PrivateAttr(default=None)
    _init_lock: Optional[asyncio.Lock] = PrivateAttr(default=None)
    _is_closed: bool = PrivateAttr(default=False)
    _logger: Any = PrivateAttr(default=None)

    # MCP Client Info
    _client_info: MCPImplementation = PrivateAttr(default=MCPImplementation(name="MCPPythonExecutorToolClient", version="0.2.0"))

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def _resolve_paths_and_init_logger_validator(self) -> 'MCPPythonExecutorTool':
        if self._logger is None:
            self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
            # Basic handler setup if no handlers are configured for this logger
            if not self._logger.hasHandlers():
                handler = logging.StreamHandler(sys.stderr) # Or sys.stdout
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self._logger.addHandler(handler)
                self._logger.propagate = False # Avoid duplicate messages if root logger also has handlers

        # Resolve python_executable_for_server
        if not self.python_executable_for_server or str(self.python_executable_for_server) == "python":
            resolved_executable = shutil.which("python3") or shutil.which("python") or sys.executable
            if resolved_executable:
                self.python_executable_for_server = Path(resolved_executable)
            else:
                self._logger.critical("Could not find a Python interpreter for the server. Tool will likely fail.")
                self.python_executable_for_server = Path("python") # Fallback
        self._logger.info(f"Using python_executable_for_server: {self.python_executable_for_server}")

        # Resolve server_script_path
        if not self.server_script_path.is_absolute():
            self.server_script_path = (Path(__file__).parent.parent.parent / self.server_script_path).resolve()
        self._logger.info(f"Using server_script_path: {self.server_script_path}")

        if not self.server_script_path.exists():
            self._logger.warning(f"Python server script path does not exist: {self.server_script_path}")
        return self

    async def _initialize_async_primitives_if_needed(self):
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()
        if self._session_ready_event is None:
            self._session_ready_event = asyncio.Event()
        if self._shutdown_event is None:
            self._shutdown_event = asyncio.Event()

    def _get_server_params(self) -> StdioServerParameters:
        # Ensure paths are resolved and exist (validator should handle this, but double-check)
        if not self.python_executable_for_server.exists():
             self._logger.warning(f"Python executable for server not found at: {self.python_executable_for_server}. Subprocess will likely fail.")
        if not self.server_script_path.exists():
            # This should ideally prevent server startup, but _ensure_session_ready handles task failure.
            self._logger.error(f"Python server script not found at: {self.server_script_path}. Cannot create server parameters.")
            raise McpPythonToolError(f"Server script not found: {self.server_script_path}")

        cmd_str = str(self.python_executable_for_server.resolve())
        args_list = [str(self.server_script_path.resolve())]
        cwd_str = str(self.server_script_path.parent.resolve())

        self._logger.debug(f"Server params: command='{cmd_str}', args={args_list}, cwd='{cwd_str}'")
        return StdioServerParameters(command=cmd_str, args=args_list, cwd=cwd_str)

    async def _manage_session_lifecycle(self):
        self._logger.info("Starting MCP Python session lifecycle...")
        try:
            server_params = self._get_server_params()
            self._logger.debug(f"Server params: cmd='{server_params.command}', args={server_params.args}, cwd='{server_params.cwd}'")

            async with stdio_client(server_params, errlog=sys.stderr) as (rs, ws):
                self._logger.info("Stdio streams obtained. Creating ClientSession.")
                async with ClientSession(rs, ws, client_info=self._client_info) as session:
                    self._logger.info("ClientSession created. Initializing...")
                    init_result: InitializeResult = await asyncio.wait_for(
                        session.initialize(), timeout=self.session_init_timeout
                    )
                    self._logger.info(f"MCP session initialized. Server capabilities: {init_result.capabilities}")
                    self._session = session
                    self._session_ready_event.set()
                    
                    self._logger.info("Python session ready. Waiting for shutdown signal...")
                    await self._shutdown_event.wait()
                    self._logger.info("Shutdown signal received for Python session.")
        except asyncio.TimeoutError:
            self._logger.error(f"Timeout during Python session initialization ({self.session_init_timeout}s).")
        except asyncio.CancelledError:
            self._logger.info("Python session lifecycle task cancelled.")
        except Exception as e:
            self._logger.error(f"Error in Python session lifecycle: {e}", exc_info=True)
        finally:
            self._logger.info("Python session lifecycle task finished.")
            self._session = None # Clear session
            if self._session_ready_event and not self._session_ready_event.is_set():
                self._session_ready_event.set() # Ensure waiters are unblocked, they will find no session

    async def _ensure_session_ready(self):
        if self._is_closed:
            raise McpPythonToolError(f"{self.name} is closed.")

        await self._initialize_async_primitives_if_needed()

        if self._session and self._session_ready_event.is_set():
            return

        async with self._init_lock:
            if self._session and self._session_ready_event.is_set():
                return
            if self._is_closed:
                 raise McpPythonToolError(f"{self.name} was closed during session readiness check.")

            if self._lifecycle_task is None or self._lifecycle_task.done():
                self._logger.info("No active Python lifecycle task or task is done. Starting new one.")
                self._session_ready_event.clear()
                self._shutdown_event.clear()
                self._lifecycle_task = asyncio.create_task(self._manage_session_lifecycle())
            else:
                self._logger.debug("Python lifecycle task already running.")

            self._logger.debug("Waiting for Python session to become ready...")
            try:
                await asyncio.wait_for(self._session_ready_event.wait(), timeout=self.session_init_timeout + 5.0)
            except asyncio.TimeoutError:
                self._logger.error("Timeout waiting for MCP Python session to become ready.")
                if self._lifecycle_task and not self._lifecycle_task.done():
                    self._lifecycle_task.cancel()
                    try: await self._lifecycle_task
                    except asyncio.CancelledError: self._logger.info("Python lifecycle task cancelled after timeout.")
                    except Exception as e_cancel: self._logger.error(f"Error cancelling Python lifecycle task: {e_cancel}")
                raise McpPythonToolError("Timeout establishing MCP Python session.")

            if not self._session or not self._session_ready_event.is_set():
                self._logger.error("Python session ready event set, but session is not available/initialized.")
                raise McpPythonToolError("Failed to establish a valid MCP Python session.")
            
            self._logger.info("MCP Python session is ready.")

    async def _arun(self, code: str, timeout: Optional[float] = None, run_manager: Optional[Any] = None) -> str:
        """Asynchronously executes the Python code snippet."""
        if self._is_closed:
            self._logger.warning(f"Attempt to run code on closed tool: {self.name}")
            return "Error: Tool is closed."
        
        effective_timeout = timeout if timeout is not None else self.tool_call_timeout
        self._logger.info(f"Executing Python code (timeout {effective_timeout}s): {code[:200]}{'...' if len(code) > 200 else ''}")
        
        try:
            await self._ensure_session_ready()

            if not self._session:
                self._logger.error("MCP Python session not available for _arun.")
                # This case should ideally be caught by _ensure_session_ready raising an error.
                # If it reaches here, it means _ensure_session_ready might have an issue.
                return "Error: MCP Python session not available after readiness check."

            # This dictionary directly matches the fields of PythonCommandInput on the server
            python_command_input_params = {
                "code": code, 
                "timeout": effective_timeout
            }
            
            # FastMCP server's @mcp.tool decorated function `execute_python_code_tool(input: PythonCommandInput)`
            # expects the arguments to be nested under a key matching the parameter name 'input'.
            arguments_for_server = {
                "input": python_command_input_params 
            }
            
            self._logger.debug(f"Calling MCP 'execute_python_code' with 'arguments' payload: {arguments_for_server}")
            
            response: CallToolResult = await asyncio.wait_for(
                self._session.call_tool(name="execute_python_code", arguments=arguments_for_server),
                timeout=effective_timeout + 10.0 # Add buffer for MCP communication + server processing
            )
            
            self._logger.debug(f"_arun Python: Received response. isError: {response.isError}, Content type: {type(response.content)}")

            stdout_str = ""
            stderr_str = ""
            exit_code = -99 # Default for parsing failure

            if response.isError:
                self._logger.error("_arun Python: Tool call failed on server (response.isError=True).")
                error_detail = "Server indicated an error."
                if response.content and isinstance(response.content, list) and len(response.content) > 0:
                    first_content_item = response.content[0]
                    if isinstance(first_content_item, ErrorData) and first_content_item.message:
                        error_detail = first_content_item.message
                    elif hasattr(first_content_item, 'text') and isinstance(first_content_item.text, str):
                        # Sometimes errors might come as simple TextContent
                        error_detail = first_content_item.text
                    else:
                        self._logger.warning(f"_arun Python: Error response content item not ErrorData or TextContent: {type(first_content_item)}")
                        try: error_detail = str(first_content_item) # Fallback
                        except: pass 
                else:
                    self._logger.warning("_arun Python: Error response, but content is empty or not a list.")
                stderr_str = error_detail # Main error detail goes to stderr
                exit_code = -98 # Indicate server-side tool error response
            
            elif response.content and isinstance(response.content, list) and len(response.content) > 0:
                first_content = response.content[0]
                self._logger.debug(f"_arun Python: Successful response, first_content type: {type(first_content)}")

                tool_output_data = None
                # The python_server.py's execute_python_code_tool returns PythonCommandOutput.
                # FastMCP typically wraps this in a ModelContent, or if simple, could be TextContent (JSON string).
                if isinstance(first_content, TextContent) and first_content.text:
                    self._logger.debug(f"_arun Python: first_content is TextContent. Text: {first_content.text[:200]}")
                    try:
                        tool_output_data = json.loads(first_content.text)
                    except json.JSONDecodeError as e:
                        self._logger.error(f"_arun Python: Failed to parse JSON from TextContent: {e}")
                        stderr_str = f"Failed to parse JSON response from server: {first_content.text}"
                elif hasattr(first_content, 'data') and first_content.data is not None: # For ModelContent
                    self._logger.debug("_arun Python: first_content has 'data' attribute (likely ModelContent).")
                    tool_output_data = first_content.data 
                else: # Should not happen if server returns PythonCommandOutput correctly
                    self._logger.error("_arun Python: Successful response, but content[0] structure is not recognized (not TextContent with JSON, nor ModelContent with data).")
                    stderr_str = "Server returned success, but tool output data was not in expected structure."

                if tool_output_data and isinstance(tool_output_data, dict):
                    # Expecting a dict matching PythonCommandOutput fields
                    stdout_str = str(tool_output_data.get('stdout', ''))
                    # Preserve earlier stderr_str (e.g., JSON parse error) if already set
                    stderr_str = str(tool_output_data.get('stderr', '') or stderr_str) 
                    exit_code = int(tool_output_data.get('exit_code', -99)) # Default if 'exit_code' missing
                elif not stderr_str: # If no parsing error yet, but tool_output_data is bad
                    self._logger.warning(f"_arun Python: tool_output_data was not a dict. Type: {type(tool_output_data)}. Data: {str(tool_output_data)[:200]}")
                    stderr_str = "Failed to parse structured output (stdout, stderr, exit_code) from server response."
            else:
                self._logger.error("_arun Python: Successful response, but response.content is empty, not a list, or None.")
                stderr_str = "Server returned success, but response content was empty or malformed."

            self._logger.info(f"_arun Python: Command processed. Parsed Exit code: {exit_code}")

            # Construct final output string
            formatted_output = f"Exit Code: {exit_code}"
            if stdout_str:
                formatted_output += f"\n---\nSTDOUT:\n{stdout_str.strip()}"
            if stderr_str: # Only add STDERR section if there's content for it
                formatted_output += f"\n---\nSTDERR:\n{stderr_str.strip()}"
            return formatted_output.strip()

        except asyncio.TimeoutError:
            self._logger.error(f"Timeout executing Python code (MCP call): {code[:100]}...")
            # This timeout is for the self._session.call_tool()
            return f"Error: Timeout during MCP communication for Python code execution (tool_call_timeout: {effective_timeout + 10.0}s)."
        except McpPythonToolError as e: # Errors from _ensure_session_ready
            self._logger.error(f"_arun Python: McpPythonToolError: {e}")
            return f"Error preparing Python execution environment: {e}"
        except Exception as e:
            self._logger.error(f"Unexpected error executing Python code: {e}", exc_info=True)
            return f"Unexpected error during Python execution: {e}"
            
    async def close(self):
        if self._is_closed and (self._lifecycle_task is None or self._lifecycle_task.done()):
            self._logger.debug(f"{self.name} already closed or Python lifecycle task not running.")
            return
        
        self._logger.info(f"Closing {self.name} (Python)...")
        self._is_closed = True
        await self._initialize_async_primitives_if_needed()

        if self._shutdown_event: self._shutdown_event.set()

        if self._lifecycle_task and not self._lifecycle_task.done():
            self._logger.debug("Waiting for Python lifecycle task to complete...")
            try:
                await asyncio.wait_for(self._lifecycle_task, timeout=10.0)
            except asyncio.TimeoutError:
                self._logger.warning("Timeout waiting for Python lifecycle task. Cancelling.")
                self._lifecycle_task.cancel()
                try: await self._lifecycle_task
                except asyncio.CancelledError: self._logger.info("Python lifecycle task cancelled.")
                except Exception as e_final: self._logger.error(f"Error awaiting cancelled Python task: {e_final}")
            except Exception as e: self._logger.error(f"Error during Python task shutdown: {e}")
        
        if self._session_ready_event: self._session_ready_event.clear()
        if self._shutdown_event: self._shutdown_event.clear()
        self._session = None
        self._logger.info(f"{self.name} (Python) closed.")

    def _run(self, code: str, timeout: Optional[float] = None, **kwargs: Any) -> str:
        raise NotImplementedError(
            "MCPPythonExecutorTool is async-native. Use _arun or implement synchronous bridging if needed."
        )

# Example usage:
async def main_python_tool_test():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(name)s - [%(threadName)s] - %(message)s'
    )
    tool_logger = logging.getLogger("__main__.MCPPythonExecutorTool_Test")
    tool_logger.info("Initializing MCPPythonExecutorTool for example usage...")
    
    python_tool = MCPPythonExecutorTool() # Uses default logger from class if not passed

    test_scripts = [
        ("print('Hello from Python MCP tool!')", False, 60.0),
        ("import sys; sys.stderr.write('This is an error message\\n'); sys.exit(1)", True, 60.0),
        ("print('Looping...'); import time; [time.sleep(0.1) for _ in range(3)]; print('Done looping.')", False, 60.0),
        ("print('Testing timeout'); import time; time.sleep(5)", True, 2.0), # Expected to timeout
        ("for i in range(3): print(f'Count: {i}')", False, 60.0),
        ("1/0", True, 60.0) # Division by zero error
    ]

    try:
        for i, (script, expect_stderr_or_nonzero_exit, script_timeout) in enumerate(test_scripts):
            tool_logger.info(f"\n--- Test {i+1}: Running Python script (timeout {script_timeout}s) ---")
            tool_logger.info(f"Script:\n{script}")
            
            # For _arun, the timeout kwarg is directly passed to the method.
            result_str = await python_tool._arun(code=script, timeout=script_timeout)
            tool_logger.info(f"Test {i+1} Result:\n{result_str}")
            
            # Basic check for expected outcome
            has_stderr = "STDERR:" in result_str
            has_nonzero_exit = "Exit Code: 0" not in result_str
            
            if expect_stderr_or_nonzero_exit:
                if not (has_stderr or has_nonzero_exit):
                    tool_logger.error(f"Test {i+1} was expected to produce an error/stderr/non-zero exit, but output suggests success.")
            else: # Expected success
                if has_stderr or has_nonzero_exit:
                    tool_logger.error(f"Test {i+1} was expected to succeed, but output suggests failure.")

    except Exception as e:
        tool_logger.error(f"Error during example usage: {e}", exc_info=True)
    finally:
        tool_logger.info("Closing MCPPythonExecutorTool from example usage's finally block.")
        await python_tool.close()
        tool_logger.info("Example usage finished.")

if __name__ == '__main__':
    asyncio.run(main_python_tool_test())