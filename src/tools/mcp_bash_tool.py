import asyncio
import os
import shutil
import sys
import logging
import json
from pathlib import Path
from typing import Any, Dict, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import Field, BaseModel, model_validator, PrivateAttr # Changed for Pydantic v2

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import Implementation as MCPImplementation, ClientCapabilities, InitializeResult, CallToolResult, LATEST_PROTOCOL_VERSION, TextContent

# --- Custom Error ---
class McpToolError(Exception):
    """Custom exception for MCP tool errors."""
    pass

# --- Input Schema ---
class BashInput(BaseModel):
    command_to_execute: str = Field(description="The bash command string to execute.")

# --- Main Tool Class (Async Version) ---
class MCPBashExecutorTool(BaseTool, BaseModel):
    """
    A tool for executing bash commands or entire multi-line bash scripts via an MCP (Model Context Protocol) server.
    
    This tool connects to a dedicated MCP server that exposes a bash execution capability.
    It handles the lifecycle of the server process (starting and stopping it) and manages
    the communication session for sending commands/scripts and receiving their output (stdout, stderr, exit code).

    Input to the tool's _arun method is a string containing the bash command or script to execute.
    Output is a string containing the stdout of the command if successful, or a formatted error message
    including stderr and exit code if the command fails or an error occurs during execution.
    """
    # In MCPBashExecutorTool class definition:
    name: str = "MCPBashExecutor"
    description: str = (
        "Executes non-interactive bash commands or entire multi-line bash scripts in a secure environment "
        "and returns their stdout, stderr, and exit code. "
        "The 'action_input' MUST be a JSON object containing a 'command_to_execute' field " # Changed 'command' to 'command_to_execute'
        "with the bash command or script.\n"
        "Example: {\"command_to_execute\": \"echo hello && ls -la\"}" # Corrected example to use 'command_to_execute'
    )
    args_schema: Type[BaseModel] = BashInput
    return_direct: bool = False

    # Configuration for the MCP server process
    python_executable: Path = Field(
        default_factory=lambda: Path(shutil.which("python3") or shutil.which("python") or sys.executable or "python")
    )
    server_script_path: Path = Field(
        default_factory=lambda: (Path(__file__).parent.parent.parent / "mcp_servers" / "bash_server.py").resolve()
    )
    server_restart_delay: float = 5.0  # Seconds to wait before restarting the server process
    session_init_timeout: float = 30.0 # Seconds to wait for MCP session Initialize
    tool_call_timeout: float = 60.0    # Seconds to wait for a tool call to complete

    # Internal state, not part of the model's schema for validation purposes if using model_construct
    _session: Optional[ClientSession] = PrivateAttr(default=None)
    _stdio_client: Optional[Any] = PrivateAttr(default=None) # Opaque type for StdioClient's async context manager result
    _lifecycle_task: Optional[asyncio.Task] = PrivateAttr(default=None)
    _session_ready_event: Optional[asyncio.Event] = PrivateAttr(default=None)
    _shutdown_event: Optional[asyncio.Event] = PrivateAttr(default=None)
    _session_failed_event: Optional[asyncio.Event] = PrivateAttr(default=None)
    _init_lock: Optional[asyncio.Lock] = PrivateAttr(default=None)
    _is_closed: bool = PrivateAttr(default=False)
    _logger: Any = PrivateAttr(default=None) # Will be initialized in post_init

    # MCP Client Info
    _client_info: MCPImplementation = PrivateAttr(default=MCPImplementation(name="MCPBashExecutorToolClient", version="0.2.0"))

    class Config:
        arbitrary_types_allowed = True
        # For Pydantic v2, if we want to exclude private attrs from model_dump, etc.
        # We can use PrivateAttr or underscore prefix and model_fields_set logic.
        # Langchain BaseTool might have its own Pydantic config considerations.

    @model_validator(mode='after')
    def _resolve_paths_and_init_logger(self) -> 'MCPBashExecutorTool':
        """Ensure logger is set up and paths are resolved after model initialization."""
        if self._logger is None:
            self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
            if not self._logger.hasHandlers():
                handler = logging.StreamHandler(sys.stderr)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self._logger.addHandler(handler)
                self._logger.propagate = False
        
        # Resolve python_executable (default_factory should have run)
        # If it's still the default sentinel or needs re-validation:
        if not self.python_executable or str(self.python_executable) == "python": # Check if factory failed to find better
            resolved_executable = shutil.which("python3") or shutil.which("python") or sys.executable
            if resolved_executable:
                self.python_executable = Path(resolved_executable)
            else:
                self._logger.critical("Could not find a Python interpreter. MCPBashExecutorTool will likely fail.")
                self.python_executable = Path("python") # Fallback, will likely fail if python isn't in PATH
        self._logger.info(f"Using python_executable: {self.python_executable}")

        # Resolve server_script_path (default_factory should have run)
        # Ensure it's absolute and exists.
        if not self.server_script_path.is_absolute():
             # This case should ideally be handled by a robust default_factory or validator for the field itself.
             # Forcing resolution relative to this file's parent's parent for safety if it wasn't made absolute.
            self.server_script_path = (Path(__file__).parent.parent.parent / self.server_script_path).resolve()
        self._logger.info(f"Using server_script_path: {self.server_script_path}")

        if not self.server_script_path.exists():
            self._logger.warning(f"Server script path does not exist: {self.server_script_path}")
        
        return self

    def _get_server_params(self) -> StdioServerParameters:
        # python_executable and server_script_path should be resolved Path objects by now.
        # The @model_validator should have ensured they are properly set.
        if not self.python_executable or not isinstance(self.python_executable, Path):
            self._logger.error("python_executable not properly resolved. Validator might have issues.")
            # Fallback logic, though validator should prevent this state.
            py_exec_fallback = shutil.which("python3") or shutil.which("python") or sys.executable
            resolved_executable = Path(py_exec_fallback) if py_exec_fallback else Path("python")
        else:
            resolved_executable = self.python_executable.resolve()

        if not self.server_script_path or not isinstance(self.server_script_path, Path) or not self.server_script_path.is_absolute():
            self._logger.error("server_script_path not properly resolved. Validator might have issues.")
            # Fallback logic
            resolved_script_path = (Path(__file__).parent.parent.parent / "mcp_servers" / "bash_server.py").resolve()
        else:
            resolved_script_path = self.server_script_path.resolve()

        cmd_str = str(resolved_executable)
        args_list = [str(resolved_script_path)]
        cwd_str = str(resolved_script_path.parent)

        self._logger.debug(f"Server params for Popen: command='{cmd_str}', args={args_list}, cwd='{cwd_str}'")

        return StdioServerParameters(
            command=cmd_str, # Executable path as a string
            args=args_list,  # Script path and any other args as a list
            cwd=cwd_str,
        )

    async def _initialize_async_primitives_if_needed(self):
        """Initializes asyncio primitives if they don't exist.
           Must be called from within an async context where a loop is running.
        """
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()
        if self._session_ready_event is None:
            self._session_ready_event = asyncio.Event()
        if self._shutdown_event is None:
            self._shutdown_event = asyncio.Event()
        if self._session_failed_event is None:
            self._session_failed_event = asyncio.Event()

    async def _manage_session_lifecycle(self):
        """Manages the MCP server process and client session."""
        self._logger.info("Starting MCP session lifecycle...")
        try:
            server_params = self._get_server_params()
            self._logger.debug(f"Server params: cmd='{server_params.command}', args={server_params.args}, cwd='{server_params.cwd}'")

            async with stdio_client(server_params, errlog=sys.stderr) as (rs, ws):
                self._logger.info("Stdio streams obtained. Creating ClientSession.")
                async with ClientSession(rs, ws, client_info=self._client_info) as session:
                    self._logger.info("ClientSession created. Initializing...")
                    init_result: InitializeResult = await session.initialize()
                    self._logger.info(f"MCP session initialized. Server capabilities: {init_result.capabilities}")
                    self._session = session
                    if self._session_ready_event: # Should exist
                        self._session_ready_event.set()
                    
                    self._logger.info("Session ready. Waiting for shutdown signal...")
                    if self._shutdown_event: # Should exist
                        await self._shutdown_event.wait()
                    self._logger.info("Shutdown signal received.")
        
        except asyncio.CancelledError:
            self._logger.info("Session lifecycle task cancelled.")
            # stdio_client and ClientSession context managers will handle cleanup
        except Exception as e:
            self._logger.error(f"Error in session lifecycle: {e}", exc_info=True)
            # Ensure waiters are unblocked even on error
            if self._session_ready_event and not self._session_ready_event.is_set():
                self._session_ready_event.set()
            if self._session_failed_event and not self._session_failed_event.is_set():
                self._session_failed_event.set()
        finally:
            self._logger.info("Session lifecycle task finished.")
            self._session = None # Clear session
            # If shutdown_event wasn't set, and task exited for other reasons,
            # ensure ready_event is set to unblock potential waiters (who will find no session).
            if self._session_ready_event and not self._session_ready_event.is_set():
                self._session_ready_event.set()

    async def _ensure_session_ready(self) -> None:
        """Ensures the MCP session is initialized and ready for use."""
        if self._is_closed:
            raise McpToolError(f"{self.name} is closed.")

        await self._initialize_async_primitives_if_needed() # Initialize locks/events if not already

        if self._session and self._session_ready_event.is_set():
            return

        async with self._init_lock: # Prevent multiple concurrent initializations
            # Double-check after acquiring lock
            if self._session and self._session_ready_event.is_set():
                return
            if self._is_closed: # Check again after lock
                 raise McpToolError(f"{self.name} was closed during session readiness check.")

            if self._lifecycle_task is None or self._lifecycle_task.done():
                self._logger.info("No active lifecycle task or task is done. Starting new one.")
                self._session_ready_event.clear() # Clear before starting new task
                self._shutdown_event.clear()    # Clear before starting new task
                self._session_failed_event.clear() # Clear before starting new task
                self._lifecycle_task = asyncio.create_task(self._manage_session_lifecycle())
            else:
                self._logger.debug("Lifecycle task already running.")

            self._logger.debug("Waiting for session to become ready...")
            try:
                await asyncio.wait_for(self._session_ready_event.wait(), timeout=self.tool_call_timeout + 10)
            except asyncio.TimeoutError:
                self._logger.error("Timeout waiting for MCP session to become ready.")
                # Attempt graceful shutdown of the task if it's stuck
                if self._lifecycle_task and not self._lifecycle_task.done():
                    self._lifecycle_task.cancel()
                    try:
                        await self._lifecycle_task # Await cancellation
                    except asyncio.CancelledError:
                        self._logger.info("Lifecycle task successfully cancelled after timeout.")
                    except Exception as e_cancel:
                        self._logger.error(f"Error during lifecycle task cancellation: {e_cancel}")
                raise McpToolError("Timeout establishing MCP session.")

            if not self._session or not self._session_ready_event.is_set():
                # This means session_ready_event was set, but session is not valid
                # _manage_session_lifecycle likely hit an error after setting event or in finally
                self._logger.error("Session ready event set, but session is not available/initialized.")
                raise McpToolError("Failed to establish a valid MCP session.")
            
            self._logger.info("MCP session is ready.")

    async def _arun(self, command_to_execute: str, run_manager: Optional[Any] = None) -> str:
        """Asynchronously executes the bash command."""
        self._logger.info(f"MCPBashExecutorTool._arun received command_to_execute: {command_to_execute!r}") # Added logging
        if self._is_closed:
            self._logger.warning(f"Attempt to run command on closed tool: {self.name}")
            return "Error: Tool is closed."
        
        self._logger.info(f"Received command for _arun: {command_to_execute}")
        
        try:
            await self._ensure_session_ready()

            if not self._session:
                self._logger.error("MCP session not available for _arun.")
                return "Error: MCP session not available."

            bash_command_input_dict = {
                "command": command_to_execute, # This comes from args_schema BashInput's command_to_execute
                "timeout": int(self.tool_call_timeout)
            }
            # The MCP server tool `execute_bash_command_tool(input: BashCommandInput)`
            # expects the arguments for BashCommandInput to be under the key 'input'.
            mcp_server_tool_arguments = {"input": bash_command_input_dict}

            self._logger.debug(f"Calling MCP 'execute_bash_command' with arguments: {mcp_server_tool_arguments}")
            response: CallToolResult = await asyncio.wait_for(
                self._session.call_tool(name="execute_bash_command", arguments=mcp_server_tool_arguments),
                timeout=self.tool_call_timeout
            )

            self._logger.info(f"MCPBashExecutorTool._arun raw response from MCP server: {response!r}") # Added logging

            # The actual result is in response.content[0].text as a JSON string
            parsed_result = None
            if response and response.content and isinstance(response.content, list) and len(response.content) > 0:
                if isinstance(response.content[0], TextContent) and response.content[0].text:
                    try:
                        parsed_result = json.loads(response.content[0].text)
                        self._logger.debug(f"_arun: Successfully parsed JSON from response.content[0].text: {parsed_result}")
                    except json.JSONDecodeError as e:
                        self._logger.error(f"Failed to decode JSON from response.content[0].text: {e}. Raw text: {response.content[0].text!r}")
                        # Construct an error message similar to how it was handled before
                        return f"Invalid response: Failed to decode JSON. Details: {e}. Raw text: {response.content[0].text!r}"
                else:
                    self._logger.error(f"MCP server response.content[0] is not TextContent or text is empty. Full response: {response!r}")
            else:
                self._logger.error(f"MCP server response.content is missing or empty. Full response: {response!r}")

            if parsed_result and isinstance(parsed_result, dict):
                stdout_str = parsed_result.get('stdout', '')
                stderr_str = parsed_result.get('stderr', '')
                exit_code = parsed_result.get('exit_code', -99) # Default to -99 if not found
                self._logger.debug(f"_arun: Parsed Exit code: {exit_code}")
                if stdout_str: self._logger.debug(f"_arun: Parsed Stdout: {stdout_str[:200]}{'...' if len(stdout_str) > 200 else ''}")
                if stderr_str: self._logger.debug(f"_arun: Parsed Stderr: {stderr_str[:200]}{'...' if len(stderr_str) > 200 else ''}")

                # Check response.isError as a primary indicator of an issue reported by the server/SDK itself.
                # Also consider exit_code. If exit_code is -99, it implies parsing failed or key was missing.
                command_failed = response.isError or (exit_code != 0 and exit_code != -99) or exit_code == -99

                if command_failed:
                    msg_parts = [f"Command failed."]
                    if exit_code != -99: msg_parts.append(f"Exit code: {exit_code}.")
                    else: msg_parts.append("Exit code: Unknown/Parsing_Error.")
                    
                    # Append stdout/stderr if they exist, as they might contain useful info even on failure
                    if stdout_str: msg_parts.append(f"Stdout:\n{stdout_str}")
                    if stderr_str: msg_parts.append(f"Stderr:\n{stderr_str}")
                    elif not stderr_str and response.isError : msg_parts.append("Server error flag set, no specific stderr parsed.")
                    
                    return "\n".join(msg_parts)
                else:
                    return stdout_str if stdout_str else "Command executed successfully with no output."

            else: # This 'else' now covers cases where parsed_result is None or not a dict
                self._logger.error(f"MCP server returned an unexpected response structure or content. Full response: {response!r}, Parsed result: {parsed_result!r}")
                if not response:
                    return "No response received from server."
                
                if response.isError:
                    err_msg = "Server indicated an error."
                    if response.error and hasattr(response.error, 'message') and response.error.message: # Check MCP RpcError details
                        err_msg += f" Details: {response.error.message}"
                    # It's unlikely response.result would be a string if response.content[0].text was the source
                    # But we can check if parsed_result itself (if not a dict) could be an error string, though less likely now.
                    elif isinstance(parsed_result, str) and parsed_result: 
                        err_msg += f" Details from parsed content: {parsed_result}"
                    return err_msg

                # If not response.isError, but the structure of parsed_result is still not the expected dict:
                if parsed_result is None: # More specific check
                    return f"Invalid response: Parsed result was None. Full response: {response!r}"
                if not isinstance(parsed_result, dict):
                    type_of_result = type(parsed_result).__name__
                    return f"Invalid response structure: Parsed result was type '{type_of_result}', expected 'dict'. Full response: {response!r}"
                
                # Fallback for any other unhandled case within this 'else' block
                return f"Invalid or incomplete response from server (unknown structure issue). Full response: {response!r}"

        except asyncio.TimeoutError:
            self._logger.error(f"Timeout executing command: {command_to_execute}")
            return f"Error: Timeout executing command '{command_to_execute}'."
        except McpToolError as e: # This is our custom error for session/init issues
            self._logger.error(f"MCP Tool Error during command execution: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            self._logger.error(f"Unexpected error in _arun: {e}", exc_info=True)
            return f"Unexpected error: {str(e)}"

    async def close(self):
        """Closes the MCP session and stops the server."""
        if self._is_closed and (self._lifecycle_task is None or self._lifecycle_task.done()):
            self._logger.debug(f"{self.name} already closed or lifecycle task not running.")
            return
        
        self._logger.info(f"Closing {self.name}...")
        self._is_closed = True
        
        # Initialize primitives if `close` is called before any `_arun`
        await self._initialize_async_primitives_if_needed()

        if self._shutdown_event:
            self._shutdown_event.set()

        if self._lifecycle_task and not self._lifecycle_task.done():
            self._logger.debug("Waiting for lifecycle task to complete...")
            try:
                # Give it a chance to shut down gracefully
                await asyncio.wait_for(self._lifecycle_task, timeout=10.0)
            except asyncio.TimeoutError:
                self._logger.warning("Timeout waiting for lifecycle task to complete. Cancelling.")
                self._lifecycle_task.cancel()
                try:
                    await self._lifecycle_task # Await cancellation
                except asyncio.CancelledError:
                    self._logger.info("Lifecycle task successfully cancelled during close.")
                except Exception as e_final:
                    self._logger.error(f"Error awaiting cancelled lifecycle task: {e_final}")
            except Exception as e:
                self._logger.error(f"Error during lifecycle task shutdown: {e}")
        
        # Reset all events and session state for potential re-use, though typically new instance is created
        if self._session_ready_event: self._session_ready_event.clear()
        if self._shutdown_event: self._shutdown_event.clear()
        if self._session_failed_event: self._session_failed_event.clear()
        self._session = None
        self._stdio_client = None # Stdio client is closed by its context manager
        # self._lifecycle_task should be done or cancelled now

        self._logger.info(f"{self.name} closed.")

    def _run(self, command_to_execute: str, run_manager: Optional[Any] = None) -> str:
        """Synchronous execution is not supported for this async-native tool."""
        raise NotImplementedError(
            "MCPBashExecutorTool is async-native and does not support synchronous execution. Use _arun."
        )

# Example usage:
async def main_tool_test():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(name)s - [%(threadName)s] - %(message)s'
    )
    tool_logger = logging.getLogger("__main__.MCPBashExecutorTool_Test")
    tool_logger.info("Initializing MCPBashExecutorTool for example usage...")
    
    bash_tool = MCPBashExecutorTool(logger=tool_logger) # Pass the logger instance

    test_commands = [
        ("echo Hello Alpy from async MCP bash tool", False), # Expected to succeed
        ("ls /non_existent_path && echo This_should_not_print", True), # Expected to fail
        ("pwd && ls -la && echo 'Done listing'", False), # Expected to succeed
        # New test case for multi-line script
        ("""
        echo "Starting multi-line script test..."
        for i in 1 2 3; do
          echo "Loop iteration $i"
        done
        echo "Multi-line script test finished."
        """, False) # Expected to succeed
    ]

    try:
        for command, expect_failure in test_commands:
            tool_logger.info(f"Test: Running '{command}'")
            result = await bash_tool._arun(command)
            tool_logger.info(f"Result:\n{result}")
            if expect_failure and "Error" not in result:
                tool_logger.error(f"Expected failure, but command succeeded: {command}")
            elif not expect_failure and "Error" in result:
                tool_logger.error(f"Expected success, but command failed: {command}")
        
    except Exception as e:
        tool_logger.error(f"Error during example usage: {e}", exc_info=True)
    finally:
        tool_logger.info("Closing MCPBashExecutorTool from example usage's finally block.")
        await bash_tool.close()
        tool_logger.info("Example usage finished.")

if __name__ == '__main__':
    asyncio.run(main_tool_test())