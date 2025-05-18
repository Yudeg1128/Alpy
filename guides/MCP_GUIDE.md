## Guide: Implementing LangChain Tools for External MCP Servers

This guide outlines how to create Python-based LangChain tools that communicate with external Model Context Protocol (MCP) servers. It covers using the `mcp-sdk` (Python), handling `asyncio`, managing server subprocesses, Pydantic for data modeling, and integrating with LangChain.

**Core Philosophy:** For an asynchronous Alpy agent, tools should also be async-native to leverage the main event loop efficiently, avoiding custom threading for asyncio bridging.

### I. Understanding the Architecture

1.  **Alpy Agent (Async):** The primary application (e.g., `AlpyAgent`) is asynchronous, using `async/await` and an `asyncio` event loop.
2.  **LangChain Tool (Async Client):** Each MCP tool will be a Python class inheriting from `langchain_core.tools.BaseTool` and `pydantic.BaseModel`. It acts as an MCP *client*. Its main execution logic will be in `async def _arun(...)`.
3.  **MCP Server (External Process):** This is a separate process (e.g., a Python/FastMCP script, a Node.js application) that implements the MCP specification and exposes specific "tools" (RPC-like functions) that the LangChain tool will call.
4.  **Communication (`stdio_client`):** The LangChain tool will typically start its corresponding MCP server as a subprocess and communicate with it over standard input/output (stdio) using `mcp.client.stdio.stdio_client`.
5.  **MCP Session (`ClientSession`):** The `mcp.client.session.ClientSession` from the SDK manages the actual MCP message exchange (initialize, call_tool, notifications, etc.) over the stdio streams.

### II. Key MCP Python SDK Components and Usage

The `modelcontextprotocol/python-sdk` provides the necessary building blocks.

1.  **`mcp.client.stdio.StdioServerParameters`**:
    *   **Purpose:** Defines how to start the external MCP server subprocess.
    *   **Key Fields:**
        *   `command: str`: The absolute path to the executable (e.g., `/usr/bin/python3`, `/usr/local/bin/node`).
        *   `args: List[str]`: A list of arguments to pass to the command. The first argument is typically the path to the server script. Subsequent arguments are for the server script itself (e.g., allowed directories for a filesystem server).
        *   `cwd: str`: The current working directory for the server subprocess. This is crucial for the server to find its own relative files or modules (e.g., for a Node.js server, this should be its project root).
        *   `env: Optional[Dict[str, str]]`: Environment variables for the server process (defaults to `os.environ.copy()`).
    *   **Example:**
        ```python
        from mcp.client.stdio import StdioServerParameters
        params = StdioServerParameters(
            command="/usr/bin/python3",
            args=["/path/to/your/mcp_server.py", "--port", "8081"],
            cwd="/path/to/your/"
        )
        ```

2.  **`mcp.client.stdio.stdio_client`**:
    *   **Purpose:** An asynchronous context manager that starts the server subprocess (using `StdioServerParameters`) and provides `async` read/write streams connected to its stdio.
    *   **Usage:**
        ```python
        from mcp.client.stdio import stdio_client
        # async with stdio_client(server_params, errlog=sys.stderr) as (read_stream, write_stream):
        #    # read_stream and write_stream are now available
        ```
        *   `errlog=sys.stderr` (or a file) is highly recommended to capture the server's stderr for debugging.

3.  **`mcp.client.session.ClientSession`**:
    *   **Purpose:** Manages the MCP session over the provided read/write streams. Handles protocol handshakes, request/response mapping, and notifications.
    *   **Key `__init__` Parameters (refer to SDK for full list):**
        *   `read_stream`, `write_stream`: Obtained from `stdio_client`.
        *   `client_info: mcp.types.Implementation`: Describes your client tool (e.g., `MCPImplementation(name="MyToolClient", version="0.1.0")`).
        *   `logging_callback: Optional[Callable[[LoggingMessageNotificationParams], Awaitable[None]]])`: An `async` callback to handle log messages *from* the MCP server.
    *   **Key Methods:**
        *   `await session.initialize()`: Performs the MCP handshake. Must be called successfully before other operations. Returns `InitializeResult` containing server capabilities.
        *   `await session.call_tool(name: str, arguments: Optional[Dict[str, Any]] = None) -> CallToolResult`: Sends a `tools/call` request to the server.
            *   `name`: The exact name of the tool registered on the MCP server.
            *   `arguments`: A dictionary containing the parameters for the server-side tool. The structure of this dictionary *must* match what the server-side tool (and its Pydantic input model if using FastMCP) expects. Often, for FastMCP, if a server tool is `async def my_server_tool(input_param: MyInputModel)`, then `arguments` should be `{"input_param": {"field1": ..., "field2": ...}}`.
        *   `await session.close()`: Gracefully closes the session (sends shutdown/exit). Usually handled by the `async with ClientSession(...)` context manager's `__aexit__`.
    *   **Usage:**
        ```python
        from mcp.client.session import ClientSession
        from mcp.types import Implementation as MCPImplementation

        client_info = MCPImplementation(name="MyToolClient", version="0.1.0")
        # async with ClientSession(read_stream, write_stream, client_info=client_info) as session:
        #     init_res = await session.initialize()
        #     tool_res = await session.call_tool(name="server_tool_name", arguments={"param1": "value1"})
        ```

4.  **`mcp.types`**: This module contains all the Pydantic models for MCP messages and data structures.
    *   `CallToolResult`: The object returned by `session.call_tool()`. Key attributes:
        *   `isError: bool`: True if the server indicated an error.
        *   `content: Optional[List[Union[TextContent, ModelContent, ErrorData, ...]]]` : A list of content parts. For successful tool calls from FastMCP servers, this often contains a single `TextContent` (if the server tool returns a simple type or a JSON string) or a `ModelContent` (if the server tool returns a Pydantic model). For errors, it might contain `ErrorData`.
    *   `TextContent`: Has a `.text: str` attribute.
    *   `ErrorData`: Has `.message: str`, `.code: int`, etc.
    *   `InitializeResult`: Contains `.capabilities`, `.serverInfo`, etc.
    *   Other models for requests, notifications, and capabilities.

### III. Implementing an Async-Native LangChain Tool for an MCP Server

This structure assumes your LangChain agent (`AlpyAgent`) is asynchronous.

```python
import asyncio
import logging
import os
import shutil # For shutil.which
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Type, List # Add List if using it

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, model_validator, PrivateAttr # Pydantic v2

# MCP SDK Imports
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import (
    Implementation as MCPImplementation,
    # ClientCapabilities, # Not passed to ClientSession constructor directly
    InitializeResult,
    CallToolResult,
    TextContent,
    ErrorData as MCPErrorData, # Alias to avoid conflict
    LoggingMessageNotificationParams
)

# Define a logger for your tool module
logger = logging.getLogger(__name__) # Or a more specific name

class McpToolCustomError(ToolException): # Inherit from ToolException for LangChain
    """Custom error for this specific MCP tool."""
    pass

# 1. Define Pydantic Input Schema for LangChain
class MyToolLangChainInput(BaseModel):
    # Parameters the LLM will fill for this tool
    # Example:
    target_path: str = Field(description="The path for the operation.")
    some_option: bool = Field(default=False, description="An optional flag.")

# 2. Create the Tool Class
class MyAsyncMcpTool(BaseTool, BaseModel): # Inherit BaseModel for Pydantic config
    name: str = "my_async_mcp_tool"
    description: str = "Description of what this tool does for the LLM."
    args_schema: Type[BaseModel] = MyToolLangChainInput
    return_direct: bool = False # LangChain specific

    # Configuration for the MCP server process
    # Ensure Path types are used for path-like configurations
    server_executable: Path = Field(default_factory=lambda: Path(shutil.which("python3") or "python3"))
    server_script: Path = Field(default_factory=lambda: (Path(__file__).parent.parent.parent / "mcp_servers" / "my_mcp_server.py").resolve())
    server_cwd_path: Optional[Path] = Field(default=None, description="CWD for server, defaults to script's parent dir.")

    # Timeouts
    session_init_timeout: float = 30.0
    tool_call_timeout: float = 60.0 # Default for individual tool calls on the server

    # Internal Async State (using PrivateAttr for Pydantic v2 compatibility)
    _session: Optional[ClientSession] = PrivateAttr(default=None)
    _lifecycle_task: Optional[asyncio.Task] = PrivateAttr(default=None)
    _session_ready_event: Optional[asyncio.Event] = PrivateAttr(default=None)
    _shutdown_event: Optional[asyncio.Event] = PrivateAttr(default=None)
    _init_lock: Optional[asyncio.Lock] = PrivateAttr(default=None)
    _is_closed: bool = PrivateAttr(default=False)
    _logger_instance: logging.Logger = PrivateAttr(default=None) # Instance-specific logger

    _client_info: MCPImplementation = PrivateAttr(default_factory=lambda: MCPImplementation(name="MyAsyncToolClient", version="1.0.0"))

    class Config:
        arbitrary_types_allowed = True # For Path, asyncio objects etc.

    # 3. Pydantic Validator (Post-Initialization Setup)
    @model_validator(mode='after')
    def _tool_post_init(self) -> 'MyAsyncMcpTool':
        if self._logger_instance is None:
            self._logger_instance = logger.getChild(f"{self.name}.{id(self)}")
            # Add a basic handler if none are configured for this logger to ensure visibility
            if not self._logger_instance.hasHandlers():
                _handler = logging.StreamHandler(sys.stdout) # Or sys.stderr
                _formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s')
                _handler.setFormatter(_formatter)
                self._logger_instance.addHandler(_handler)
                self._logger_instance.propagate = False
        
        # Resolve paths and log them
        self.server_executable = self.server_executable.resolve()
        self._logger_instance.info(f"Using server_executable: {self.server_executable}")
        
        if not self.server_script.is_absolute():
            self.server_script = (Path(__file__).parent / self.server_script).resolve()
        self._logger_instance.info(f"Using server_script: {self.server_script}")
        if not self.server_script.exists():
             self._logger_instance.warning(f"Server script does not exist: {self.server_script}")


        if self.server_cwd_path is None:
            self.server_cwd_path = self.server_script.parent
        else:
            self.server_cwd_path = self.server_cwd_path.resolve()
        self._logger_instance.info(f"Using server_cwd_path: {self.server_cwd_path}")

        return self

    # 4. Async Primitives Initializer
    async def _initialize_async_primitives(self):
        """Initializes asyncio primitives if they don't exist."""
        if self._init_lock is None: self._init_lock = asyncio.Lock()
        if self._session_ready_event is None: self._session_ready_event = asyncio.Event()
        if self._shutdown_event is None: self._shutdown_event = asyncio.Event()

    # 5. Server Parameters Helper
    def _get_server_params(self) -> StdioServerParameters:
        if not self.server_executable.exists():
            msg = f"Server executable not found: {self.server_executable}"
            self._logger_instance.error(msg)
            raise McpToolCustomError(msg)
        if not self.server_script.exists():
            msg = f"Server script not found: {self.server_script}"
            self._logger_instance.error(msg)
            raise McpToolCustomError(msg)

        return StdioServerParameters(
            command=str(self.server_executable),
            args=[str(self.server_script)], # Add other server args if needed
            cwd=str(self.server_cwd_path)
        )

    # 6. MCP Server Logging Callback (Optional)
    async def _mcp_server_log_callback(self, params: LoggingMessageNotificationParams) -> None:
        level = getattr(logging, params.level.upper(), logging.INFO)
        self._logger_instance.log(level, f"[MCP Server - {params.scope}]: {params.message}")

    # 7. Session Lifecycle Management Task
    async def _manage_session_lifecycle(self):
        self._logger_instance.info("Starting MCP session lifecycle...")
        try:
            server_params = self._get_server_params()
            async with stdio_client(server_params, errlog=sys.stderr) as (rs, ws): # Log server stderr
                self._logger_instance.info("Stdio streams obtained.")
                async with ClientSession(
                    rs, ws, 
                    client_info=self._client_info,
                    logging_callback=self._mcp_server_log_callback # Optional
                ) as session:
                    self._logger_instance.info("ClientSession created. Initializing...")
                    init_result = await asyncio.wait_for(session.initialize(), timeout=self.session_init_timeout)
                    self._logger_instance.info(f"MCP session initialized. Server caps: {init_result.capabilities}")
                    self._session = session
                    self._session_ready_event.set()
                    self._logger_instance.info("Session ready. Waiting for shutdown signal...")
                    await self._shutdown_event.wait()
                    self._logger_instance.info("Shutdown signal received.")
        except asyncio.TimeoutError:
            self._logger_instance.error(f"Timeout ({self.session_init_timeout}s) during session initialization.")
        except asyncio.CancelledError:
            self._logger_instance.info("Session lifecycle task cancelled.")
        except McpToolCustomError as e: # Errors from _get_server_params
             self._logger_instance.error(f"Lifecycle error (setup): {e}")
        except Exception as e:
            self._logger_instance.error(f"Error in session lifecycle: {e}", exc_info=True)
        finally:
            self._logger_instance.info("Session lifecycle task finished.")
            self._session = None
            if self._session_ready_event and not self._session_ready_event.is_set():
                self._session_ready_event.set() # Unblock waiters, they'll find no session

    # 8. Ensure Session is Ready Helper
    async def _ensure_session_ready(self):
        if self._is_closed: raise McpToolCustomError(f"{self.name} is closed.")
        await self._initialize_async_primitives()

        if self._session and self._session_ready_event.is_set(): return

        async with self._init_lock: # Prevent concurrent initializations
            if self._session and self._session_ready_event.is_set(): return
            if self._is_closed: raise McpToolCustomError(f"{self.name} closed during readiness check.")

            if self._lifecycle_task is None or self._lifecycle_task.done():
                self._logger_instance.info("Starting new session lifecycle task.")
                self._session_ready_event.clear()
                self._shutdown_event.clear()
                self._lifecycle_task = asyncio.create_task(self._manage_session_lifecycle())
            
            try:
                await asyncio.wait_for(self._session_ready_event.wait(), timeout=self.session_init_timeout + 5.0)
            except asyncio.TimeoutError:
                self._logger_instance.error("Timeout waiting for session to become ready.")
                if self._lifecycle_task and not self._lifecycle_task.done(): self._lifecycle_task.cancel()
                raise McpToolCustomError("Timeout establishing MCP session.")

            if not self._session or not self._session_ready_event.is_set():
                raise McpToolCustomError("Failed to establish a valid MCP session.")
            self._logger_instance.info("MCP session is ready.")

    # 9. Main Asynchronous Execution Logic (_arun)
    async def _arun(self, target_path: str, some_option: bool = False, run_manager: Optional[Any] = None) -> str:
        """
        This is the method LangChain's async AgentExecutor will call.
        Input arguments match fields in MyToolLangChainInput.
        """
        if self._is_closed:
            return json.dumps({"error": f"{self.name} is closed."})
        
        self._logger_instance.info(f"Action: {self.name}, Path: {target_path}, Option: {some_option}")
        await self._ensure_session_ready()
        if not self._session:
            return json.dumps({"error": "MCP session not available."})

        # A. Determine the *actual* tool name on the MCP server
        #    This might be different from self.name or based on the action.
        #    For this example, assume a single server tool.
        server_tool_name = "my_actual_server_tool_name" # e.g., "do_filesystem_operation"

        # B. Prepare the 'arguments' dictionary for session.call_tool
        #    The structure of this dictionary *must* match what the server-side tool expects.
        #    If server tool is: `async def my_actual_server_tool_name(input_param: ServerInputModel)`
        #    Then arguments_payload should be: `{"input_param": {"path": target_path, "option": some_option}}`
        server_tool_params = {
            "path": target_path,
            "option": some_option
            # Add other params as expected by the server tool's input model
        }
        arguments_payload_for_server = {
            "input_param_name_on_server": server_tool_params # Replace 'input_param_name_on_server'
        }
        # OR, if the server tool takes parameters flatly:
        # arguments_payload_for_server = server_tool_params

        self._logger_instance.debug(f"Calling MCP server tool '{server_tool_name}' with payload: {arguments_payload_for_server}")
        try:
            response: CallToolResult = await asyncio.wait_for(
                self._session.call_tool(name=server_tool_name, arguments=arguments_payload_for_server),
                timeout=self.tool_call_timeout
            )

            if response.isError:
                err_msg = f"Server error for {server_tool_name}."
                if response.content and isinstance(response.content[0], MCPErrorData):
                    err_msg = response.content[0].message
                elif response.content and isinstance(response.content[0], TextContent):
                    err_msg = response.content[0].text
                self._logger_instance.error(err_msg)
                return json.dumps({"error": err_msg})

            if response.content and isinstance(response.content[0], TextContent):
                # Assuming server sends JSON string in TextContent for success
                # Or adapt if server sends plain text for some ops, or structured ModelContent
                return response.content[0].text # This should be a JSON string
            
            return json.dumps({"status": "success", "message": "Operation completed."})

        except asyncio.TimeoutError:
            return json.dumps({"error": f"Timeout calling {server_tool_name}."})
        except McpToolCustomError as e: # From _ensure_session_ready
            return json.dumps({"error": str(e)})
        except Exception as e:
            self._logger_instance.error(f"Unexpected error in _arun: {e}", exc_info=True)
            return json.dumps({"error": f"Unexpected error: {str(e)}"})

    # 10. Asynchronous Close Method
    async def close(self):
        if self._is_closed and (self._lifecycle_task is None or self._lifecycle_task.done()): return
        self._logger_instance.info(f"Closing {self.name}...")
        self._is_closed = True
        await self._initialize_async_primitives() # Ensure events exist

        if self._shutdown_event: self._shutdown_event.set()
        if self._lifecycle_task and not self._lifecycle_task.done():
            try:
                await asyncio.wait_for(self._lifecycle_task, timeout=10.0)
            except asyncio.TimeoutError: self._lifecycle_task.cancel()
            except Exception: pass # Logged in lifecycle task
        
        if self._session_ready_event: self._session_ready_event.clear()
        if self._shutdown_event: self._shutdown_event.clear()
        self._session = None
        self._logger_instance.info(f"{self.name} closed.")

    # 11. Synchronous _run (Optional, for compatibility if BaseTool requires it)
    def _run(self, *args: Any, **kwargs: Any) -> str:
        # For a purely async agent, this should ideally not be called.
        # If your LangChain setup might call it, you'd need a sync bridge here.
        # For simplicity in a fully async setup:
        raise NotImplementedError(
            f"{self.name} is an async-native tool. Use its `_arun` method."
        )
        # If a sync bridge is absolutely needed (like in your MCPFileSystemTool):
        # You'd re-introduce the threading.Thread, threading.Event, and asyncio.run_coroutine_threadsafe logic.
        # But for a new tool in an async agent, aim to avoid this.

    # 12. __del__ (Optional, for cleanup warnings)
    def __del__(self):
        if not self._is_closed and self._logger_instance:
             self._logger_instance.warning(f"{self.name} instance deleted without explicit close.")
        # Cannot reliably call async close() here.
```

### IV. Implementing the MCP Server (Example: Python with FastMCP)

Refer to your `bash_server.py` and `python_server.py` as excellent examples.

```python
# Example: mcp_servers/my_mcp_server.py
import asyncio
import logging
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # Basic config for server

mcp_app = FastMCP(
    name="MyCustomServer", 
    version="1.0.0",
    description="A custom MCP server."
)

# Define Pydantic models for your tool's input and output on the server
class ServerInputModel(BaseModel):
    path: str
    option: bool

class ServerOutputModel(BaseModel):
    result_data: str
    status_code: int

@mcp_app.tool(
    name="my_actual_server_tool_name", # This name MUST match what client's call_tool uses
    description="Performs a custom operation."
    # FastMCP infers input_model from 'input_param_name_on_server: ServerInputModel'
    # and output_model from '-> ServerOutputModel'
)
async def my_actual_server_tool_name(input_param_name_on_server: ServerInputModel) -> ServerOutputModel:
    logger.info(f"Server tool called with: {input_param_name_on_server.model_dump_json()}")
    
    # ... Your actual server-side logic here ...
    # Access input_param_name_on_server.path, input_param_name_on_server.option
    
    processed_result = f"Processed {input_param_name_on_server.path} with option {input_param_name_on_server.option}"
    return ServerOutputModel(result_data=processed_result, status_code=0)

if __name__ == "__main__":
    logger.info("Starting MyCustomServer via mcp_app.run()...")
    mcp_app.run() # Handles stdio communication
```

**Key points for the server:**
*   Use `FastMCP` for easy Python-based MCP server creation.
*   Define Pydantic models for tool inputs and outputs for automatic validation and schema generation.
*   The name given in `@mcp_app.tool(name="...")` is the crucial identifier for the client.
*   The parameter name in the tool function (e.g., `input_param_name_on_server`) is the key the client must use when structuring the `arguments` payload for `session.call_tool`.

### V. LangChain and Pydantic v2 Handling

*   **Pydantic v2:** LangChain is moving to Pydantic v2.
    *   Update your model imports: `from pydantic import BaseModel, Field, model_validator, PrivateAttr` instead of `langchain_core.pydantic_v1`.
    *   `@model_validator(mode='after')` is the Pydantic v2 way for post-init logic (replaces `@root_validator`).
    *   `PrivateAttr` can be used for internal state attributes that shouldn't be part of the Pydantic model's schema or serialization.
*   **`BaseTool` Integration:**
    *   Your tool class should inherit from `langchain_core.tools.BaseTool` and `pydantic.BaseModel`.
    *   Implement `name: str`, `description: str`, and `args_schema: Type[BaseModel]`.
    *   For async agents, implement `async def _arun(...)`. The parameters of `_arun` should match the fields defined in your `args_schema` Pydantic model. LangChain will use the `args_schema` to parse the LLM's tool input string into these arguments.
    *   The `_run` method can raise `NotImplementedError` if your agent is purely async.

### VI. Roadmap for Implementing a New MCP Tool

1.  **Define Server-Side Logic:**
    *   What operation(s) will the server perform?
    *   Choose a server implementation (e.g., Python/FastMCP, Node.js/mcp-sdk).
    *   If Python/FastMCP:
        *   Define Pydantic input/output models for each server tool.
        *   Implement the `async def tool_function(...)` using `@mcp_app.tool()`.
        *   Ensure the server script can be run (`if __name__ == "__main__": mcp_app.run()`).

2.  **Define LangChain Tool (Python Client):**
    *   Create a new Python file (e.g., `src/tools/my_new_mcp_tool.py`).
    *   Define the LangChain Pydantic input schema (`MyNewToolLangChainInput(BaseModel)`).
    *   Create the tool class `MyNewAsyncMcpTool(BaseTool, BaseModel)`.
        *   Set `name`, `description`, `args_schema`.
        *   Configure fields for `server_executable`, `server_script`, `server_cwd_path`.
        *   Implement the Pydantic `@model_validator(mode='after')` for logger setup and path resolution.
        *   Copy and adapt the async helper methods: `_initialize_async_primitives`, `_get_server_params`, `_mcp_server_log_callback` (optional), `_manage_session_lifecycle`, `_ensure_session_ready`.
        *   Implement `async def _arun(...)`:
            *   Its parameters should match `MyNewToolLangChainInput`.
            *   Call `await self._ensure_session_ready()`.
            *   Determine the correct server tool name(s) to call.
            *   Construct the `arguments` dictionary for `self._session.call_tool()` matching the server tool's expected input structure (often `{"server_param_name": {...your_data...}}`).
            *   Call `await self._session.call_tool(...)`.
            *   Parse the `CallToolResult` (handle `isError`, `content` which is likely `TextContent` containing a JSON string from the server).
            *   **Consistently return a JSON string** from `_arun`.
        *   Implement `async def close()`.
        *   Add a `NotImplementedError` `_run` method or remove it if LangChain doesn't strictly require it for async agents.
    *   Add an `if __name__ == "__main__":` block with `asyncio.run(test_function())` to test your tool standalone.

3.  **Integrate into `AlpyAgent`:**
    *   Import your new tool in `src/agent.py`.
    *   Instantiate it in `AlpyAgent.__init__`.
    *   Add it to `self.tools` list.
    *   Ensure its `close()` method is called in `AlpyAgent.close()`.

4.  **Testing:**
    *   Test the MCP server standalone (e.g., `python mcp_servers/my_mcp_server.py` and interact via a simple MCP client if needed, or rely on the tool's test).
    *   Test the LangChain tool standalone using its `if __name__ == "__main__":` test script.
    *   Test the integrated tool via the Alpy agent.

By following this detailed structure, focusing on async-native implementation for the tool client, and carefully matching the `arguments` payload with the server's expectations, you should be able to integrate new MCP servers more smoothly.