# src/tools/mcp_playwright_tool.py

import asyncio
import json
import logging
import os
import shutil # For shutil.which
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, Union

from langchain_core.tools import BaseTool, ToolException
from langchain_core.callbacks.manager import AsyncCallbackManagerForToolRun # Optional
from pydantic import BaseModel, Field, model_validator, PrivateAttr, RootModel

# MCP SDK Imports
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import (
    Implementation as MCPImplementation,
    InitializeResult,
    CallToolResult,
    TextContent,
    ErrorData as MCPErrorData,
    LoggingMessageNotificationParams
)

# Module logger
logger = logging.getLogger(__name__)

class MCPPlaywrightToolError(ToolException):
    """Custom exception for the MCPPlaywrightTool."""
    pass

# --- Pydantic Schemas for LangChain Tool Actions (Input from LLM) ---
# These define the structure of `action_input` the LLM should generate.

class InnerPlaywrightNavigateAction(BaseModel):
    action: Literal["navigate"] = Field(default="navigate", frozen=True)
    url: Optional[str] = Field(default=None, description="URL to navigate. Optional if only connecting via wsEndpoint or if server already has a page.")
    launchOptions: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Playwright launch/connect options. To connect to an existing browser via its CDP WebSocket endpoint, include a key like 'wsEndpoint': 'ws://127.0.0.1:9222/devtools/browser/...'. "
            "If 'prompt_for_ws_endpoint' is true, this can be overridden by user input. "
            "For launching a new browser, typical options include {'headless': True/False} or {'browser_type': 'firefox'}. Default is Chromium."
        )
    )
    prompt_for_ws_endpoint: Optional[bool] = Field(
        default=False, 
        description="If true, the tool will prompt the user to enter a WebSocket endpoint at runtime. This will be used if provided, otherwise 'launchOptions' (if any) will be used."
    )

class InnerPlaywrightDescribeElementsAction(BaseModel):
    action: Literal["describe_elements"] = Field(default="describe_elements", frozen=True)

class InnerPlaywrightClickAction(BaseModel):
    action: Literal["click"] = Field(default="click", frozen=True)
    element_id: str = Field(description="The temporary ID of the element to click, from 'describe_elements'.")

class InnerPlaywrightHoverAction(BaseModel):
    action: Literal["hover"] = Field(default="hover", frozen=True)
    element_id: str = Field(description="The temporary ID of the element to hover, from 'describe_elements'.")

class InnerPlaywrightFillAction(BaseModel):
    action: Literal["fill"] = Field(default="fill", frozen=True)
    element_id: str = Field(description="The temporary ID of the input field to fill, from 'describe_elements'.")
    value: str = Field(description="The text value to fill.")

class InnerPlaywrightSelectAction(BaseModel):
    action: Literal["select"] = Field(default="select", frozen=True)
    element_id: str = Field(description="The temporary ID of the <select> element, from 'describe_elements'.")
    value: str = Field(description="The value of the <option> to select.")

class InnerPlaywrightScreenshotAction(BaseModel):
    action: Literal["screenshot"] = Field(default="screenshot", frozen=True)
    name: str = Field(description="A descriptive name for the screenshot (e.g., 'login-page').")
    element_id: Optional[str] = Field(default=None, description="Optional temporary ID of an element to screenshot.")
    encoded: Optional[bool] = Field(default=True, description="If true (default), returns base64 data URI.")
    full_page: Optional[bool] = Field(default=False, description="Capture full scrollable page (if element_id is not specified).")

class InnerPlaywrightEvaluateAction(BaseModel):
    action: Literal["evaluate"] = Field(default="evaluate", frozen=True)
    script: str = Field(description="JavaScript code to execute. If element_id is provided, element is 1st arg to script.")
    element_id: Optional[str] = Field(default=None, description="Optional temporary ID of an element for script context.")

class InnerPlaywrightPressKeyAction(BaseModel):
    action: Literal["press_key"] = Field(default="press_key", frozen=True)
    element_id: str = Field(description="The temporary ID of the focused element, from 'describe_elements'.")
    key: str = Field(description="Name of the key to press (e.g., 'Enter', 'Tab', 'ArrowDown', 'Control+C'). For a comprehensive list of key names, refer to Playwright documentation for `locator.press()`. 'Enter' is common for submitting forms.")

# --- Main LangChain Tool Input Schema ---
class MCPPlaywrightActionInput(RootModel[Union[
    InnerPlaywrightNavigateAction, InnerPlaywrightDescribeElementsAction,
    InnerPlaywrightClickAction, InnerPlaywrightHoverAction, InnerPlaywrightFillAction,
    InnerPlaywrightSelectAction, InnerPlaywrightScreenshotAction, InnerPlaywrightEvaluateAction,
    InnerPlaywrightPressKeyAction
]]):
    root: Union[
        InnerPlaywrightNavigateAction, InnerPlaywrightDescribeElementsAction,
        InnerPlaywrightClickAction, InnerPlaywrightHoverAction, InnerPlaywrightFillAction,
        InnerPlaywrightSelectAction, InnerPlaywrightScreenshotAction, InnerPlaywrightEvaluateAction,
        InnerPlaywrightPressKeyAction
    ] = Field(..., discriminator='action')

    def __getattr__(self, item: str) -> Any: # For convenience e.g. input.url
        try: root_obj = object.__getattribute__(self, 'root')
        except AttributeError: return super().__getattr__(item)
        if hasattr(root_obj, 'model_fields') and item in root_obj.model_fields: return getattr(root_obj, item)
        return super().__getattr__(item)

# --- Main Tool Class ---
class MCPPlaywrightTool(BaseTool, BaseModel):
    name: str = "MCPPlaywright"
    description: str = (
        "**IMPORTANT SESSION MANAGEMENT GUIDELINES:**\n"
        "1.  **STARTING A NEW BROWSING SESSION / RECOVERING CONNECTION:**\n"
        "    *   When you begin a new multi-step browsing task for the first time, OR if you suspect the browser connection has been lost or the state is unknown, your **very first action MUST be `navigate`**. \n"
        "    *   In this initial/recovery `navigate` call, you **MUST set `prompt_for_ws_endpoint: true`**. This allows the user to connect to an existing browser (via WebSocket endpoint) or to confirm the launch of a new browser instance. This is crucial for establishing the session correctly.\n"
        "2.  **CONTINUING AN ACTIVE BROWSING SESSION (after the initial setup):**\n"
        "    *   **If the browser is already open and active from a previous step in the current overall task:**\n"
        "        *   **To go to a new URL:** Call `navigate` again, but **ONLY provide the `url` argument.** DO NOT include `launchOptions` and DO NOT set `prompt_for_ws_endpoint: true` again. This reuses the existing browser session for the new URL.\n"
        "        *   **To interact with the CURRENT PAGE:** **DO NOT call `navigate` again.** Instead, use actions like `describe_elements` (to understand the current page), `click`, `fill`, `press_key`, etc., to interact directly with the elements on the page you are already on.\n"
        "    *   The tool is designed to keep the browser session active between your actions. Assume the session is active unless you receive an error indicating a disconnection or a problem with the browser state.\n\n"
        "--- Tool Details ---\n"
        "Automates web browser interactions using Playwright via a local MCP server. "
        "Supports launching a new browser instance (visible or headless) or connecting to an existing one if its WebSocket endpoint is provided (as per guideline #1).\n\n"
        "**Primary Interaction Workflow (Action Details):**\n"
        "1.  **`navigate`**: (Used for initial session setup or changing URLs in an active session - see SESSION MANAGEMENT above for when to use different parameters).\n"
        "    *   `url`: The target URL.\n"
        "    *   `launchOptions` (Optional): Dictionary to control browser launch (e.g., `{\"headless\": false}`, `{\"browser_type\": \"firefox\"}`). Primarily for initial setup (Guideline #1) or if you explicitly need to restart the browser with different settings. **Omit for subsequent URL changes in an active session (Guideline #2).**\n"
        "    *   `wsEndpoint` (within `launchOptions`): To connect to an **existing browser** via its CDP WebSocket endpoint. Primarily for initial setup.\n"
        "    *   `prompt_for_ws_endpoint` (boolean): As per SESSION MANAGEMENT Guideline #1, set to `true` for the very first `navigate` action of a session or if recovering. **Set to `false` or omit for subsequent `navigate` calls intended only to change the URL in an active session (Guideline #2).**\n"
        "2.  **`describe_elements`**: **This is the primary method for understanding the page content and identifying interactable elements.** After navigating or performing an action that changes the page, use this to get a list of interactable elements. Each element is returned with a unique `element_id`, its tag, text, and other attributes. This information is crucial for deciding the next action and for general orientation on the page.\n"
        "3.  **Interact using Element IDs**: Actions like `click`, `fill`, `hover`, `select`, and `press_key` **MUST** use the `element_id` argument. This `element_id` is a temporary identifier obtained from a previous call to `describe_elements`. **Do NOT use CSS selectors directly with these actions; always use the `element_id` provided by `describe_elements`.**\n"
        "4.  **Other Actions**: \n"
        "    *   `screenshot`: Capture an image of the current page or a specific element (using `element_id`). By default, returns base64 encoded image data. **Use this action sparingly, primarily when explicitly requested by the user, or if `describe_elements` is insufficient for understanding a complex visual layout or for verifying a visual state. Be aware that base64 image data can be very large and may exceed token limits.**\n"
        "    *   `evaluate`: Execute custom JavaScript code on the page or on a specific element.\n"
        "    *   `press_key`: Simulates pressing a key on a focused element (identified by `element_id`). Useful for submitting forms by pressing 'Enter' in a search bar, or for triggering keyboard shortcuts. Requires `element_id` of the target element and the `key` to press (e.g., 'Enter').\n\n"
        "**IMPORTANT GENERAL GUIDANCE:**\n"
        "*   **Prefer `describe_elements` for page orientation and element identification.**\n"
        "*   Element IDs obtained from `describe_elements` become **invalid** after navigation or actions that significantly change the page DOM. Always call `describe_elements` again on a new or modified page state before attempting further interactions with elements from the old state.\n\n"
        "**Examples:**\n\n"
        "1. Launch a new VISIBLE browser, navigate, and describe elements:\n"
        "   Action 1 (Navigate visible):\n"
        "   ```json\n"
        "   {\"action\": \"MCPPlaywright\", \"action_input\": {\"action\": \"navigate\", \"url\": \"https://example.com\", \"launchOptions\": {\"headless\": false}}}\n"
        "   ```\n"
        "   Observation 1: (JSON output from navigate, e.g., `{\"status\": \"success\", \"message\": \"Navigated to https://example.com/\", ...}`)\n\n"
        "   Action 2 (Describe elements - primary way to understand the page):\n"
        "   ```json\n"
        "   {\"action\": \"MCPPlaywright\", \"action_input\": {\"action\": \"describe_elements\"}}\n"
        "   ```\n"
        "   Observation 2: (JSON list of elements, e.g., `{\"status\": \"success\", \"elements\": [{\"element_id\": \"auto_el_0\", ...}]}`)\n\n"
        "2. Connect to an existing browser (user prompted), then, if specifically needed, take a screenshot:\n"
        "   Action 1 (Navigate & Connect - User Prompt):\n"
        "   ```json\n"
        "   {\"action\": \"MCPPlaywright\", \"action_input\": {\"action\": \"navigate\", \"url\": \"https://example.com\", \"prompt_for_ws_endpoint\": true}}\n"
        "   ```\n"
        "   (Tool will prompt user for WebSocket endpoint. If user provides `ws://127.0.0.1:9222/devtools/browser/ABC`, it connects there.)\n"
        "   Observation 1: (Output from navigate)\n\n"
        "   Action 2 (Optional: Screenshot, if textual description is insufficient or user asked):\n"
        "   ```json\n"
        "   {\"action\": \"MCPPlaywright\", \"action_input\": {\"action\": \"screenshot\", \"name\": \"example-page-for-visual-check\"}}\n"
        "   ```\n"
        "   Observation 2: (JSON output with base64 screenshot data)"
    )

    args_schema: Type[BaseModel] = MCPPlaywrightActionInput
    return_direct: bool = False 
    handle_tool_error: bool = True 

    server_executable: Path = Field(default_factory=lambda: Path(shutil.which("python") or "python3" or sys.executable))
    server_script: Path = Field(default_factory=lambda: (Path(__file__).resolve().parent.parent.parent / "mcp_servers" / "playwright_server.py"))
    server_cwd_path: Optional[Path] = Field(default=None)

    session_init_timeout: float = 75.0 
    tool_call_timeout: float = 90.0 

    _session_async: Optional[ClientSession] = PrivateAttr(default=None)
    _lifecycle_task_async: Optional[asyncio.Task] = PrivateAttr(default=None)
    _session_ready_event_async: Optional[asyncio.Event] = PrivateAttr(default=None)
    _shutdown_event_async: Optional[asyncio.Event] = PrivateAttr(default=None)
    _init_lock_async: Optional[asyncio.Lock] = PrivateAttr(default=None)
    _is_closed_async: bool = PrivateAttr(default=False)
    _logger_instance: logging.Logger = PrivateAttr(default=None)
    _client_info: MCPImplementation = PrivateAttr(default_factory=lambda: MCPImplementation(name="AlpyAsyncPlaywrightClient", version="1.0.2")) # Version bump

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def _tool_post_init_validator(self) -> 'MCPPlaywrightTool':
        if self._logger_instance is None:
            self._logger_instance = logger.getChild(f"{self.name}.{id(self)}")
            if not self._logger_instance.hasHandlers() and not logging.getLogger().hasHandlers():
                _handler = logging.StreamHandler(sys.stdout)
                _formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s')
                _handler.setFormatter(_formatter)
                self._logger_instance.addHandler(_handler)
                self._logger_instance.propagate = False
                self._logger_instance.setLevel(logging.INFO)
        
        self.server_executable = self.server_executable.resolve()
        self.server_script = self.server_script.resolve()
        if self.server_cwd_path is None: self.server_cwd_path = self.server_script.parent
        else: self.server_cwd_path = self.server_cwd_path.resolve()

        self._logger_instance.info(f"Server executable: {self.server_executable}")
        self._logger_instance.info(f"Server script: {self.server_script}")
        self._logger_instance.info(f"Server CWD: {self.server_cwd_path}")
        if not self.server_script.exists():
             self._logger_instance.warning(f"MCP Playwright server script DOES NOT EXIST: {self.server_script}")
        self._logger_instance.info("Note: Tool requires Playwright server with its own Playwright install (`pip install playwright && playwright install` in server's env).")
        return self

    async def _initialize_async_primitives(self):
        if self._init_lock_async is None: self._init_lock_async = asyncio.Lock()
        if self._session_ready_event_async is None: self._session_ready_event_async = asyncio.Event()
        if self._shutdown_event_async is None: self._shutdown_event_async = asyncio.Event()

    def _get_server_params(self) -> StdioServerParameters:
        if not self.server_executable.exists(): raise MCPPlaywrightToolError(f"Server exec not found: {self.server_executable}")
        if not self.server_script.exists(): raise MCPPlaywrightToolError(f"Server script not found: {self.server_script}")
        if not self.server_cwd_path or not self.server_cwd_path.is_dir(): raise MCPPlaywrightToolError(f"Server CWD invalid: {self.server_cwd_path}")
        return StdioServerParameters(command=str(self.server_executable), args=[str(self.server_script)], cwd=str(self.server_cwd_path), env=os.environ.copy())

    async def _mcp_server_log_callback(self, params: LoggingMessageNotificationParams) -> None:
        level = getattr(logging, params.level.upper(), logging.INFO)
        self._logger_instance.log(level, f"[MCP Server ({self.server_script.name}) - {params.scope}]: {params.message}")

    async def _manage_session_lifecycle(self):
        self._logger_instance.info(f"Starting {self.name} (server: {self.server_script.name}) MCP session lifecycle...")
        try:
            server_params = self._get_server_params()
            log_file_path = Path(os.getcwd()) / f"playwright_mcp_client_sees_{self.server_script.name}.log" # Distinguish client view of server log
            self._logger_instance.info(f"MCP Server ({self.server_script.name}) stderr will be logged to: {log_file_path}")
            
            with open(log_file_path, 'a', encoding='utf-8') as ferr:
                async with stdio_client(server_params, errlog=ferr) as (rs, ws):
                    self._logger_instance.info(f"Stdio streams obtained for {self.server_script.name}.")
                    async with ClientSession(rs, ws, client_info=self._client_info, logging_callback=self._mcp_server_log_callback) as session:
                        self._logger_instance.info(f"ClientSession created. Initializing {self.name} session...")
                        init_result: InitializeResult = await asyncio.wait_for(session.initialize(), timeout=self.session_init_timeout)
                        self._logger_instance.info(f"{self.name} MCP session initialized. Server caps: {init_result.capabilities}")
                        self._session_async = session
                        self._session_ready_event_async.set()
                        self._logger_instance.info(f"{self.name} session ready. Waiting for shutdown signal...")
                        await self._shutdown_event_async.wait()
                        self._logger_instance.info(f"{self.name} shutdown signal received.")
        except asyncio.TimeoutError: self._logger_instance.error(f"Timeout ({self.session_init_timeout}s) during {self.name} session init.")
        except asyncio.CancelledError: self._logger_instance.info(f"{self.name} session lifecycle task cancelled.")
        except MCPPlaywrightToolError as e: self._logger_instance.error(f"Lifecycle error (setup): {e}")
        except Exception as e: self._logger_instance.error(f"Error in {self.name} session lifecycle: {e}", exc_info=True)
        finally:
            self._logger_instance.info(f"{self.name} session lifecycle task finished.")
            self._session_async = None
            if self._session_ready_event_async and not self._session_ready_event_async.is_set():
                self._session_ready_event_async.set() 

    async def _ensure_session_ready(self):
        if self._is_closed_async: raise MCPPlaywrightToolError(f"{self.name} is closed.")
        await self._initialize_async_primitives()
        if self._session_async and self._session_ready_event_async.is_set(): return
        async with self._init_lock_async:
            if self._session_async and self._session_ready_event_async.is_set(): return
            if self._is_closed_async: raise MCPPlaywrightToolError(f"{self.name} closed during readiness check.")
            if self._lifecycle_task_async is None or self._lifecycle_task_async.done():
                self._logger_instance.info(f"Starting new {self.name} session lifecycle task.")
                self._session_ready_event_async.clear(); self._shutdown_event_async.clear()
                self._lifecycle_task_async = asyncio.create_task(self._manage_session_lifecycle())
            else: self._logger_instance.info(f"Waiting for existing {self.name} session setup.")
            try: await asyncio.wait_for(self._session_ready_event_async.wait(), timeout=self.session_init_timeout + 10.0)
            except asyncio.TimeoutError:
                self._logger_instance.error(f"Timeout waiting for {self.name} session ready.")
                if self._lifecycle_task_async and not self._lifecycle_task_async.done(): self._lifecycle_task_async.cancel()
                raise MCPPlaywrightToolError(f"Timeout establishing {self.name} MCP session.")
            if not self._session_async or not self._session_ready_event_async.is_set():
                raise MCPPlaywrightToolError(f"Failed to establish valid {self.name} MCP session.")
            self._logger_instance.info(f"{self.name} MCP session is ready.")

    def _process_mcp_response(self, response: CallToolResult, action_name: str) -> str:
        self._logger_instance.debug(f"Raw MCP Response for '{action_name}': isError={response.isError}, Content: {response.content}")
        if response.isError or not response.content or not isinstance(response.content[0], TextContent):
            err_msg = f"Error from MCP server for '{action_name}'."
            details = "Protocol error or unexpected response format."
            if response.content and isinstance(response.content[0], MCPErrorData): err_msg = response.content[0].message or err_msg
            elif response.content and isinstance(response.content[0], TextContent): err_msg = response.content[0].text # Server might put its JSON error here
            self._logger_instance.error(f"MCP Server Error for '{action_name}': {err_msg}")
            # Attempt to return server's JSON error string if available
            if response.content and isinstance(response.content[0], TextContent) and response.content[0].text.startswith('{"status": "error"'):
                return response.content[0].text
            return json.dumps({"status": "error", "error": "MCP_CLIENT_PROCESSING_ERROR", "message": err_msg, "details": details})

        # Expect TextContent with a JSON string from the server
        try:
            json.loads(response.content[0].text) # Validate it's JSON
            return response.content[0].text # Return the server's JSON string as is
        except (json.JSONDecodeError, AttributeError, IndexError) as e:
            self._logger_instance.error(f"Invalid/unexpected JSON from server for '{action_name}': {e}. Content: {str(response.content[0])[:200]}...")
            return json.dumps({"status": "error", "error": "INVALID_SERVER_JSON_RESPONSE", "message": "Server response was not valid JSON.", "details": str(response.content[0])[:200]})

    async def _arun(self, run_manager: Optional[AsyncCallbackManagerForToolRun] = None, **kwargs: Any) -> str:
        if self._is_closed_async: return json.dumps({"status": "error", "error": f"{self.name} is closed."})

        inner_action_name = kwargs.get("action")
        if not inner_action_name: return json.dumps({"status": "error", "error": "Missing 'action' in action_input."})
        self._logger_instance.info(f"Executing Playwright Action via MCP: '{inner_action_name}', Args (raw from LLM): {kwargs}")
        
        # Prepare payload for server (all kwargs except 'action' and client-side flags)
        server_payload = {k: v for k, v in kwargs.items() if k != 'action'}

        if inner_action_name == "navigate":
            if server_payload.pop("prompt_for_ws_endpoint", False): # Remove before sending to server
                try:
                    # Synchronous input in async context via to_thread
                    ws_endpoint_input = await asyncio.to_thread(
                        input, 
                        f"MCPPlaywrightTool ({self.name}): Enter Playwright WebSocket endpoint (e.g., ws://127.0.0.1:9222/...) or press Enter to skip: "
                    )
                    if ws_endpoint_input and ws_endpoint_input.strip():
                        current_launch_options = server_payload.get("launchOptions", {}) or {} # Ensure dict
                        current_launch_options["wsEndpoint"] = ws_endpoint_input.strip()
                        server_payload["launchOptions"] = current_launch_options
                        self._logger_instance.info(f"Using user-provided WS endpoint: {ws_endpoint_input.strip()}")
                    else: self._logger_instance.info("No WS endpoint from user, using existing launchOptions or server defaults.")
                except Exception as e:
                    self._logger_instance.error(f"Error during WS endpoint prompt: {e}")
                    return json.dumps({"status": "error", "error": "PROMPT_ERROR", "message": str(e)})
        
        # Wrap the payload under "input_data" key for the server
        mcp_arguments_for_server = {"input_data": server_payload}

        try:
            await self._ensure_session_ready()
            if not self._session_async: return json.dumps({"status": "error", "error": f"{self.name} session not available."})

            server_tool_name: str = f"playwright_{inner_action_name}"
            self._logger_instance.debug(f"Calling MCP server tool '{server_tool_name}' with wrapped arguments: {mcp_arguments_for_server}")

            try:
                response: CallToolResult = await asyncio.wait_for(
                    self._session_async.call_tool(name=server_tool_name, arguments=mcp_arguments_for_server),
                    timeout=self.tool_call_timeout
                )
            except asyncio.TimeoutError:
                self._logger_instance.error(f"Timeout calling MCP server tool '{server_tool_name}'.")
                return json.dumps({"status": "error", "error": "TOOL_CALL_TIMEOUT", "message": f"Tool call to '{server_tool_name}' timed out after {self.tool_call_timeout}s."})

            return self._process_mcp_response(response, inner_action_name)

        except MCPPlaywrightToolError as e:
            return json.dumps({"status": "error", "error": "MCP_TOOL_CLIENT_ERROR", "message": str(e)})
        except Exception as e:
            self._logger_instance.error(f"Unexpected error during Playwright action '{inner_action_name}': {e}", exc_info=True)
            return json.dumps({"status": "error", "error": "UNEXPECTED_CLIENT_ERROR", "message": f"Unexpected error: {str(e)}"})

    async def close(self): # Standard close logic
        if self._is_closed_async and (self._lifecycle_task_async is None or self._lifecycle_task_async.done()): return
        self._logger_instance.info(f"Closing {self.name} (server: {self.server_script.name})...")
        self._is_closed_async = True
        await self._initialize_async_primitives() 
        if self._shutdown_event_async: self._shutdown_event_async.set()
        if self._lifecycle_task_async and not self._lifecycle_task_async.done():
            self._logger_instance.info(f"Waiting for {self.name} lifecycle task shutdown...")
            try: await asyncio.wait_for(self._lifecycle_task_async, timeout=15.0)
            except asyncio.TimeoutError:
                self._logger_instance.warning(f"Timeout waiting for {self.name} task. Cancelling.")
                self._lifecycle_task_async.cancel(); await asyncio.sleep(0) # Allow cancellation to propagate
                try: await self._lifecycle_task_async
                except asyncio.CancelledError: self._logger_instance.info(f"{self.name} task cancelled.")
            except Exception: pass # Logged in task
        if self._session_ready_event_async: self._session_ready_event_async.clear()
        self._session_async = None; self._lifecycle_task_async = None
        self._logger_instance.info(f"{self.name} (server: {self.server_script.name}) closed.")

    def _run(self, *args: Any, **kwargs: Any) -> str: # Sync version not implemented
        raise NotImplementedError(f"{self.name} is async-native. Use `_arun` or `ainvoke`.")

    def __del__(self):
         if not self._is_closed_async and hasattr(self, '_logger_instance') and self._logger_instance:
              self._logger_instance.warning(f"{self.name} (ID: {id(self)}) instance deleted without explicit .close().")

# --- Example Usage (for standalone testing) ---
async def main_test_mcp_playwright_tool():
    log_level_str = os.getenv("LOG_LEVEL_TOOL_TEST", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    if not logging.getLogger().hasHandlers(): logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s [%(threadName)s] - %(message)s', stream=sys.stdout)
    if log_level > logging.DEBUG: logging.getLogger("asyncio").setLevel(logging.WARNING)
    test_logger = logging.getLogger(__name__ + ".MCPPlaywrightTool_Test")
    test_logger.setLevel(log_level)
    
    test_logger.info("Starting MCPPlaywrightTool standalone test...")
    test_logger.warning("Ensure 'playwright_server.py' is in mcp_servers/ and server's env has Playwright (`pip install playwright && playwright install`).")

    tool: Optional[MCPPlaywrightTool] = None
    try:
        tool = MCPPlaywrightTool()
        if tool._logger_instance and log_level <= logging.DEBUG: tool._logger_instance.setLevel(logging.DEBUG)

        print("\n--- [Test Case 1: Navigate example.com (default launch)] ---")
        # Note: action_input for _arun is already flattened by LangChain usually.
        # We simulate that by passing kwargs directly.
        result_json = await tool._arun(action="navigate", url="https://example.com")
        print(f"Result: {result_json}")
        result = json.loads(result_json)
        assert result.get("status") == "success", f"Navigation failed: {result.get('message')}"

        print("\n--- [Test Case 2: Describe elements] ---")
        result_json = await tool._arun(action="describe_elements")
        result = json.loads(result_json)
        assert result.get("status") == "success", f"Describe failed: {result.get('message')}"
        elements = result.get("elements", [])
        print(f"Found {len(elements)} elements. First few: {json.dumps(elements[:2], indent=2)}")
        assert isinstance(elements, list) and len(elements) > 0

        link_id = next((el['element_id'] for el in elements if el['tag'] == 'a' and 'information' in el.get('text','').lower()), None)
        if link_id:
            print(f"\n--- [Test Case 3: Click link '{link_id}'] ---")
            result_json = await tool._arun(action="click", element_id=link_id)
            print(f"Result: {result_json}")
            result = json.loads(result_json)
            assert result.get("status") == "success", f"Click failed: {result.get('message')}"
        else: test_logger.warning("Test Case 3: 'more information' link not found to test click.")

        print("\n--- [Test Case 4: Navigate (prompt for WS, user can skip for new launch)] ---")
        print("INFO: You will be prompted for a WebSocket endpoint. Press Enter to skip (tool launches new browser).")
        result_json = await tool._arun(action="navigate", url="https://httpbin.org/forms/post", prompt_for_ws_endpoint=True)
        print(f"Result: {result_json}")
        result = json.loads(result_json)
        assert result.get("status") == "success", f"Navigate with prompt failed: {result.get('message')}"
        
        print("\n--- [Test Case 5: Screenshot after nav to httpbin] ---")
        result_json = await tool._arun(action="screenshot", name="httpbin-forms-page", encoded=True)
        result = json.loads(result_json)
        assert result.get("status") == "success", f"Screenshot failed: {result.get('message')}"
        assert result.get("base64_data","").startswith("data:image/png;base64,")
        print(f"Screenshot successful (base64 data snippet): {result.get('base64_data', '')[:100]}...")

    except Exception as e:
        test_logger.error(f"Error during MCPPlaywrightTool test: {e}", exc_info=True)
    finally:
        if tool:
            test_logger.info("Closing MCPPlaywrightTool...")
            await tool.close()
            test_logger.info("MCPPlaywrightTool closed.")
        test_logger.info("MCPPlaywrightTool standalone test finished.")

if __name__ == "__main__":
    asyncio.run(main_test_mcp_playwright_tool())