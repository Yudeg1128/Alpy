# src/tools/mcp_puppeteer_tool.py

import asyncio
import json
import logging
import os
import shutil # For shutil.which
import sys
import re # For parsing potential complex outputs if needed
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, Union

from langchain_core.tools import BaseTool, ToolException
from langchain_core.callbacks.manager import AsyncCallbackManagerForToolRun
from pydantic import BaseModel, Field, model_validator, PrivateAttr, RootModel, Json

# MCP SDK Imports
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import (
    Implementation as MCPImplementation,
    InitializeResult,
    CallToolResult,
    TextContent,
    ImageContent, # Screenshots might return ImageContent if not encoded
    ErrorData as MCPErrorData,
    LoggingMessageNotificationParams,
    LATEST_PROTOCOL_VERSION
)

# Module logger
logger = logging.getLogger(__name__)

class MCPPuppeteerToolError(ToolException):
    """Custom exception for the MCPPuppeteerTool."""
    pass

# --- Pydantic Schemas for Server Tool Parameters ---
# Define models for the parameters of each puppeteer server tool

class InnerPuppeteerNavigateAction(BaseModel):
    action: Literal["navigate"] = Field(default="navigate", frozen=True)
    url: str = Field(description="The URL to navigate the browser page to.")
    launchOptions: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional PuppeteerJS LaunchOptions object. If changed from previous non-null value, browser restarts. Example: {\"headless\": false}"
    )
    allowDangerous: Optional[bool] = Field(
        default=False,
        description="Allow dangerous LaunchOptions that reduce security (e.g., --no-sandbox). Default false."
    )

class InnerPuppeteerScreenshotAction(BaseModel):
    action: Literal["screenshot"] = Field(default="screenshot", frozen=True)
    name: str = Field(description="A descriptive name for the screenshot resource (e.g., 'login-page', 'search-results').")
    selector: Optional[str] = Field(default=None, description="Optional CSS selector for a specific element to screenshot. If omitted, captures the viewport.")
    # Width/Height are deprecated in modern Playwright/Puppeteer screenshot options usually, using viewport or fullPage.
    # Let's align with the README provided, acknowledging they might be older options.
    width: Optional[int] = Field(default=800, description="Screenshot width (may depend on viewport settings).")
    height: Optional[int] = Field(default=600, description="Screenshot height (may depend on viewport settings).")
    encoded: Optional[bool] = Field(
        default=False,
        description="If true, returns screenshot as base64-encoded data URI text. If false (default), returns binary image content (less useful for LLM directly, but creates screenshot:// resource)."
    )

class InnerPuppeteerClickAction(BaseModel):
    action: Literal["click"] = Field(default="click", frozen=True)
    selector: str = Field(description="CSS selector for the element to click.")

class InnerPuppeteerHoverAction(BaseModel):
    action: Literal["hover"] = Field(default="hover", frozen=True)
    selector: str = Field(description="CSS selector for the element to hover over.")

class InnerPuppeteerFillAction(BaseModel):
    action: Literal["fill"] = Field(default="fill", frozen=True)
    selector: str = Field(description="CSS selector for the input field to fill.")
    value: str = Field(description="The text value to fill into the input field.")

class InnerPuppeteerSelectAction(BaseModel):
    action: Literal["select"] = Field(default="select", frozen=True)
    selector: str = Field(description="CSS selector for the <select> element.")
    value: str = Field(description="The value of the <option> to select within the <select> element.")

class InnerPuppeteerEvaluateAction(BaseModel):
    action: Literal["evaluate"] = Field(default="evaluate", frozen=True)
    script: str = Field(description="A string containing JavaScript code to execute in the page's context.")

# --- Main LangChain Tool Input Schema ---
# This defines the structure of the `action_input` for MCPPuppeteerTool
# using a discriminated union based on the nested 'action' field.
class MCPPuppeteerActionInput(RootModel[Union[
    InnerPuppeteerNavigateAction,
    InnerPuppeteerScreenshotAction,
    InnerPuppeteerClickAction,
    InnerPuppeteerHoverAction,
    InnerPuppeteerFillAction,
    InnerPuppeteerSelectAction,
    InnerPuppeteerEvaluateAction
]]):
    root: Union[
        InnerPuppeteerNavigateAction,
        InnerPuppeteerScreenshotAction,
        InnerPuppeteerClickAction,
        InnerPuppeteerHoverAction,
        InnerPuppeteerFillAction,
        InnerPuppeteerSelectAction,
        InnerPuppeteerEvaluateAction
    ] = Field(..., discriminator='action')

    def __getattr__(self, item: str) -> Any:
        # Check if 'root' attribute itself exists and is initialized.
        # object.__getattribute__ is used to prevent recursive calls to __getattr__.
        try:
            root_obj = object.__getattribute__(self, 'root')
        except AttributeError:
            # 'root' itself doesn't exist yet (e.g. during Pydantic's own initialization),
            # so defer to Pydantic's __getattr__.
            return super().__getattr__(item)

        # If 'root' exists and 'item' is a field of the model stored in 'root'.
        if hasattr(root_obj, 'model_fields') and item in root_obj.model_fields:
            return getattr(root_obj, item)
        
        # Otherwise, defer to Pydantic's default __getattr__ behavior.
        return super().__getattr__(item)


# Main Tool Class (Async Version)
class MCPPuppeteerTool(BaseTool, BaseModel):
    name: str = "MCPPuppeteer" # This is the LangChain tool name
    description: str = (
        "Provides browser automation capabilities using Puppeteer via an MCP server launched with npx. "
        "Use this tool to navigate web pages, interact with elements (click, hover, fill forms, select dropdowns), "
        "take screenshots, and execute arbitrary JavaScript in the browser context.\n"
        "The `action_input` MUST be a JSON object containing an 'action' field specifying the operation, and its corresponding parameters.\n\n"
        "Available 'action' types for `action_input`:\n"
        "1. \"navigate\": Navigates the browser to a given URL.\n"
        "   - Required: 'url' (string).\n"
        "   - Optional: 'launchOptions' (object, e.g., {\"headless\": false}), 'allowDangerous' (boolean, default: false).\n"
        "2. \"screenshot\": Captures a screenshot of the current page or element.\n"
        "   - Required: 'name' (string, identifier for the screenshot resource like 'login-page').\n"
        "   - Optional: 'selector' (string, CSS selector for element), 'encoded' (boolean, default: false, if true returns base64 data URI text instead of binary).\n"
        "3. \"click\": Clicks an element matching the CSS selector.\n"
        "   - Required: 'selector' (string).\n"
        "4. \"hover\": Hovers over an element matching the CSS selector.\n"
        "   - Required: 'selector' (string).\n"
        "5. \"fill\": Enters text into an input field.\n"
        "   - Required: 'selector' (string, for the input field), 'value' (string, text to enter).\n"
        "6. \"select\": Selects an option within a <select> dropdown element.\n"
        "   - Required: 'selector' (string, for the <select> element), 'value' (string, the value attribute of the <option> to select).\n"
        "7. \"evaluate\": Executes JavaScript code in the page context.\n"
        "   - Required: 'script' (string, the JavaScript code to run). The result of the script (if any) is returned.\n\n"
        "IMPORTANT: Browser state is persistent between actions within this tool instance. Actions operate on the current page after navigation.\n\n"
        "Example - Navigate and take a screenshot:\n"
        "Action 1:\n"
        "```json\n"
        "{\"action\": \"MCPPuppeteer\", \"action_input\": {\"action\": \"navigate\", \"url\": \"https://example.com\"}}\n"
        "```\n"
        "Observation 1: (Output from navigate, often just confirms success or error)\n"
        "Action 2:\n"
        "```json\n"
        "{\"action\": \"MCPPuppeteer\", \"action_input\": {\"action\": \"screenshot\", \"name\": \"example-home\", \"encoded\": true}}\n"
        "```\n"
        "Observation 2: (Output from screenshot, likely containing the base64 data URI if encoded=true)\n\n"
        "Example - Fill a login form:\n"
        "Action 1 (Fill username):\n"
        "```json\n"
        "{\"action\": \"MCPPuppeteer\", \"action_input\": {\"action\": \"fill\", \"selector\": \"#username\", \"value\": \"myUser\"}}\n"
        "```\n"
        "Action 2 (Fill password):\n"
        "```json\n"
        "{\"action\": \"MCPPuppeteer\", \"action_input\": {\"action\": \"fill\", \"selector\": \"#password\", \"value\": \"myPass\"}}\n"
        "```\n"
        "Action 3 (Click login):\n"
        "```json\n"
        "{\"action\": \"MCPPuppeteer\", \"action_input\": {\"action\": \"click\", \"selector\": \"button[type='submit']\"}}\n"
        "```"
    )
    args_schema: Type[BaseModel] = MCPPuppeteerActionInput # Use the new main input schema
    return_direct: bool = False
    handle_tool_error: bool = True

    # Configuration for the MCP server process
    npx_executable_path: Path = Field(default_factory=lambda: Path(shutil.which("npx") or "npx"))
    server_package_name: str = Field(default="@modelcontextprotocol/server-puppeteer") # Correct package name
    server_process_cwd_str: Optional[str] = Field(default=None, description="Optional CWD for the npx process. Defaults to Alpy's current working directory.")

    session_init_timeout: float = 60.0 # Puppeteer server might take longer to init Playwright
    tool_call_timeout: float = 90.0 # Browser actions can take time

    # Internal Async State
    _session_async: Optional[ClientSession] = PrivateAttr(default=None)
    _lifecycle_task_async: Optional[asyncio.Task] = PrivateAttr(default=None)
    _session_ready_event_async: Optional[asyncio.Event] = PrivateAttr(default=None)
    _shutdown_event_async: Optional[asyncio.Event] = PrivateAttr(default=None)
    _init_lock_async: Optional[asyncio.Lock] = PrivateAttr(default=None)
    _is_closed_async: bool = PrivateAttr(default=False)
    _logger_instance: logging.Logger = PrivateAttr(default=None)
    _server_process_cwd_resolved: Optional[Path] = PrivateAttr(default=None)

    _client_info: MCPImplementation = PrivateAttr(default_factory=lambda: MCPImplementation(name="AlpyAsyncPuppeteerClient", version="1.0.0"))

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def _tool_post_init_validator(self) -> 'MCPPuppeteerTool':
        # --- Standard Logger and Path Setup (Identical to MCPFetcherWebTool) ---
        if self._logger_instance is None:
            self._logger_instance = logger.getChild(f"{self.name}.{id(self)}")
            if not self._logger_instance.hasHandlers() and not logging.getLogger().hasHandlers():
                _handler = logging.StreamHandler(sys.stdout)
                _formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s')
                _handler.setFormatter(_formatter)
                self._logger_instance.addHandler(_handler)
                self._logger_instance.propagate = False
                self._logger_instance.setLevel(logging.INFO)
        if str(self.npx_executable_path) == "npx" or not self.npx_executable_path.exists():
            resolved_npx = shutil.which("npx")
            if resolved_npx: self.npx_executable_path = Path(resolved_npx)
            else: raise ValueError("NPX executable not found. Ensure npx (Node.js/npm) is installed and in PATH.")
        self._logger_instance.info(f"NPX Executable: {self.npx_executable_path}")
        self._logger_instance.info(f"MCP Server Package (via npx): {self.server_package_name}")
        if self.server_process_cwd_str: self._server_process_cwd_resolved = Path(self.server_process_cwd_str).resolve()
        else: self._server_process_cwd_resolved = Path(os.getcwd()).resolve()
        self._logger_instance.info(f"CWD for NPX server process: {self._server_process_cwd_resolved}")
        # --- End Standard Setup ---

        # Add specific warning for Playwright dependency
        self._logger_instance.info("Note: This tool requires Playwright browsers to be installed. Run 'npx playwright install' if needed.")
        return self

    async def _initialize_async_primitives(self):
        # --- Standard Async Primitives Init (Identical to MCPFetcherWebTool) ---
        if self._init_lock_async is None: self._init_lock_async = asyncio.Lock()
        if self._session_ready_event_async is None: self._session_ready_event_async = asyncio.Event()
        if self._shutdown_event_async is None: self._shutdown_event_async = asyncio.Event()
        # --- End Standard Init ---

    def _get_server_params(self) -> StdioServerParameters:
        if not self.npx_executable_path or not self.npx_executable_path.exists():
            raise MCPPuppeteerToolError(f"NPX executable not found: {self.npx_executable_path}")
        if not self._server_process_cwd_resolved or not self._server_process_cwd_resolved.is_dir():
            raise MCPPuppeteerToolError(f"Server CWD for NPX is invalid: {self._server_process_cwd_resolved}")

        env = os.environ.copy()

        # *** ADD THIS SECTION ***
        # --- Attempt to override Chrome path ---
        # Replace this path with the actual path you found in step 1
        found_chrome_path = "/home/me/.cache/puppeteer/chrome/linux-136.0.7103.92/chrome-linux64/chrome" 
        if Path(found_chrome_path).exists():
            env["PUPPETEER_EXECUTABLE_PATH"] = found_chrome_path
            self._logger_instance.info(f"Setting env PUPPETEER_EXECUTABLE_PATH={found_chrome_path}")
        else:
            # Try finding system chrome as a fallback
            system_chrome = shutil.which("google-chrome") or shutil.which("chromium-browser") or shutil.which("chromium")
            if system_chrome:
                env["PUPPETEER_EXECUTABLE_PATH"] = system_chrome
                self._logger_instance.info(f"Setting env PUPPETEER_EXECUTABLE_PATH={system_chrome} (system browser)")
            else:
                self._logger_instance.warning(f"Could not find Chrome executable at {found_chrome_path} or system path. Relying on default Puppeteer discovery.")
        # --- End ADDED SECTION ---

        return StdioServerParameters(
            command=str(self.npx_executable_path.resolve()),
            args=["-y", self.server_package_name],
            cwd=str(self._server_process_cwd_resolved),
            env=env # Pass the modified environment
        )

    async def _mcp_server_log_callback(self, params: LoggingMessageNotificationParams) -> None:
        # --- Standard Log Callback (Identical to MCPFetcherWebTool) ---
        level = getattr(logging, params.level.upper(), logging.INFO)
        self._logger_instance.log(level, f"[MCP Server ({self.server_package_name}) - {params.scope}]: {params.message}")
        # --- End Standard Log Callback ---

    async def _manage_session_lifecycle(self):
        # --- Standard Session Lifecycle (Identical to MCPFetcherWebTool, uses self variables) ---
        self._logger_instance.info(f"Starting {self.name} (via npx {self.server_package_name}) MCP session lifecycle...")
        try:
            server_params = self._get_server_params()
            log_file_name = f"puppeteer_mcp_npx_server.log" # Unique log file
            puppeteer_log_path = Path(os.getcwd()) / log_file_name
            self._logger_instance.info(f"MCP Server ({self.server_package_name} via npx) stderr logged to: {puppeteer_log_path}")
            with open(puppeteer_log_path, 'a', encoding='utf-8') as ferr:
                async with stdio_client(server_params, errlog=ferr) as (rs, ws):
                    self._logger_instance.info(f"Stdio streams obtained for npx {self.server_package_name} server.")
                    async with ClientSession(rs, ws, client_info=self._client_info, logging_callback=self._mcp_server_log_callback) as session:
                        self._logger_instance.info(f"ClientSession created. Initializing {self.name} session...")
                        init_result: InitializeResult = await asyncio.wait_for(session.initialize(), timeout=self.session_init_timeout)
                        self._logger_instance.info(f"{self.name} MCP session initialized. Server ({self.server_package_name}) caps: {init_result.capabilities}")
                        self._session_async = session
                        self._session_ready_event_async.set()
                        self._logger_instance.info(f"{self.name} session ready. Waiting for shutdown signal...")
                        await self._shutdown_event_async.wait()
                        self._logger_instance.info(f"{self.name} shutdown signal received.")
        except asyncio.TimeoutError: self._logger_instance.error(f"Timeout ({self.session_init_timeout}s) during {self.name} session init.")
        except asyncio.CancelledError: self._logger_instance.info(f"{self.name} session lifecycle task cancelled.")
        except MCPPuppeteerToolError as e: self._logger_instance.error(f"Lifecycle error (setup for npx): {e}")
        except Exception as e: self._logger_instance.error(f"Error in {self.name} session lifecycle: {e}", exc_info=True)
        finally:
            self._logger_instance.info(f"{self.name} session lifecycle task finished.")
            self._session_async = None
            if self._session_ready_event_async and not self._session_ready_event_async.is_set(): self._session_ready_event_async.set()
        # --- End Standard Session Lifecycle ---

    async def _ensure_session_ready(self):
        # --- Standard Session Readiness Check (Identical to MCPFetcherWebTool) ---
        if self._is_closed_async: raise MCPPuppeteerToolError(f"{self.name} is closed.")
        await self._initialize_async_primitives()
        if self._session_async and self._session_ready_event_async.is_set(): return
        async with self._init_lock_async:
            if self._session_async and self._session_ready_event_async.is_set(): return
            if self._is_closed_async: raise MCPPuppeteerToolError(f"{self.name} closed during readiness check.")
            if self._lifecycle_task_async is None or self._lifecycle_task_async.done():
                self._logger_instance.info(f"Starting new {self.name} session lifecycle task.")
                self._session_ready_event_async.clear()
                self._shutdown_event_async.clear()
                self._lifecycle_task_async = asyncio.create_task(self._manage_session_lifecycle())
            else: self._logger_instance.info(f"Waiting for existing {self.name} session setup.")
            try: await asyncio.wait_for(self._session_ready_event_async.wait(), timeout=self.session_init_timeout + 10.0)
            except asyncio.TimeoutError:
                self._logger_instance.error(f"Timeout waiting for {self.name} session ready.")
                if self._lifecycle_task_async and not self._lifecycle_task_async.done(): self._lifecycle_task_async.cancel()
                raise MCPPuppeteerToolError(f"Timeout establishing {self.name} MCP session.")
            if not self._session_async or not self._session_ready_event_async.is_set():
                raise MCPPuppeteerToolError(f"Failed to establish valid {self.name} MCP session.")
            self._logger_instance.info(f"{self.name} MCP session is ready.")
        # --- End Standard Readiness Check ---

    def _process_mcp_response(self, response: CallToolResult, action_name: str) -> str:
        """Helper to process MCP response and return a JSON string."""
        self._logger_instance.debug(f"Processing MCP Response for action '{action_name}': isError={response.isError}, Content items: {len(response.content) if response.content else 0}")

        if response.isError:
            # ... (Error handling remains the same as previous version) ...
            error_message = f"MCP protocol error during action '{action_name}'."
            error_source = "MCP_PROTOCOL_ERROR"; error_code = None
            if response.content and isinstance(response.content, list) and len(response.content) > 0:
                item = response.content[0]
                if isinstance(item, MCPErrorData):
                    error_message = item.message or error_message; error_code = item.code
                    error_source = "SERVER_ERROR_DATA"
                    self._logger_instance.error(f"Server returned ErrorData: Code={item.code}, Msg='{item.message}', Data={item.data}")
                elif isinstance(item, TextContent) and item.text:
                    error_message = f"Server returned error text: {item.text}"; error_source = "SERVER_ERROR_TEXT"
                    self._logger_instance.error(error_message)
                else: error_message += f" Raw error content type: {type(item).__name__}, Value: {str(item)[:100]}"
            else: self._logger_instance.error(f"{error_message} (No specific details in response.content)")
            error_payload = {"status": "error", "error": error_message, "source": error_source}
            if error_code is not None: error_payload["code"] = error_code
            return json.dumps(error_payload)

        # --- Handling for successful response (isError=False) ---
        if response.content and isinstance(response.content, list) and len(response.content) > 0:
            # Special handling for screenshot - check all content parts
            if action_name == "screenshot":
                text_result = None
                base64_data = None
                image_resource_info = {"message": None, "media_type": None} # Store info about binary resource

                for item in response.content:
                    if isinstance(item, TextContent) and item.text:
                        if item.text.startswith("data:image/png;base64,"):
                            base64_data = item.text # Found base64
                            self._logger_instance.debug("Screenshot returned Base64 data URI.")
                        else:
                            text_result = item.text # Found confirmation text
                            self._logger_instance.debug("Screenshot returned confirmation text.")
                    elif isinstance(item, ImageContent):
                        # CORRECTED: Check for imageData or assume resource created
                        if hasattr(item, 'imageData') and item.imageData:
                             base64_data = item.imageData # Found base64 in ImageContent object
                             self._logger_instance.debug("Screenshot returned Base64 data URI via ImageContent.imageData.")
                        else:
                            # Assume binary resource was created, but URI isn't directly available via .uri
                            # We can infer the URI structure from the README: screenshot://<name>
                            # But we don't have the 'name' easily here unless we parse the text_result
                            image_resource_info["message"] = "Binary screenshot resource likely created by server."
                            if hasattr(item, 'mediaType') and item.mediaType:
                                image_resource_info["media_type"] = item.mediaType
                            self._logger_instance.debug(f"Screenshot likely created binary resource (MediaType: {image_resource_info['media_type']}). No direct URI found in ImageContent.")

                # Prioritize returning the most useful data
                if base64_data: # If base64 exists (either from Text or ImageContent)
                    return json.dumps({"status": "success", "result_type": "base64_data_uri", "result": base64_data, "message": text_result}) # Include confirmation message if also present
                elif image_resource_info["message"]: # If binary resource was detected
                     # Attempt to parse name from confirmation text to construct assumed URI
                     resource_uri_guess = None
                     if text_result and "Screenshot '" in text_result and "' taken" in text_result:
                          try:
                              name_match = re.search(r"Screenshot '(.*?)' taken", text_result)
                              if name_match:
                                  resource_uri_guess = f"screenshot://{name_match.group(1)}"
                          except Exception: pass # Ignore parsing errors

                     response_payload = {
                         "status": "success",
                         "message": text_result or image_resource_info["message"],
                         "resource_info": "Binary image created by server."
                     }
                     if resource_uri_guess:
                          response_payload["assumed_resource_uri"] = resource_uri_guess
                     if image_resource_info["media_type"]:
                          response_payload["media_type"] = image_resource_info["media_type"]
                     return json.dumps(response_payload)
                elif text_result: # Only confirmation text was found
                     return json.dumps({"status": "success", "message": text_result})
                else: # Shouldn't happen if content has items, fallback
                     return json.dumps({"status": "success", "message": f"Action '{action_name}' completed but content format unclear."})
            # --- End special screenshot handling ---

            # --- Handling for other actions ---
            item = response.content[0]
            if isinstance(item, TextContent):
                try: # Try parsing as JSON (for evaluate)
                    parsed_json = json.loads(item.text)
                    return json.dumps({"status": "success", "result": parsed_json})
                except json.JSONDecodeError: # Return raw text otherwise
                    return json.dumps({"status": "success", "result": item.text})
            elif isinstance(item, MCPErrorData):
                 self._logger_instance.error(f"Server returned MCPErrorData despite isError=False: {item.message}")
                 return json.dumps({"status": "error", "error": item.message or "Server error data.", "code": item.code, "source": "SERVER_ERROR_DATA_UNEXPECTED"})
            else:
                self._logger_instance.warning(f"Unhandled successful content type: {type(item).__name__} for action '{action_name}'")
                return json.dumps({"status": "success", "message": "Unhandled response format.", "raw_content_type": type(item).__name__})
        else:
             # Success but no content - typical for click, hover, fill, select, navigate
             return json.dumps({"status": "success", "message": f"Action '{action_name}' completed."})

    async def _arun(self, run_manager: Optional[AsyncCallbackManagerForToolRun] = None, **kwargs: Any) -> str:
        """Execute the specified Puppeteer action via MCP."""
        if self._is_closed_async:
            return json.dumps({"error": f"{self.name} is closed."})

        # The actual action and its parameters are within kwargs now,
        # matching the fields of the specific Inner...Action model chosen by the Union.
        inner_action = kwargs.get("action")
        if not inner_action:
            return json.dumps({"error": "Missing 'action' in action_input."})

        self._logger_instance.info(f"Executing Puppeteer Action: '{inner_action}', Args: { {k:v for k,v in kwargs.items() if k != 'action'} }")

        try:
            await self._ensure_session_ready()
            if not self._session_async:
                return json.dumps({"error": f"{self.name} session not available."})

            # Map inner action to server tool name and prepare args
            server_tool_name: Optional[str] = None
            server_tool_args: Dict[str, Any] = {}

            if inner_action == "navigate":
                server_tool_name = "puppeteer_navigate"
                schema = InnerPuppeteerNavigateAction.model_fields # Pydantic v2
                server_tool_args = {k: kwargs.get(k) for k in schema if k != 'action' and kwargs.get(k) is not None}
            elif inner_action == "screenshot":
                server_tool_name = "puppeteer_screenshot"
                schema = InnerPuppeteerScreenshotAction.model_fields
                server_tool_args = {k: kwargs.get(k) for k in schema if k != 'action' and kwargs.get(k) is not None}
            elif inner_action == "click":
                server_tool_name = "puppeteer_click"
                schema = InnerPuppeteerClickAction.model_fields
                server_tool_args = {k: kwargs.get(k) for k in schema if k != 'action' and kwargs.get(k) is not None}
            elif inner_action == "hover":
                server_tool_name = "puppeteer_hover"
                schema = InnerPuppeteerHoverAction.model_fields
                server_tool_args = {k: kwargs.get(k) for k in schema if k != 'action' and kwargs.get(k) is not None}
            elif inner_action == "fill":
                server_tool_name = "puppeteer_fill"
                schema = InnerPuppeteerFillAction.model_fields
                server_tool_args = {k: kwargs.get(k) for k in schema if k != 'action' and kwargs.get(k) is not None}
            elif inner_action == "select":
                server_tool_name = "puppeteer_select"
                schema = InnerPuppeteerSelectAction.model_fields
                server_tool_args = {k: kwargs.get(k) for k in schema if k != 'action' and kwargs.get(k) is not None}
            elif inner_action == "evaluate":
                server_tool_name = "puppeteer_evaluate"
                schema = InnerPuppeteerEvaluateAction.model_fields
                server_tool_args = {k: kwargs.get(k) for k in schema if k != 'action' and kwargs.get(k) is not None}
            else:
                return json.dumps({"error": f"Unknown inner action specified: {inner_action}"})

            # Validate required args for the specific action were provided
            # Pydantic validation on MCPPuppeteerActionInput should handle this before _arun is called,
            # but basic checks can be added here if needed.

            self._logger_instance.debug(f"Calling MCP server tool '{server_tool_name}' with arguments: {server_tool_args}")

            try:
                response: CallToolResult = await asyncio.wait_for(
                    self._session_async.call_tool(name=server_tool_name, arguments=server_tool_args),
                    timeout=self.tool_call_timeout
                )
            except asyncio.TimeoutError:
                self._logger_instance.error(f"Timeout calling MCP server tool '{server_tool_name}'.")
                return json.dumps({"error": f"Tool call to '{server_tool_name}' timed out."})

            # Process and return the response
            return self._process_mcp_response(response, inner_action)

        except MCPPuppeteerToolError as e: # Catch tool-specific errors (like session init failure)
            self._logger_instance.error(f"MCPPuppeteerToolError: {e}")
            return json.dumps({"status": "error", "error": str(e)})
        except Exception as e: # Catch any other unexpected errors
            self._logger_instance.error(f"Unexpected error during Puppeteer action '{inner_action}': {e}", exc_info=True)
            return json.dumps({"status": "error", "error": f"Unexpected error during '{inner_action}': {str(e)}"})

    async def close(self):
        # --- Standard Close Logic (Identical to MCPFetcherWebTool) ---
        if self._is_closed_async and (self._lifecycle_task_async is None or self._lifecycle_task_async.done()): return
        self._logger_instance.info(f"Closing {self.name} (npx {self.server_package_name})...")
        self._is_closed_async = True
        await self._initialize_async_primitives()
        if self._shutdown_event_async: self._shutdown_event_async.set()
        if self._lifecycle_task_async and not self._lifecycle_task_async.done():
            self._logger_instance.info(f"Waiting for {self.name} lifecycle task shutdown...")
            try: await asyncio.wait_for(self._lifecycle_task_async, timeout=10.0)
            except asyncio.TimeoutError:
                self._logger_instance.warning(f"Timeout waiting for {self.name} task. Cancelling.")
                self._lifecycle_task_async.cancel()
                try: await self._lifecycle_task_async
                except asyncio.CancelledError: self._logger_instance.info(f"{self.name} task cancelled.")
                except Exception as e_cancel: self._logger_instance.error(f"Error awaiting cancelled task: {e_cancel}")
            except Exception as e_wait: self._logger_instance.error(f"Error waiting for task: {e_wait}")
        if self._session_ready_event_async: self._session_ready_event_async.clear()
        if self._shutdown_event_async: self._shutdown_event_async.clear()
        self._session_async = None
        self._lifecycle_task_async = None
        self._logger_instance.info(f"{self.name} (npx {self.server_package_name}) closed.")
        # --- End Standard Close Logic ---

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError(f"{self.name} is async-native. Use its `_arun` method.")

    def __del__(self):
         # --- Standard __del__ Warning (Identical to MCPFetcherWebTool) ---
         if not self._is_closed_async and hasattr(self, '_logger_instance') and self._logger_instance:
              self._logger_instance.warning(f"{self.name} (ID: {id(self)}) deleted without explicit close.")
         # --- End Standard __del__ ---


# --- Example Usage (for standalone testing of this tool) ---
async def main_test_mcp_puppeteer_tool():
    log_level_str = os.getenv("LOG_LEVEL_TOOL_TEST", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s', stream=sys.stdout)
    if log_level > logging.DEBUG: logging.getLogger("asyncio").setLevel(logging.WARNING)
    test_logger = logging.getLogger(__name__ + ".MCPPuppeteerTool_Test")
    test_logger.info("Starting MCPPuppeteerTool standalone async test...")
    test_logger.warning("Ensure Playwright browsers are installed ('npx playwright install') before running this test.")

    tool: Optional[MCPPuppeteerTool] = None
    try:
        tool = MCPPuppeteerTool()
        if tool._logger_instance: tool._logger_instance.setLevel(logging.DEBUG) # Verbose tool logs for test

        print("\n--- [Test Case 1: Navigate to example.com] ---")
        # Simulate LLM providing action_input which LangChain parses into kwargs for _arun
        result_nav = await tool._arun(action="navigate", url="https://example.com")
        print(f"Result (Navigate): {result_nav}")
        json.loads(result_nav) # Validate JSON

        print("\n--- [Test Case 2: Take Base64 Encoded Screenshot] ---")
        result_ss_encoded = await tool._arun(action="screenshot", name="example-home-encoded", encoded=True)
        print(f"Result (Screenshot Encoded): {result_ss_encoded[:150]}...") # Show snippet
        json.loads(result_ss_encoded)

        print("\n--- [Test Case 3: Take Binary Screenshot (Resource)] ---")
        result_ss_binary = await tool._arun(action="screenshot", name="example-home-binary", encoded=False)
        print(f"Result (Screenshot Binary): {result_ss_binary}")
        json.loads(result_ss_binary)

        print("\n--- [Test Case 4: Evaluate JS to get title] ---")
        result_eval = await tool._arun(action="evaluate", script="document.title")
        print(f"Result (Evaluate): {result_eval}")
        json.loads(result_eval)

        print("\n--- [Test Case 5: Navigate to a page with inputs] ---")
        # Using a simple public form for testing (replace if unreliable)
        # Be careful with automation on external sites.
        # Let's use a local test file if possible, or a known simple form.
        # For now, let's simulate on example.com - these will likely fail but test the call structure.
        print("\n--- [Test Case 6: Fill (Simulated on example.com - expect failure)] ---")
        result_fill = await tool._arun(action="fill", selector="#nonexistent-input", value="test value")
        print(f"Result (Fill): {result_fill}")
        json.loads(result_fill)

        print("\n--- [Test Case 7: Click (Simulated on example.com - expect failure)] ---")
        result_click = await tool._arun(action="click", selector="a[href='https://www.iana.org/domains/example']") # Link exists on example.com
        print(f"Result (Click): {result_click}")
        json.loads(result_click)
        # Note: After clicking, the browser state changes. Subsequent actions might be on the new page.


    except Exception as e:
        test_logger.error(f"An error occurred during the MCPPuppeteerTool test: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
    finally:
        if tool:
            test_logger.info("Closing MCPPuppeteerTool...")
            await tool.close()
            test_logger.info("MCPPuppeteerTool closed.")
        test_logger.info("MCPPuppeteerTool standalone async test finished.")

if __name__ == "__main__":
    # IMPORTANT: Run `npx playwright install` before running this test.
    asyncio.run(main_test_mcp_puppeteer_tool())