# src/tools/mcp_media_display_tool.py

import asyncio
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Type, List, Literal
import tempfile

from langchain_core.tools import BaseTool, ToolException
from langchain_core.callbacks.manager import AsyncCallbackManagerForToolRun
from pydantic import BaseModel, Field, model_validator, PrivateAttr

# MCP SDK Imports - Assuming standard imports now
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client # Use the standard context manager
from mcp.types import (
    Implementation as MCPImplementation,
    InitializeResult, CallToolResult, TextContent, ErrorData as MCPErrorData,
    LoggingMessageNotificationParams, LATEST_PROTOCOL_VERSION
)

logger = logging.getLogger(__name__)

class MCPMediaDisplayToolError(ToolException):
    pass

class MCPMediaDisplayToolInput(BaseModel):
    source: str = Field(description="URL or absolute local file path of the media.")
    media_type: Optional[Literal["image", "audio", "video", "document", "unknown"]] = Field(default="unknown")

class MCPMediaDisplayTool(BaseTool, BaseModel):
    name: str = "MCPMediaDisplay"
    description: str = (
        "Displays media (images, audio, video, documents) from a URL or local file path using system defaults. "
        "The 'action_input' MUST be a JSON object containing a 'source' field with the URL or absolute local file path. " # Emphasize 'source' field
        "Optionally include a 'media_type' hint ('image', 'audio', 'video', 'document', 'unknown').\n"
        "If you receive a success message, it means the media is displayed successfully. At that point stop the agent loop."
        "Example: {\"source\": \"https://some.domain/image.jpg\", \"media_type\": \"image\"}\n" # Corrected example
        "Example: {\"source\": \"file:///path/to/your/local/file.pdf\"}"
        "IMPORTANT: When providing file URIs (starting with 'file://'), ensure special characters in filenames "
        "are correctly URL-encoded. For example, space is '%20', standard '?' is '%3F', "
        "full-width '？' is '%EF%BC%9F'.\n"
        "Example with special char: {\"source\": \"file:///path/to/file%20with%EF%BC%9Fspecial.mp3\"}"
    )
    args_schema: Type[BaseModel] = MCPMediaDisplayToolInput
    
    # Config resolved in validator
    _server_executable_path: Optional[Path] = PrivateAttr(default=None)
    _server_script_path: Optional[Path] = PrivateAttr(default=None)
    _server_cwd_path: Optional[Path] = PrivateAttr(default=None)

    session_init_timeout: float = 30.0
    tool_call_timeout: float = 60.0 

    # Internal Async State
    _session_async: Optional[ClientSession] = PrivateAttr(default=None)
    _lifecycle_task_async: Optional[asyncio.Task] = PrivateAttr(default=None)
    _session_ready_event_async: Optional[asyncio.Event] = PrivateAttr(default=None)
    _shutdown_event_async: Optional[asyncio.Event] = PrivateAttr(default=None)
    _init_lock_async: Optional[asyncio.Lock] = PrivateAttr(default=None)
    _is_closed_async: bool = PrivateAttr(default=False)
    _logger_instance: logging.Logger = PrivateAttr(default=None)

    _client_info: MCPImplementation = PrivateAttr(default_factory=lambda: MCPImplementation(name="AlpyMediaDisplayClient", version="1.0.0"))

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def _tool_post_init(self) -> 'MCPMediaDisplayTool':
        # Logger Setup
        if self._logger_instance is None:
            self._logger_instance = logger.getChild(f"{self.name}.{id(self)}")
            # ... (handler setup if needed - Ensure it's working or simplify)
            if not self._logger_instance.hasHandlers():
                 _handler = logging.StreamHandler(sys.stdout)
                 _formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s')
                 _handler.setFormatter(_formatter)
                 self._logger_instance.addHandler(_handler)
                 self._logger_instance.propagate = False
                 self._logger_instance.setLevel(logging.INFO)

        # --- START FIX ---
        # Directly use sys.executable which points to the CURRENT Python interpreter
        current_python_executable = sys.executable
        if current_python_executable:
             self._server_executable_path = Path(current_python_executable).resolve() # Resolve to absolute path
             if not self._server_executable_path.exists():
                 # This should be very rare if sys.executable is correct
                 self._logger_instance.error(f"sys.executable path does not seem to exist: {self._server_executable_path}")
                 self._server_executable_path = None # Mark as invalid
             else:
                 self._logger_instance.info(f"Resolved server python executable: {self._server_executable_path}")
        else:
             # Fallback if sys.executable is somehow None or empty (highly unlikely)
             self._logger_instance.error("sys.executable is not set. Trying PATH lookup.")
             py3_path = shutil.which("python3")
             py_path = shutil.which("python")
             if py3_path: self._server_executable_path = Path(py3_path)
             elif py_path: self._server_executable_path = Path(py_path)
             else: self._server_executable_path = None
        
        if not self._server_executable_path:
             self._logger_instance.critical("Could not determine Python executable path for server.")
             # Optionally raise error here to prevent tool init?
             # raise MCPMediaDisplayToolError("Failed to find Python executable.")

        # Resolve Server Script Path (relative to THIS file's location)
        try:
            # Assuming media_display_server.py is in ../../mcp_servers relative to this tool file
            script_rel_path = "../../mcp_servers/media_display_server.py"
            self._server_script_path = (Path(__file__).parent / script_rel_path).resolve()
            self._logger_instance.info(f"Resolved server script path: {self._server_script_path}")
            if not self._server_script_path.exists():
                self._logger_instance.warning(f"Server script path does not exist: {self._server_script_path}")
        except Exception as e:
            self._logger_instance.error(f"Error resolving server script path: {e}")
            self._server_script_path = None

        # Resolve CWD (e.g., Alpy project root or script's parent)
        try:
            # Assuming Alpy project root is 3 levels up from this tool file
            self._server_cwd_path = (Path(__file__).parent.parent.parent).resolve()
            self._logger_instance.info(f"Resolved server CWD path: {self._server_cwd_path}")
        except Exception as e:
             self._logger_instance.error(f"Error resolving server CWD path: {e}")
             self._server_cwd_path = None
        # --- END FIX ---
            
        return self

    async def _initialize_async_primitives(self):
        # Identical to other async tools
        if self._init_lock_async is None: self._init_lock_async = asyncio.Lock()
        if self._session_ready_event_async is None: self._session_ready_event_async = asyncio.Event()
        if self._shutdown_event_async is None: self._shutdown_event_async = asyncio.Event()

    def _get_server_params(self) -> StdioServerParameters:
        # ... (existing path checks) ...

        server_env = os.environ.copy() # Start with current environment

        # --- START ADDITION ---
        # Explicitly pass potentially required session variables
        # These are common ones needed for desktop interaction on Linux
        vars_to_pass = ["DISPLAY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"]
        self._logger_instance.debug("Checking for session environment variables to pass to server:")
        for var in vars_to_pass:
            value = os.environ.get(var)
            if value:
                server_env[var] = value
                self._logger_instance.debug(f"  - Passing: {var}={value}")
            else:
                 self._logger_instance.debug(f"  - Not found/set: {var}")
        # --- END ADDITION ---
        
        # API Key handling (if this tool needed one, it would go here)
        # if self._api_key_to_use: server_env["YOUR_API_KEY"] = self._api_key_to_use

        return StdioServerParameters(
            command=str(self._server_executable_path),
            args=[str(self._server_script_path)],
            cwd=str(self._server_cwd_path),
            env=server_env # Pass the modified environment
        )

    async def _mcp_server_log_callback(self, params: LoggingMessageNotificationParams) -> None:
        # Identical to other async tools
        level = getattr(logging, params.level.upper(), logging.INFO)
        self._logger_instance.log(level, f"[MCP Server - {params.scope}]: {params.message}")

    async def _manage_session_lifecycle(self):
        # Identical lifecycle management as the other async tools
        self._logger_instance.info(f"Starting {self.name} MCP session lifecycle...")
        try:
            server_params = self._get_server_params()
            async with stdio_client(server_params, errlog=sys.stderr) as (rs, ws):
                self._logger_instance.info(f"Stdio streams obtained for {self.name} server.")
                async with ClientSession(
                    rs, ws, client_info=self._client_info, logging_callback=self._mcp_server_log_callback
                ) as session:
                    self._logger_instance.info(f"ClientSession created. Initializing {self.name} session...")
                    init_result = await asyncio.wait_for(session.initialize(), timeout=self.session_init_timeout)
                    self._logger_instance.info(f"{self.name} MCP session initialized. Server caps: {init_result.capabilities}")
                    self._session_async = session
                    self._session_ready_event_async.set()
                    self._logger_instance.info(f"{self.name} session ready. Waiting for shutdown signal...")
                    await self._shutdown_event_async.wait()
                    self._logger_instance.info(f"{self.name} shutdown signal received.")
        # ... (identical error/finally handling as other async tools) ...
        except asyncio.TimeoutError: self._logger_instance.error(f"Timeout during {self.name} init.")
        except asyncio.CancelledError: self._logger_instance.info(f"{self.name} lifecycle task cancelled.")
        except Exception as e: self._logger_instance.error(f"Error in {self.name} lifecycle: {e}", exc_info=True)
        finally:
            self._logger_instance.info(f"{self.name} session lifecycle task finished.")
            self._session_async = None
            if self._session_ready_event_async and not self._session_ready_event_async.is_set():
                self._session_ready_event_async.set()

    async def _ensure_session_ready(self):
        # Identical readiness check logic as other async tools
        if self._is_closed_async: raise MCPMediaDisplayToolError(f"{self.name} is closed.")
        await self._initialize_async_primitives()
        if self._session_async and self._session_ready_event_async.is_set(): return

        async with self._init_lock_async:
            if self._session_async and self._session_ready_event_async.is_set(): return
            if self._is_closed_async: raise MCPMediaDisplayToolError(f"{self.name} closed during readiness check.")

            if self._lifecycle_task_async is None or self._lifecycle_task_async.done():
                self._logger_instance.info(f"Starting new {self.name} session lifecycle task.")
                self._session_ready_event_async.clear()
                self._shutdown_event_async.clear()
                self._lifecycle_task_async = asyncio.create_task(self._manage_session_lifecycle())
            
            try:
                await asyncio.wait_for(self._session_ready_event_async.wait(), timeout=self.session_init_timeout + 5.0)
            except asyncio.TimeoutError:
                self._logger_instance.error(f"Timeout waiting for {self.name} session.")
                if self._lifecycle_task_async and not self._lifecycle_task_async.done(): self._lifecycle_task_async.cancel()
                raise MCPMediaDisplayToolError(f"Timeout establishing {self.name} MCP session.")

            if not self._session_async or not self._session_ready_event_async.is_set():
                raise MCPMediaDisplayToolError(f"Failed to establish a valid {self.name} MCP session.")
            self._logger_instance.info(f"{self.name} MCP session is ready.")

    async def _arun(
        self, 
        source: str, 
        media_type: Optional[str] = "unknown", # Keep optional as per schema
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        if self._is_closed_async:
             return json.dumps({"error": f"{self.name} is closed."})
             
        self._logger_instance.info(f"Attempting to display media: source='{source}', type='{media_type}'")
        
        try: # Added try block for session readiness
            await self._ensure_session_ready()
            if not self._session_async:
                # This case should be less likely now if _ensure_session_ready raises properly
                return json.dumps({"error": f"{self.name} session not available."})

            # --- START FIX ---
            # Prepare the dictionary matching the MediaDisplayInput model on the server
            media_display_input_data = {
                "source": source,
                "media_type": media_type or "unknown" # Ensure default if None
            }

            # Nest this data under the key matching the server function's parameter name ('input_params')
            server_tool_arguments = {
                "input_params": media_display_input_data 
            }
            # --- END FIX ---
            
            server_tool_name = "display_media" # Assumed name on the server

            self._logger_instance.debug(f"Calling MCP server tool '{server_tool_name}' with arguments payload: {server_tool_arguments}")
            response: CallToolResult = await asyncio.wait_for(
                self._session_async.call_tool(name=server_tool_name, arguments=server_tool_arguments), # Pass the nested structure
                timeout=self.tool_call_timeout
            )
            
            # ... (rest of your response parsing logic - seems okay) ...
            self._logger_instance.debug(f"{self.name} Response: isError={response.isError}, Content type: {type(response.content)}")

            if response.isError:
                error_message = f"Server error displaying media '{source[:50]}...'."
                if response.content and isinstance(response.content, list) and len(response.content) > 0:
                    item = response.content[0]
                    if isinstance(item, MCPErrorData) and item.message: error_message = item.message
                    elif isinstance(item, TextContent) and item.text: error_message = f"Server error: {item.text}"
                    else: error_message += f" Raw: {str(item)[:100]}"
                self._logger_instance.error(error_message)
                return json.dumps({"error": error_message})

            message = "Media display requested." # Default success message
            if response.content and isinstance(response.content, list) and len(response.content) > 0:
                 item = response.content[0]
                 if isinstance(item, TextContent) and item.text:
                     # Try to parse the server's potential JSON output first
                     try:
                         server_output = json.loads(item.text)
                         message = server_output.get("message", item.text) # Use message field if available
                     except json.JSONDecodeError:
                         message = item.text # Use raw text if not JSON
            
            self._logger_instance.info(f"Media display request successful: {message}")
            return json.dumps({"status": "success", "message": message})


        except asyncio.TimeoutError:
            self._logger_instance.error(f"Timeout calling {server_tool_name} for '{source[:50]}...'.")
            return json.dumps({"error": f"Timeout requesting media display."})
        except MCPMediaDisplayToolError as e: # From _ensure_session_ready
            self._logger_instance.error(f"MCPMediaDisplayToolError: {e}")
            return json.dumps({"error": str(e)})
        except Exception as e:
            self._logger_instance.error(f"Unexpected error during media display request: {e}", exc_info=True)
            return json.dumps({"error": f"Unexpected error: {str(e)}"})

    async def close(self):
        # Identical async close logic as other tools
        if self._is_closed_async and (self._lifecycle_task_async is None or self._lifecycle_task_async.done()): return
        self._logger_instance.info(f"Closing {self.name}...")
        self._is_closed_async = True
        await self._initialize_async_primitives()
        if self._shutdown_event_async: self._shutdown_event_async.set()
        if self._lifecycle_task_async and not self._lifecycle_task_async.done():
            try: await asyncio.wait_for(self._lifecycle_task_async, timeout=10.0)
            except asyncio.TimeoutError: self._lifecycle_task_async.cancel()
            except Exception: pass
        if self._session_ready_event_async: self._session_ready_event_async.clear()
        if self._shutdown_event_async: self._shutdown_event_async.clear()
        self._session_async = None
        self._logger_instance.info(f"{self.name} closed.")

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError(f"{self.name} is async-native. Use _arun.")

    def __del__(self):
        if not self._is_closed_async and hasattr(self, '_logger_instance') and self._logger_instance:
            self._logger_instance.warning(f"{self.name} instance deleted without explicit close.")

# Example usage (requires media_display_server.py and src.config)
async def main_media_test():
    log_level_str = os.getenv("LOG_LEVEL", "DEBUG").upper() # Default to DEBUG for testing
    log_level = getattr(logging, log_level_str, logging.DEBUG)
    logging.basicConfig(level=log_level)
    test_logger = logging.getLogger(__name__ + ".MediaDisplay_Test")
    test_logger.info("Starting MCPMediaDisplayTool async example usage...")

    # Create a dummy text file for testing local path
    temp_dir = tempfile.TemporaryDirectory()
    local_file_path = Path(temp_dir.name) / "test_display.txt"
    local_file_path.write_text("This is a test document.")
    local_file_uri = local_file_path.as_uri() # e.g., file:///tmp/tmpxxxxx/test_display.txt

    # Example image URL (replace with a working one if needed)
    image_url = "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png"

    tool = MCPMediaDisplayTool()
    tool._logger_instance.setLevel(log_level) # Set tool logger level

    try:
        print("\n--- Testing Image URL ---")
        result = await tool._arun(source=image_url, media_type="image")
        print(f"Result: {result}")
        
        print("\n--- Testing Local Document ---")
        result = await tool._arun(source=local_file_uri, media_type="document")
        print(f"Result: {result}")
        
        # Test with just path instead of URI
        print("\n--- Testing Local Document (using path) ---")
        result = await tool._arun(source=str(local_file_path), media_type="document")
        print(f"Result: {result}")

        print("\n--- Testing Unknown Type (server should try to guess) ---")
        result = await tool._arun(source=local_file_uri) # No media_type hint
        print(f"Result: {result}")

    except Exception as e:
        print(f"\nAn error occurred during testing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        print("\nClosing tool...")
        await tool.close()
        print("Tool closed.")
        temp_dir.cleanup() # Clean up temporary file/dir

if __name__ == "__main__":
    asyncio.run(main_media_test())