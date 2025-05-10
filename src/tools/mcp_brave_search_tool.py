# src/tools/mcp_brave_search_tool.py

import asyncio
import json
import logging
import os
import shutil # For shutil.which
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Type, List, Literal

from langchain_core.tools import BaseTool, ToolException
from langchain_core.callbacks.manager import AsyncCallbackManagerForToolRun # Often optional unless using run manager explicitly
from pydantic import BaseModel, Field, model_validator, PrivateAttr # Pydantic v2

# MCP SDK Imports
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import (
    Implementation as MCPImplementation,
    InitializeResult,
    CallToolResult,
    TextContent,
    ErrorData as MCPErrorData,
    LoggingMessageNotificationParams,
    LATEST_PROTOCOL_VERSION # Assuming this is available or use specific version string
)

# Attempt to import BRAVE_SEARCH_API_KEY from Alpy's config
try:
    # Assumes config.py is in src/, and this tool is in src/tools/
    from ..config import BRAVE_SEARCH_API_KEY 
except ImportError:
    # Fallback if config structure is different or key isn't defined
    BRAVE_SEARCH_API_KEY = None 
    logging.warning("Could not import BRAVE_SEARCH_API_KEY from src.config. Falling back to environment variable.")

# Module logger
logger = logging.getLogger(__name__)

class MCPBraveSearchToolError(ToolException):
    """Custom exception for the MCPBraveSearchTool."""
    pass

# Pydantic Input Schema for LangChain Tool
# This defines what the LLM needs to structure within the 'action_input'
class MCPBraveSearchLangChainInput(BaseModel):
    search_type: Literal["web", "news", "image", "video", "summarize"] = Field(
        description="The type of Brave search to perform."
    )
    query: str = Field(description="The search query string.")
    count: Optional[int] = Field(default=5, description="Number of results desired (1-20).", ge=1, le=20)
    offset: Optional[int] = Field(default=0, description="Pagination offset (0-9 for web/image/video, >=0 for news).", ge=0)
    country: Optional[str] = Field(default=None, description="Two-letter country code (e.g., 'US', 'GB').")
    search_lang: Optional[str] = Field(default=None, description="Search language code (e.g., 'en').")
    safesearch: Optional[Literal["off"]] = Field(default="off", description="Safesearch level.")
    # Add specific fields if needed only for certain types (though often handled by server)
    # e.g., freshness for video search could be added here if desired, or passed via general kwargs if server accepts them loosely.


# Main Tool Class (Async Version)
class MCPBraveSearchTool(BaseTool, BaseModel): # Inherit from BaseModel for Pydantic features
    name: str = "MCPBraveSearch" # This is the single tool name the LLM will use
    description: str = (
        "Performs different types of searches (web, news, image, video, summarize) using the Brave Search API "
        "via a dedicated MCP server. Requires 'search_type' and 'query' in the input. "
        "The 'action_input' MUST be a JSON object specifying the 'search_type' (one of 'web', 'news', 'image', 'video', 'summarize'), "
        "the 'query', and optionally other parameters like 'count', 'offset', 'country', 'search_lang', 'safesearch'.\n"
        "Example for web search: {\"search_type\": \"web\", \"query\": \"latest AI news\", \"count\": 3}\n"
        "Example for news search: {\"search_type\": \"news\", \"query\": \"LangChain updates\"}\n"
        "Example for summarizer: {\"search_type\": \"summarize\", \"query\": \"What is the Model Context Protocol?\"}"
    )
    args_schema: Type[BaseModel] = MCPBraveSearchLangChainInput
    return_direct: bool = False
    handle_tool_error: bool = True

    # Configuration for the MCP server process (using custom brave_server.py)
    python_executable_for_server: Path = Field(
        default_factory=lambda: Path(shutil.which("python3") or shutil.which("python") or sys.executable or "python")
    )
    server_script_path_relative: str = Field(default_factory=lambda: os.path.join(
         "..", "..", "mcp_servers", "brave_server.py" # Relative to this tool file
    ))
    server_cwd_path_str: Optional[str] = Field(default=None, description="CWD for server, defaults to server script's directory.")

    _api_key_to_use: Optional[str] = PrivateAttr(default=None) # Will be populated in validator

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
    _server_script_path_resolved: Optional[Path] = PrivateAttr(default=None)
    _server_cwd_path_resolved: Optional[Path] = PrivateAttr(default=None)

    _client_info: MCPImplementation = PrivateAttr(default_factory=lambda: MCPImplementation(name="AlpyAsyncBraveSearchClient", version="1.1.0"))

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def _tool_post_init_validator(self) -> 'MCPBraveSearchTool':
        # Logger setup
        if self._logger_instance is None:
            self._logger_instance = logger.getChild(f"{self.name}.{id(self)}")
            if not self._logger_instance.hasHandlers() and not logging.getLogger().hasHandlers():
                _handler = logging.StreamHandler(sys.stdout) 
                _formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s')
                _handler.setFormatter(_formatter)
                self._logger_instance.addHandler(_handler)
                self._logger_instance.propagate = False
                self._logger_instance.setLevel(logging.INFO) # Default level

        # Resolve Python executable for server
        if not self.python_executable_for_server or str(self.python_executable_for_server) == "python":
            resolved = shutil.which("python3") or shutil.which("python") or sys.executable
            if resolved: self.python_executable_for_server = Path(resolved)
            else: self._logger_instance.critical("Python interpreter not found for server.")
        self._logger_instance.info(f"Server Python Executable: {self.python_executable_for_server}")

        # Resolve server script path
        self._server_script_path_resolved = (Path(__file__).parent / self.server_script_path_relative).resolve()
        self._logger_instance.info(f"Server Script Path: {self._server_script_path_resolved}")
        if not self._server_script_path_resolved.exists():
             self._logger_instance.warning(f"Server script does not exist: {self._server_script_path_resolved}")

        # Resolve CWD for server
        if self.server_cwd_path_str:
            self._server_cwd_path_resolved = Path(self.server_cwd_path_str).resolve()
        else: # Default to script's directory
            self._server_cwd_path_resolved = self._server_script_path_resolved.parent
        self._logger_instance.info(f"Server CWD: {self._server_cwd_path_resolved}")

        # Determine API Key
        self._api_key_to_use = BRAVE_SEARCH_API_KEY # From src.config import
        if not self._api_key_to_use:
            self._api_key_to_use = os.getenv("BRAVE_SEARCH_API_KEY")
        if not self._api_key_to_use:
            self._logger_instance.warning("Brave API key not found in config or environment.")
        else:
             self._logger_instance.info("Brave Search API key configured.")
             
        return self

    async def _initialize_async_primitives(self):
        """Initializes asyncio primitives."""
        if self._init_lock_async is None: self._init_lock_async = asyncio.Lock()
        if self._session_ready_event_async is None: self._session_ready_event_async = asyncio.Event()
        if self._shutdown_event_async is None: self._shutdown_event_async = asyncio.Event()

    def _get_server_params(self) -> StdioServerParameters:
        """Prepares parameters for launching the server."""
        if not self.python_executable_for_server or not self.python_executable_for_server.exists():
             raise MCPBraveSearchToolError(f"Server Python executable not found: {self.python_executable_for_server}")
        if not self._server_script_path_resolved or not self._server_script_path_resolved.exists():
             raise MCPBraveSearchToolError(f"Server script not found: {self._server_script_path_resolved}")
        if not self._server_cwd_path_resolved or not self._server_cwd_path_resolved.is_dir():
             raise MCPBraveSearchToolError(f"Server CWD is invalid: {self._server_cwd_path_resolved}")

        env = os.environ.copy() # Start with a copy of the current environment
        # self._api_key_to_use is loaded from src.config by _tool_post_init_validator.
        # We are no longer passing it via environment to the server.
        # The server (brave_server.py) is now responsible for loading it directly from src.config.
        if self._api_key_to_use:
            self._logger_instance.info(f"Tool is aware of BRAVE_SEARCH_API_KEY (length: {len(self._api_key_to_use)}) from config.")
        else:
            self._logger_instance.warning("Tool: BRAVE_SEARCH_API_KEY not found in config or is empty.")

        server_command = [
            str(self.python_executable_for_server.resolve()),
            str(self._server_script_path_resolved)
        ]

        return StdioServerParameters(
            command=server_command[0],
            args=server_command[1:],
            cwd=str(self._server_cwd_path_resolved),
            env=env
        )

    async def _mcp_server_log_callback(self, params: LoggingMessageNotificationParams) -> None:
        """Handles log messages from the MCP server."""
        level = getattr(logging, params.level.upper(), logging.INFO)
        self._logger_instance.log(level, f"[MCP Server - {params.scope}]: {params.message}")

    async def _manage_session_lifecycle(self):
        """Manages the MCP server process and client session."""
        # Identical lifecycle management as the other async tools
        self._logger_instance.info(f"Starting {self.name} MCP session lifecycle...")
        try:
            server_params = self._get_server_params()
            async with stdio_client(server_params, errlog=sys.stderr) as (rs, ws):
                self._logger_instance.info(f"Stdio streams obtained for {self.name} server.")
                async with ClientSession(
                    rs, ws, 
                    client_info=self._client_info,
                    logging_callback=self._mcp_server_log_callback
                ) as session:
                    self._logger_instance.info(f"ClientSession created. Initializing {self.name} session...")
                    init_result = await asyncio.wait_for(session.initialize(), timeout=self.session_init_timeout)
                    self._logger_instance.info(f"{self.name} MCP session initialized. Server caps: {init_result.capabilities}")
                    self._session_async = session
                    self._session_ready_event_async.set()
                    self._logger_instance.info(f"{self.name} session ready. Waiting for shutdown signal...")
                    await self._shutdown_event_async.wait()
                    self._logger_instance.info(f"{self.name} shutdown signal received.")
        except asyncio.TimeoutError:
            self._logger_instance.error(f"Timeout ({self.session_init_timeout}s) during {self.name} session initialization.")
        except asyncio.CancelledError:
            self._logger_instance.info(f"{self.name} session lifecycle task cancelled.")
        except MCPBraveSearchToolError as e:
             self._logger_instance.error(f"Lifecycle error (setup): {e}")
        except Exception as e:
            self._logger_instance.error(f"Error in {self.name} session lifecycle: {e}", exc_info=True)
        finally:
            self._logger_instance.info(f"{self.name} session lifecycle task finished.")
            self._session_async = None
            if self._session_ready_event_async and not self._session_ready_event_async.is_set():
                self._session_ready_event_async.set() # Unblock waiters

    async def _ensure_session_ready(self):
        """Ensures the MCP session is initialized and ready."""
        # Identical readiness check logic as other async tools
        if self._is_closed_async: raise MCPBraveSearchToolError(f"{self.name} is closed.")
        await self._initialize_async_primitives()
        if self._session_async and self._session_ready_event_async.is_set(): return

        async with self._init_lock_async:
            if self._session_async and self._session_ready_event_async.is_set(): return
            if self._is_closed_async: raise MCPBraveSearchToolError(f"{self.name} closed during readiness check.")

            if self._lifecycle_task_async is None or self._lifecycle_task_async.done():
                self._logger_instance.info(f"Starting new {self.name} session lifecycle task.")
                self._session_ready_event_async.clear()
                self._shutdown_event_async.clear()
                self._lifecycle_task_async = asyncio.create_task(self._manage_session_lifecycle())
            
            try:
                await asyncio.wait_for(self._session_ready_event_async.wait(), timeout=self.session_init_timeout + 10.0)
            except asyncio.TimeoutError:
                self._logger_instance.error(f"Timeout waiting for {self.name} session.")
                if self._lifecycle_task_async and not self._lifecycle_task_async.done(): self._lifecycle_task_async.cancel()
                raise MCPBraveSearchToolError(f"Timeout establishing {self.name} MCP session.")

            if not self._session_async or not self._session_ready_event_async.is_set():
                raise MCPBraveSearchToolError(f"Failed to establish a valid {self.name} MCP session.")
            self._logger_instance.info(f"{self.name} MCP session is ready.")

    async def _arun(
        self, 
        search_type: Literal["web", "news", "image", "video", "summarize"], 
        query: str, 
        count: Optional[int] = 5, 
        offset: Optional[int] = 0, 
        country: Optional[str] = None, 
        search_lang: Optional[str] = None, 
        safesearch: Optional[Literal["off"]] = "off",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        if self._is_closed_async:
            return json.dumps({"error": f"{self.name} is closed."})

        self._logger_instance.info(f"Executing Brave search: type='{search_type}', query='{query[:50]}...'")
        
        try:
            await self._ensure_session_ready()
            if not self._session_async:
                return json.dumps({"error": f"{self.name} session not available."})

            # Map search_type to the corresponding MCP tool name on the server
            tool_name_map = {
                "web": "brave_web_search",
                "news": "brave_news_search",
                "image": "brave_image_search",
                "video": "brave_video_search",
                "summarize": "brave_summarize"
            }
            server_tool_name = tool_name_map.get(search_type)
            
            if not server_tool_name:
                # Should be caught by Pydantic Literal, but good practice
                return json.dumps({"error": f"Invalid search_type '{search_type}'. Valid types: {list(tool_name_map.keys())}"})

            # Prepare the arguments dictionary expected by the specific server tool
            # The server tools expect parameters matching their input models (e.g., WebSearchInput)
            server_tool_args = {
                "q": query, # Server uses 'q'
                "count": count,
                "offset": offset,
                "country": country,
                "search_lang": search_lang,
                "safesearch": safesearch
                # Add other specific params based on search_type if needed, e.g., 'freshness' for video
            }
            # Clean None values, as server models might not handle them gracefully
            if server_tool_name == "brave_summarize":
                # Only pass relevant non-None params for summarize
                server_tool_args = {k: v for k, v in server_tool_args.items() if v is not None and k in ["q", "country", "search_lang"]}
            else:
                # For other search types, remove all None values before sending
                server_tool_args = {k: v for k, v in server_tool_args.items() if v is not None}

            self._logger_instance.debug(f"Calling MCP server tool '{server_tool_name}' with arguments: {server_tool_args}")

            # Call the specific MCP server tool
            # The server expects arguments to be wrapped in "input_params"
            try:
                response: CallToolResult = await asyncio.wait_for(
                    self._session_async.call_tool(
                        name=server_tool_name,
                        arguments={"input_params": server_tool_args}
                    ),
                    timeout=self.tool_call_timeout
                )
            except asyncio.TimeoutError:
                self._logger_instance.error(f"Timeout calling MCP server tool '{server_tool_name}' after {self.tool_call_timeout}s")
                return json.dumps({"error": f"Tool call to '{server_tool_name}' timed out."})

            self._logger_instance.debug(f"Brave Search Response: isError={response.isError}, Content type: {type(response.content)}")

            if response.isError:
                error_message = f"Server error during '{search_type}' search for '{query[:50]}...'."
                if response.content and isinstance(response.content, list) and len(response.content) > 0:
                    item = response.content[0]
                    if isinstance(item, MCPErrorData) and item.message: error_message = item.message
                    elif isinstance(item, TextContent) and item.text: error_message = f"Server error: {item.text}"
                    else: error_message += f" Raw error: {str(item)[:100]}"
                self._logger_instance.error(error_message)
                return json.dumps({"error": error_message})

            if response.content and isinstance(response.content, list) and len(response.content) > 0:
                first_content_item = response.content[0]
                if isinstance(first_content_item, TextContent) and first_content_item.text is not None:
                    # Assume server sends results as JSON string in TextContent
                    # Validate and return
                    try:
                        json.loads(first_content_item.text) # Validate
                        return first_content_item.text
                    except json.JSONDecodeError:
                         self._logger_instance.error(f"Received non-JSON text response from server for '{search_type}'. Response: {first_content_item.text[:200]}")
                         return json.dumps({"error": "Received non-JSON response from server.", "raw_response": first_content_item.text})
                else:
                     self._logger_instance.warning(f"Unexpected successful content type from Brave Search server: {type(first_content_item).__name__}.")
                     return json.dumps({"error": "Received unexpected content type from server.", "raw_content": str(first_content_item)})
            else:
                self._logger_instance.warning(f"Brave Search successful but no content items returned for '{search_type}' query '{query[:50]}...'.")
                # Return empty results structure appropriate for the search type
                if search_type == "summarize":
                    return json.dumps({"summary_title": None, "summary_text": "No summary available."})
                else: # For list-based results (web, news, etc.)
                    return json.dumps({"results": []})

        except asyncio.TimeoutError:
            self._logger_instance.error(f"Timeout executing Brave Search for '{query[:50]}...'.")
            return json.dumps({"error": f"Timeout during Brave Search query."})
        except MCPBraveSearchToolError as e: # From _ensure_session_ready
            self._logger_instance.error(f"MCPBraveSearchToolError: {e}")
            return json.dumps({"error": str(e)})
        except ValueError as e: # Catch errors from API calls in server potentially
             self._logger_instance.error(f"ValueError during Brave Search '{search_type}': {e}")
             return json.dumps({"error": f"Search API error: {str(e)}"})
        except ConnectionError as e: # Catch errors from API calls in server potentially
             self._logger_instance.error(f"ConnectionError during Brave Search '{search_type}': {e}")
             return json.dumps({"error": f"Search API connection error: {str(e)}"})
        except Exception as e:
            self._logger_instance.error(f"Unexpected error during Brave Search '{search_type}': {e}", exc_info=True)
            return json.dumps({"error": f"Unexpected error during search: {str(e)}"})

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
        # Maintain consistency - no sync bridge for this purely async tool
        raise NotImplementedError(f"{self.name} is async-native. Use _arun.")

    def __del__(self):
         if not self._is_closed_async and hasattr(self, '_logger_instance') and self._logger_instance:
              self._logger_instance.warning(f"{self.name} instance deleted without explicit close.")

# --- Example Usage ---
async def main():
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(level=log_level)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    test_logger = logging.getLogger(__name__ + ".BraveSearch_Test")
    test_logger.info("Starting MCPBraveSearchTool async example usage...")

    if not (BRAVE_SEARCH_API_KEY or os.getenv("BRAVE_SEARCH_API_KEY")):
        test_logger.critical("CRITICAL: BRAVE_SEARCH_API_KEY not found. Please set it in src/config.py or environment.")
        return

    tool = MCPBraveSearchTool()
    tool._logger_instance.setLevel(logging.DEBUG) # More verbose tool logging for test

    try:
        print("\n--- Testing Web Search ---")
        result = await tool._arun(query="What is the Model Context Protocol?", search_type="web", count=2)
        print(result)

        print("\n--- Testing News Search ---")
        result = await tool._arun(query="Latest AI Safety News", search_type="news", count=3)
        print(result)

        print("\n--- Testing Summarizer ---")
        result = await tool._arun(query="Explain LangChain Agents", search_type="summarize")
        print(result)

        print("\n--- Testing Image Search ---")
        result = await tool._arun(
            search_type="image",
            query="Golden Retriever",
            count=4, 
            offset=0,
            safesearch="off"
        )
        print(result)
        
        print("\n--- Testing Video Search ---")
        result = await tool._arun(
            search_type="video",
            query="Asyncio tutorial", 
            count=3, 
            offset=0, 
            safesearch="off"
        )
        print(result)

        print("\n--- Testing Invalid Type ---")
        # This should be caught by Pydantic now if the call uses the model,
        # but testing the _arun internal handling too.
        try:
             # Simulate invalid type bypassing Pydantic validation for test
             result = await tool._arun(query="test", search_type="podcast") 
             print(result)
        except Exception as e:
            print(f"Caught expected error for invalid type: {e}")


    except Exception as e:
        print(f"\nAn error occurred during testing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        print("\nClosing tool...")
        await tool.close()
        print("Tool closed.")

if __name__ == "__main__":
    asyncio.run(main())