# src/tools/mcp_fetcher_web_tool.py

import asyncio
import json
import logging
import os
import shutil # For shutil.which
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Type, List, Literal, Union
import re

from langchain_core.tools import BaseTool, ToolException
from langchain_core.callbacks.manager import AsyncCallbackManagerForToolRun
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
    LoggingMessageNotificationParams,
    LATEST_PROTOCOL_VERSION # Or a specific version string like "0.2.0"
)

# Module logger
logger = logging.getLogger(__name__)

class MCPFetcherWebToolError(ToolException):
    """Custom exception for the MCPFetcherWebTool."""
    pass

# --- Pydantic Schemas for Server Tool Parameters ---
class FetchURLBaseParams(BaseModel):
    """Base parameters common to fetch_url and applicable to fetch_urls."""
    timeout: Optional[int] = Field(
        default=30000,
        description="Page loading timeout in milliseconds (e.g., 30000 for 30s)."
    )
    waitUntil: Optional[Literal['load', 'domcontentloaded', 'networkidle', 'commit']] = Field(
        default='load',
        description="When navigation is complete. 'networkidle' for SPAs."
    )
    maxLength: Optional[int] = Field(
        default=None, # None means no limit server-side
        description="Maximum length of returned content in characters."
    )
    waitForNavigation: Optional[bool] = Field(
        default=False,
        description="Set to true for pages with anti-bot checks or client-side redirects."
    )
    navigationTimeout: Optional[int] = Field(
        default=10000,
        description="Max time for 'waitForNavigation' if true."
    )
    extractContent: bool = Field(
        default=True,
        description="True to extract main content (default, Markdown), False for full page."
    )
    returnHtml: bool = Field(
        default=False,
        description="If extractContent is False, set to True to get HTML. If extractContent is True, set to False to get Markdown (default) or True for HTML of extracted part."
    )

class InnerFetchSingleURLAction(FetchURLBaseParams):
    action: Literal["fetch_single_url"] = Field(default="fetch_single_url", frozen=True)
    url: str = Field(description="The single, fully qualified URL to fetch.")

class InnerFetchMultipleURLsAction(FetchURLBaseParams):
    action: Literal["fetch_multiple_urls"] = Field(default="fetch_multiple_urls", frozen=True)
    urls: List[str] = Field(description="A list of fully qualified URLs to fetch in batch.")

# --- Main LangChain Tool Input Schema ---
# This uses a discriminated union based on the 'action' field,
# ensuring that parameters are validated correctly for each sub-action.
# This is the structure of the `action_input` for MCPWebFetcher.
class MCPFetcherWebActionInput(RootModel[Union[InnerFetchSingleURLAction, InnerFetchMultipleURLsAction]]):
    root: Union[InnerFetchSingleURLAction, InnerFetchMultipleURLsAction] = Field(..., discriminator='action')

    # Pydantic v2 does not automatically call dict() on RootModel instances when passing to _arun
    # So, we'll pass the whole model and unpack in _arun, or LangChain does this.
    # The goal is that _arun receives keyword arguments matching the fields of the chosen Union member.


# Main Tool Class (Async Version)
class MCPFetcherWebTool(BaseTool, BaseModel):
    name: str = "MCPWebFetcher" # This is the LangChain tool name
    description: str = (
        "Fetches content from one or more web URLs using the 'fetcher-mcp' server (which uses a headless browser via Playwright). "
        "The `action_input` for this tool MUST be a JSON object containing an 'action' field specifying the operation, and its corresponding parameters.\n\n"
        "Available 'action' types for `action_input`:\n"
        "1. \"fetch_single_url\": Fetches content from a single URL.\n"
        "   - Required parameter: 'url' (string): The URL to fetch.\n"
        "   - Optional parameters (all apply from `fetcher-mcp`'s `fetch_url` server tool):\n"
        "     - 'extractContent' (boolean, default: true): True to extract main content (as Markdown by default), False for the full page content.\n"
        "     - 'returnHtml' (boolean, default: false): If 'extractContent' is false, set true to get HTML. If 'extractContent' is true, set false for Markdown (default) or true for HTML of the *extracted* part.\n"
        "     - 'timeout' (integer, ms, default: 30000): Page loading timeout.\n"
        "     - 'waitUntil' (string, default: 'load'): When navigation is considered complete. Options: 'load', 'domcontentloaded', 'networkidle', 'commit'. 'networkidle' is useful for SPAs but can be slower.\n"
        "     - 'maxLength' (integer, default: no limit): Maximum characters for the returned content of the page.\n"
        "     - 'waitForNavigation' (boolean, default: false): Set to true for pages with anti-bot measures, client-side redirects, or that need a delay for content to appear.\n"
        "     - 'navigationTimeout' (integer, ms, default: 10000): Max time to wait if 'waitForNavigation' is true.\n\n"
        "2. \"fetch_multiple_urls\": Fetches content from a list of URLs in batch using `fetcher-mcp`'s `fetch_urls` server tool.\n"
        "   - Required parameter: 'urls' (list of strings): The URLs to fetch.\n"
        "   - Optional parameters: Same as for 'fetch_single_url' (e.g., 'extractContent', 'returnHtml', 'timeout', etc.). These settings will apply to *all* URLs in the batch.\n\n"
        "TIPS FOR USE:\n"
        "- To get structured Markdown of an article: `action_input`: `{\"action\": \"fetch_single_url\", \"url\": \"...\", \"extractContent\": true, \"returnHtml\": false}` (these are defaults for extract/returnHtml with fetch_single_url if not specified).\n"
        "- To get the full raw HTML of a page: `action_input`: `{\"action\": \"fetch_single_url\", \"url\": \"...\", \"extractContent\": false, \"returnHtml\": true}`.\n"
        "- If a page is complex (JavaScript-heavy, anti-bot checks like 'Please wait...'), try using `\"waitForNavigation\": true`. You might also need to adjust `\"timeout\"` or `\"navigationTimeout\"`.\n\n"
        "Example for fetching structured Markdown from a single URL:\n"
        "Action:\n"
        "```json\n"
        "{\n"
        "  \"action\": \"MCPWebFetcher\",\n"
        "  \"action_input\": {\n"
        "    \"action\": \"fetch_single_url\",\n"
        "    \"url\": \"https://example.com/article\"\n"
        "  }\n"
        "}\n"
        "```\n\n"
        "Example for fetching full HTML from multiple URLs with a longer timeout:\n"
        "Action:\n"
        "```json\n"
        "{\n"
        "  \"action\": \"MCPWebFetcher\",\n"
        "  \"action_input\": {\n"
        "    \"action\": \"fetch_multiple_urls\",\n"
        "    \"urls\": [\"https://site1.com\", \"https://site2.com/page\"],\n"
        "    \"extractContent\": false,\n"
        "    \"returnHtml\": true,\n"
        "    \"timeout\": 45000\n"
        "  }\n"
        "}\n"
        "```"
    )
    args_schema: Type[BaseModel] = MCPFetcherWebActionInput # Use the new main input schema
    return_direct: bool = False
    handle_tool_error: bool = True

    # Configuration for the MCP server process
    npx_executable_path: Path = Field(
        default_factory=lambda: Path(shutil.which("npx") or "npx")
    )
    server_package_name: str = Field(default="fetcher-mcp")
    server_process_cwd_str: Optional[str] = Field(default=None, description="Optional CWD for the npx process. Defaults to Alpy's current working directory.")

    session_init_timeout: float = 45.0
    tool_call_timeout: float = 120.0

    # Internal Async State
    _session_async: Optional[ClientSession] = PrivateAttr(default=None)
    _lifecycle_task_async: Optional[asyncio.Task] = PrivateAttr(default=None)
    _session_ready_event_async: Optional[asyncio.Event] = PrivateAttr(default=None)
    _shutdown_event_async: Optional[asyncio.Event] = PrivateAttr(default=None)
    _init_lock_async: Optional[asyncio.Lock] = PrivateAttr(default=None)
    _is_closed_async: bool = PrivateAttr(default=False)
    _logger_instance: logging.Logger = PrivateAttr(default=None)
    _server_process_cwd_resolved: Optional[Path] = PrivateAttr(default=None)

    _client_info: MCPImplementation = PrivateAttr(default_factory=lambda: MCPImplementation(name="AlpyAsyncFetcherClient", version="1.1.0"))

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def _tool_post_init_validator(self) -> 'MCPFetcherWebTool':
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
        return self

    async def _initialize_async_primitives(self):
        if self._init_lock_async is None: self._init_lock_async = asyncio.Lock()
        if self._session_ready_event_async is None: self._session_ready_event_async = asyncio.Event()
        if self._shutdown_event_async is None: self._shutdown_event_async = asyncio.Event()

    def _get_server_params(self) -> StdioServerParameters:
        if not self.npx_executable_path or not self.npx_executable_path.exists():
             raise MCPFetcherWebToolError(f"NPX executable not found: {self.npx_executable_path}")
        if not self._server_process_cwd_resolved or not self._server_process_cwd_resolved.is_dir():
             raise MCPFetcherWebToolError(f"Server CWD for NPX is invalid: {self._server_process_cwd_resolved}")
        env = os.environ.copy()
        return StdioServerParameters(
            command=str(self.npx_executable_path.resolve()),
            args=["-y", self.server_package_name],
            cwd=str(self._server_process_cwd_resolved),
            env=env
        )

    async def _mcp_server_log_callback(self, params: LoggingMessageNotificationParams) -> None:
        level = getattr(logging, params.level.upper(), logging.INFO)
        self._logger_instance.log(level, f"[MCP Server ({self.server_package_name}) - {params.scope}]: {params.message}")

    async def _manage_session_lifecycle(self):
        self._logger_instance.info(f"Starting {self.name} (via npx {self.server_package_name}) MCP session lifecycle...")
        try:
            server_params = self._get_server_params()
            fetcher_log_path = Path(os.getcwd()) / "fetcher_mcp_npx_server.log"
            self._logger_instance.info(f"MCP Server ({self.server_package_name} via npx) stderr will be logged to: {fetcher_log_path}")
            with open(fetcher_log_path, 'a', encoding='utf-8') as ferr:
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
        except asyncio.TimeoutError:
            self._logger_instance.error(f"Timeout ({self.session_init_timeout}s) during {self.name} session init.")
        except asyncio.CancelledError:
            self._logger_instance.info(f"{self.name} session lifecycle task cancelled.")
        except MCPFetcherWebToolError as e:
             self._logger_instance.error(f"Lifecycle error (setup for npx): {e}")
        except Exception as e:
            self._logger_instance.error(f"Error in {self.name} session lifecycle: {e}", exc_info=True)
        finally:
            self._logger_instance.info(f"{self.name} session lifecycle task finished.")
            self._session_async = None
            if self._session_ready_event_async and not self._session_ready_event_async.is_set():
                self._session_ready_event_async.set()

    async def _ensure_session_ready(self):
        if self._is_closed_async: raise MCPFetcherWebToolError(f"{self.name} is closed.")
        await self._initialize_async_primitives()
        if self._session_async and self._session_ready_event_async.is_set(): return
        async with self._init_lock_async:
            if self._session_async and self._session_ready_event_async.is_set(): return
            if self._is_closed_async: raise MCPFetcherWebToolError(f"{self.name} closed during readiness check.")
            if self._lifecycle_task_async is None or self._lifecycle_task_async.done():
                self._logger_instance.info(f"Starting new {self.name} session lifecycle task.")
                self._session_ready_event_async.clear()
                self._shutdown_event_async.clear()
                self._lifecycle_task_async = asyncio.create_task(self._manage_session_lifecycle())
            else: self._logger_instance.info(f"Waiting for existing {self.name} session setup.")
            try:
                await asyncio.wait_for(self._session_ready_event_async.wait(), timeout=self.session_init_timeout + 10.0)
            except asyncio.TimeoutError:
                self._logger_instance.error(f"Timeout waiting for {self.name} session ready.")
                if self._lifecycle_task_async and not self._lifecycle_task_async.done(): self._lifecycle_task_async.cancel()
                raise MCPFetcherWebToolError(f"Timeout establishing {self.name} MCP session.")
            if not self._session_async or not self._session_ready_event_async.is_set():
                raise MCPFetcherWebToolError(f"Failed to establish valid {self.name} MCP session.")
            self._logger_instance.info(f"{self.name} MCP session is ready.")

    def _parse_fetcher_mcp_output(self, raw_output: str, primary_url_for_context: str) -> Dict[str, Any]:
        """Parses the 'Title:\nURL:\nContent:\n\n...' format from fetcher-mcp."""
        # Initialize with the URL we *intended* to fetch, server might override if redirect
        parsed_data = {"title": None, "url": primary_url_for_context, "content": None, "error": None}
        try:
            lines = raw_output.split('\n')
            
            # Find Title:
            title_line_index = -1
            for i, line in enumerate(lines):
                if line.startswith("Title: "):
                    parsed_data["title"] = line[len("Title: "):].strip()
                    title_line_index = i
                    break
            
            # Find URL: (usually after Title)
            url_line_index = -1
            start_search_for_url = title_line_index + 1 if title_line_index != -1 else 0
            for i in range(start_search_for_url, len(lines)):
                line = lines[i]
                if line.startswith("URL: "):
                    parsed_data["url"] = line[len("URL: "):].strip() # Server's reported URL
                    url_line_index = i
                    break
            
            content_marker = "Content:\n\n"
            content_index = raw_output.find(content_marker)
            if content_index != -1:
                actual_content = raw_output[content_index + len(content_marker):].strip()
                # Remove the "[webpage X end]" marker if it's part of the captured content for an individual block
                end_marker_match = re.search(r"\[webpage \d+ end\]$", actual_content, re.MULTILINE)
                if end_marker_match:
                    actual_content = actual_content[:end_marker_match.start()].strip()

                if actual_content.startswith("<error>") and actual_content.endswith("</error>"):
                    parsed_data["error"] = actual_content[len("<error>"):-len("</error>")].strip()
                    parsed_data["content"] = None
                    # Ensure URL used for logging matches the one from the server output if available
                    log_url_context = parsed_data["url"] if parsed_data["url"] != primary_url_for_context else primary_url_for_context
                    self._logger_instance.warning(f"Fetcher-mcp reported an error for URL '{log_url_context}': {parsed_data['error']}")
                else:
                    parsed_data["content"] = actual_content
            else:
                # If no "Content:" marker, the whole thing might be an error or malformed
                # Check if the raw_output itself looks like an error block
                if raw_output.strip().startswith("<error>") and raw_output.strip().endswith("</error>"):
                    parsed_data["error"] = raw_output.strip()[len("<error>"):-len("</error>")].strip()
                    self._logger_instance.warning(f"Fetcher-mcp reported a global error (no Content marker): {parsed_data['error']}")
                else:
                    # Could not find standard "Content:" section
                    parsed_data["content"] = raw_output # Or treat as error
                    self._logger_instance.warning(f"Could not parse 'Content:' section for URL '{primary_url_for_context}'. Using raw output. Raw: {raw_output[:100]}")

        except Exception as e_parse:
            self._logger_instance.error(f"Error parsing fetcher-mcp output for URL '{primary_url_for_context}': {e_parse}. Raw: {raw_output[:200]}", exc_info=True)
            parsed_data["error"] = "Failed to parse server's custom text output."
            parsed_data["content"] = raw_output # Keep raw output on parsing failure
        
        # If an error was explicitly parsed, ensure content is None
        if parsed_data["error"] and parsed_data["content"] == raw_output : # Avoid clearing if content was parsed before error
             if not (raw_output.startswith("<error>") and raw_output.endswith("</error>")): # if raw IS the error, content should be none
                  pass # keep raw_output as content if it's not an explicit error block
             else:
                  parsed_data["content"] = None


        return parsed_data

    async def _arun(
        self,
        action: Literal["fetch_single_url", "fetch_multiple_urls"],
        url: Optional[str] = None,
        urls: Optional[List[str]] = None,
        timeout: Optional[int] = None,
        waitUntil: Optional[Literal['load', 'domcontentloaded', 'networkidle', 'commit']] = None,
        maxLength: Optional[int] = None,
        waitForNavigation: Optional[bool] = None,
        navigationTimeout: Optional[int] = None,
        extractContent: Optional[bool] = None,
        returnHtml: Optional[bool] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        if self._is_closed_async:
            return json.dumps({"error": f"{self.name} is closed."})

        self._logger_instance.info(f"Executing MCPWebFetcher: inner_action='{action}', url='{url}', urls='{urls}'")

        try:
            await self._ensure_session_ready()
            if not self._session_async:
                return json.dumps({"error": f"{self.name} session not available."})

            server_tool_args = {}
            base_params_defaults = FetchURLBaseParams()
            primary_url_for_parsing_context = "N/A" # For logging/error context

            if action == "fetch_single_url":
                server_tool_name = "fetch_url"
                if not url: return json.dumps({"error": "'url' is required for 'fetch_single_url'."})
                server_tool_args["url"] = url
                primary_url_for_parsing_context = url
            elif action == "fetch_multiple_urls":
                server_tool_name = "fetch_urls"
                if not urls: return json.dumps({"error": "'urls' (list) is required for 'fetch_multiple_urls'."})
                server_tool_args["urls"] = urls
                primary_url_for_parsing_context = urls[0] if urls else "batch_fetch"
            else:
                return json.dumps({"error": f"Invalid inner action '{action}' specified."})

            current_extract_content = extractContent if extractContent is not None else base_params_defaults.extractContent
            current_return_html = returnHtml if returnHtml is not None else base_params_defaults.returnHtml

            server_tool_args.update({
                "extractContent": current_extract_content,
                "returnHtml": current_return_html,
                "timeout": timeout if timeout is not None else base_params_defaults.timeout,
                "waitUntil": waitUntil if waitUntil is not None else base_params_defaults.waitUntil,
                "maxLength": maxLength,
                "waitForNavigation": waitForNavigation if waitForNavigation is not None else base_params_defaults.waitForNavigation,
                "navigationTimeout": navigationTimeout if navigationTimeout is not None else base_params_defaults.navigationTimeout,
            })
            
            final_server_tool_args = {k: v for k, v in server_tool_args.items() if v is not None}
            self._logger_instance.debug(f"Calling MCP server tool '{server_tool_name}' with arguments: {final_server_tool_args}")

            try:
                response: CallToolResult = await asyncio.wait_for(
                    self._session_async.call_tool(name=server_tool_name, arguments=final_server_tool_args),
                    timeout=self.tool_call_timeout
                )
            except asyncio.TimeoutError:
                self._logger_instance.error(f"Timeout calling MCP server tool '{server_tool_name}'.")
                return json.dumps({"error": f"Tool call to '{server_tool_name}' timed out."})

            self._logger_instance.debug(f"Fetcher MCP Response: isError={response.isError}, Content items: {len(response.content) if response.content else 0}")

            if response.isError:
                err_msg = response.message or f"MCP protocol error for {server_tool_name}."
                self._logger_instance.error(err_msg)
                return json.dumps({"error": err_msg, "source": "MCP_PROTOCOL_ERROR", "url_context": primary_url_for_parsing_context})

            if response.content and isinstance(response.content, list) and len(response.content) > 0:
                first_content_item = response.content[0]
                if isinstance(first_content_item, TextContent) and first_content_item.text is not None:
                    raw_server_output = first_content_item.text
                    self._logger_instance.debug(f"Raw output from fetcher-mcp: {raw_server_output[:300]}...")
                    
                    if action == "fetch_multiple_urls":
                        results_list = []
                        # Split the combined output by "[webpage X end]" and then process each chunk.
                        # A more robust regex might be needed if there are variations.
                        # The pattern seems to be "[webpage X begin]\n...\n[webpage X end]\n\n"
                        # Let's try splitting by a common end marker and then processing.
                        # A simple split for now, assuming structure is consistent.
                        
                        # Define start and end markers
                        webpage_blocks = []
                        # Use regex to find all blocks: r"\[webpage \d+ begin\]([\s\S]*?)\[webpage \d+ end\]"
                        # For simplicity with split, assuming consistent newlines after "end]"
                        
                        # A more robust splitting strategy for the batch output:
                        # The server output has "[webpage X begin]\n...\n[webpage X end]\n\n"
                        # We can split by "[webpage " then process each part that starts with a number.
                        
                        # Simpler split: Assume each result is separated by two newlines after "[webpage X end]"
                        # and starts with "[webpage X begin]"
                        
                        # Let's use a regex to capture each block
                        # Pattern: Match "[webpage X begin]", then everything until "[webpage X end]"
                        block_pattern = re.compile(r"(\[webpage \d+ begin\][\s\S]*?\[webpage \d+ end\])", re.DOTALL)
                        individual_page_outputs = block_pattern.findall(raw_server_output)

                        if not individual_page_outputs and raw_server_output.strip(): # If regex fails, maybe it's a single error for the whole batch
                            if raw_server_output.startswith("Title: Error") or "<error>" in raw_server_output : # It might be a global error for the batch
                                parsed_error = self._parse_fetcher_mcp_output(raw_server_output, "batch_error")
                                return json.dumps({"batch_results": [parsed_error]}) # Return the error as the only item


                        for i, page_output_chunk in enumerate(individual_page_outputs):
                            # The URL for this specific chunk isn't directly in the chunk if it was an error,
                            # but we know the input URLs. We can try to match or just use index.
                            current_url_context = urls[i] if urls and i < len(urls) else f"batch_item_{i+1}"
                            parsed_item = self._parse_fetcher_mcp_output(page_output_chunk, current_url_context)
                            results_list.append(parsed_item)
                        
                        if not results_list and raw_server_output.strip(): # No blocks found, but there's output
                            self._logger_instance.warning("Batch fetch for 'fetch_multiple_urls' returned non-empty output but no standard blocks were parsed. Returning raw.")
                            return json.dumps({
                                "batch_operation_status": "raw_output_returned_no_blocks_parsed",
                                "message": "Batch fetch output received, but standard page blocks were not identified. Raw output provided.",
                                "combined_raw_content": raw_server_output
                            })
                        
                        return json.dumps({"batch_results": results_list})
                    
                    else: # fetch_single_url
                        parsed_json_output = self._parse_fetcher_mcp_output(raw_server_output, primary_url_for_parsing_context)
                        return json.dumps(parsed_json_output)
                else:
                     self._logger_instance.warning(f"Unexpected content type: {type(first_content_item).__name__} for URL {primary_url_for_parsing_context}.")
                     return json.dumps({"error": "Unexpected content type from server.", "raw_content_type": type(first_content_item).__name__, "url_context": primary_url_for_parsing_context})
            else:
                self._logger_instance.warning(f"No content items returned for '{action}' on URL {primary_url_for_parsing_context}.")
                return json.dumps({"url": primary_url_for_parsing_context, "error": "Server returned success but no content.", "title": None, "content": None})

        except asyncio.TimeoutError:
            self._logger_instance.error(f"Overall timeout executing Web Fetch for {primary_url_for_parsing_context}.")
            return json.dumps({"error": f"Timeout during Web Fetch query.", "url_context": primary_url_for_parsing_context})
        except MCPFetcherWebToolError as e:
            self._logger_instance.error(f"MCPFetcherWebToolError for {primary_url_for_parsing_context}: {e}")
            return json.dumps({"error": str(e), "url_context": primary_url_for_parsing_context})
        except Exception as e:
            self._logger_instance.error(f"Unexpected error during Web Fetch '{action}' for {primary_url_for_parsing_context}: {e}", exc_info=True)
            return json.dumps({"error": f"Unexpected error during fetch: {str(e)}", "url_context": primary_url_for_parsing_context})


    async def close(self):
        if self._is_closed_async and (self._lifecycle_task_async is None or self._lifecycle_task_async.done()): return
        self._logger_instance.info(f"Closing {self.name} (npx {self.server_package_name})...")
        self._is_closed_async = True
        await self._initialize_async_primitives()
        if self._shutdown_event_async: self._shutdown_event_async.set()
        if self._lifecycle_task_async and not self._lifecycle_task_async.done():
            self._logger_instance.info(f"Waiting for {self.name} lifecycle task to complete shutdown...")
            try: await asyncio.wait_for(self._lifecycle_task_async, timeout=10.0)
            except asyncio.TimeoutError:
                self._logger_instance.warning(f"Timeout waiting for {self.name} lifecycle task. Cancelling.")
                self._lifecycle_task_async.cancel()
                try: await self._lifecycle_task_async
                except asyncio.CancelledError: self._logger_instance.info(f"{self.name} lifecycle task cancelled.")
                except Exception as e_cancel: self._logger_instance.error(f"Error awaiting cancelled task: {e_cancel}")
            except Exception as e_wait: self._logger_instance.error(f"Error waiting for task: {e_wait}")
        if self._session_ready_event_async: self._session_ready_event_async.clear()
        if self._shutdown_event_async: self._shutdown_event_async.clear()
        self._session_async = None
        self._lifecycle_task_async = None
        self._logger_instance.info(f"{self.name} (npx {self.server_package_name}) closed.")

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError(f"{self.name} is async-native. Use its `_arun` method.")

    def __del__(self):
         if not self._is_closed_async and hasattr(self, '_logger_instance') and self._logger_instance:
              self._logger_instance.warning(f"{self.name} (ID: {id(self)}) deleted without explicit close.")

# --- Example Usage (for standalone testing of this tool) ---
async def main_test_mcp_fetcher_web_tool_final():
    log_level_str = os.getenv("LOG_LEVEL_TOOL_TEST", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s', stream=sys.stdout)
    if log_level > logging.DEBUG: logging.getLogger("asyncio").setLevel(logging.WARNING)
    test_logger = logging.getLogger(__name__ + ".MCPFetcherWebTool_FinalTest")
    test_logger.info("Starting MCPFetcherWebTool standalone async test (final version)...")
    tool: Optional[MCPFetcherWebTool] = None
    try:
        tool = MCPFetcherWebTool()
        if tool._logger_instance: tool._logger_instance.setLevel(logging.DEBUG)

        print("\n--- [Test Case 1: Fetch Single URL - Structured Data (Default extract/returnHtml)] ---")
        # LLM would generate action_input like: {"action": "fetch_single_url", "url": "..."}
        # LangChain would pass these as kwargs to _arun.
        result_single_structured = await tool._arun(action="fetch_single_url", url="https://www.mozilla.org/en-US/about/manifesto/")
        print(f"Result (Single Structured): {result_single_structured[:700]}...")
        json.loads(result_single_structured) # Validate JSON

        print("\n--- [Test Case 2: Fetch Single URL - Raw HTML] ---")
        # LLM: {"action": "fetch_single_url", "url": "...", "extractContent": false, "returnHtml": true}
        result_single_html = await tool._arun(action="fetch_single_url", url="http://example.com", extractContent=False, returnHtml=True)
        print(f"Result (Single HTML): {result_single_html[:500]}...")
        json.loads(result_single_html)

        print("\n--- [Test Case 3: Fetch Multiple URLs - Default (Structured, Markdown)] ---")
        # LLM: {"action": "fetch_multiple_urls", "urls": ["http://example.com", "https://www.iana.org/domains/example"]}
        # Note: The parsing for multiple URLs in _arun is currently simplified.
        result_multi_structured = await tool._arun(
            action="fetch_multiple_urls",
            urls=["http://example.com", "https://www.iana.org/domains/reserved"] # iana.org/domains/example is too simple
        )
        print(f"Result (Multi Structured - raw combined): {result_multi_structured[:1000]}...")
        json.loads(result_multi_structured)


        print("\n--- [Test Case 4: Fetch Single URL - Non-existent domain] ---")
        # LLM: {"action": "fetch_single_url", "url": "http://thissitedoesnotexistpleasedont.com"}
        result_error_domain = await tool._arun(action="fetch_single_url", url="http://thissitedoesnotexistpleasedont.com")
        print(f"Result (Error Domain): {result_error_domain}")
        json.loads(result_error_domain)


    except Exception as e:
        test_logger.error(f"An error occurred during the MCPFetcherWebTool test: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
    finally:
        if tool:
            test_logger.info("Closing MCPFetcherWebTool...")
            await tool.close()
            test_logger.info("MCPFetcherWebTool closed.")
        test_logger.info("MCPFetcherWebTool standalone async test finished.")

if __name__ == "__main__":
    asyncio.run(main_test_mcp_fetcher_web_tool_final())