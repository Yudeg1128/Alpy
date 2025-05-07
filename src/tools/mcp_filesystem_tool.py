import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Type, Literal, Optional, Dict, List
import tempfile
import shutil # For shutil.which

from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field, PrivateAttr, model_validator # Pydantic v2 imports
import mcp.client.stdio # For stdio_client context manager
from mcp.client.stdio import StdioServerParameters # Specific import
from mcp.client.session import ClientSession      # Specific import
import mcp.types # Keep this for mcp.types.TextContent etc.
from mcp.types import (
    CallToolResult,
    ErrorData as MCPErrorData,
    InitializeResult,
    LATEST_PROTOCOL_VERSION,
    Implementation as MCPImplementation,
    LoggingMessageNotificationParams,
    TextContent # Main content type for JSON strings from Node.js server
)

# Use the module-level logger as a base
logger = logging.getLogger(__name__) # This can be used by the __main__ test block

class McpFileSystemToolError(ToolException):
    """Custom exception for MCP FileSystem Tool errors."""
    pass

class MCPFileSystemToolInput(BaseModel):
    action: Literal[
        "list_dir", "read_file", "stat_file", "create_dir", "write_file",
        "delete_file", "delete_dir", "read_multiple_files", "edit_file",
        "directory_tree", "move_file", "search_files", "list_allowed_directories"
    ] = Field(description="Action to perform.")
    uri: Optional[str] = Field(default=None, description="Primary URI for the operation (e.g., 'file:///path/to/item'). Not used by all actions.")
    content: Optional[str] = Field(default=None, description="Content for write_file.")
    recursive: Optional[bool] = Field(default=False, description="Recursive flag for delete_dir.")
    paths_uris: Optional[List[str]] = Field(default=None, description="List of URIs for read_multiple_files.")
    edits: Optional[List[Dict[str, str]]] = Field(default=None, description="List of edits for edit_file (e.g., [{'oldText': '...', 'newText': '...'}]).")
    dry_run: Optional[bool] = Field(default=False, description="Dry run flag for edit_file.")
    source_uri: Optional[str] = Field(default=None, description="Source URI for move_file.")
    destination_uri: Optional[str] = Field(default=None, description="Destination URI for move_file.")
    pattern: Optional[str] = Field(default=None, description="Glob pattern for search_files.")
    exclude_patterns: Optional[List[str]] = Field(default=None, description="List of glob patterns to exclude for search_files.")
    path: Optional[str] = Field(default=None, description="Direct path, alternative to URI for some server implementations.")
    source: Optional[str] = Field(default=None, description="Direct source path for move.")
    destination: Optional[str] = Field(default=None, description="Direct destination path for move.")


class MCPFileSystemTool(BaseTool, BaseModel): # Inherit from BaseModel for Pydantic features
    name: str = "MCPFileSystem"
    description: str = (
        "Provides access to various filesystem operations. "
        "To use this tool, the 'action' in your JSON output MUST be 'MCPFileSystem'. "
        "The 'action_input' for 'MCPFileSystem' MUST then be a JSON object (or a JSON string representing that object) "
        "which specifies the *actual filesystem operation* to perform. This inner JSON object MUST include an 'action' field "
        "(e.g., 'list_dir', 'read_file', etc.) and any other parameters for that specific filesystem operation.\n\n"
        "FOR EXAMPLE, TO LIST THE ROOT DIRECTORY:\n"
        "Thought: I need to list the root directory using MCPFileSystem.\n"
        "Action:\n"
        "```json\n"
        "{{\n"
        "  \"action\": \"MCPFileSystem\",\n"
        "  \"action_input\": {{\n"
        "    \"action\": \"list_dir\",\n" # This is the sub-action
        "    \"uri\": \"file:///\"\n"
        "  }}\n"
        "}}\n"
        "```\n\n"
        "ANOTHER EXAMPLE, TO READ THE FILE '/tmp/example.txt':\n"
        "Thought: I need to read /tmp/example.txt using MCPFileSystem.\n"
        "Action:\n"
        "```json\n"
        "{{\n"
        "  \"action\": \"MCPFileSystem\",\n"
        "  \"action_input\": {{\n"
        "    \"action\": \"read_file\",\n"     # Sub-action
        "    \"uri\": \"file:///tmp/example.txt\"\n"
        "  }}\n"
        "}}\n"
        "```\n\n"
        "Available sub-actions for the 'action' field within 'action_input' are:\n"
        "- 'list_dir': Requires 'uri' or 'path'.\n"
        "- 'read_file': Requires 'uri' or 'path'.\n"
        "- 'stat_file': Requires 'uri' or 'path'.\n"
        "- 'create_dir': Requires 'uri' or 'path'.\n"
        "- 'write_file': Requires 'uri' or 'path', and 'content'.\n"
        "- 'delete_file': Requires 'uri' or 'path'.\n"
        "- 'delete_dir': Requires 'uri' or 'path', optional 'recursive'.\n"
        "- 'read_multiple_files': Requires 'paths_uris' (list of URIs).\n"
        "- 'edit_file': Requires 'uri' or 'path', and 'edits' list. Optional 'dry_run'.\n"
        "- 'directory_tree': Requires 'uri' or 'path'.\n"
        "- 'move_file': Requires 'source_uri'/'source' and 'destination_uri'/'destination'.\n"
        "- 'search_files': Requires 'uri'/'path', and 'pattern'. Optional 'exclude_patterns'.\n"
        "- 'list_allowed_directories': No specific input parameters beyond the action itself.\n"
        "Ensure 'action_input' is correctly formatted as a JSON object or a valid JSON string."
    )
    args_schema: Type[BaseModel] = MCPFileSystemToolInput
    return_direct: bool = False # Langchain specific
    handle_tool_error: bool = True # Langchain specific

    # Configuration
    allowed_dirs: List[Path] = Field(default_factory=list)
    server_project_root_relative: str = Field(default_factory=lambda: os.path.join(
        "..", "..", "mcp_servers", "filesystem_server" 
    ))
    server_script_name_in_project: str = "dist/index.js" # Path to script within the project root
    node_executable: Path = Field(default_factory=lambda: Path(shutil.which("node") or "node"))
    session_init_timeout: float = 45.0 # Increased for Node.js server startup
    tool_call_timeout: float = 60.0 

    # Internal State (Async - for _arun)
    _session_async: Optional[ClientSession] = PrivateAttr(default=None)
    _lifecycle_task_async: Optional[asyncio.Task] = PrivateAttr(default=None)
    _session_ready_event_async: Optional[asyncio.Event] = PrivateAttr(default=None)
    _shutdown_event_async: Optional[asyncio.Event] = PrivateAttr(default=None)
    _init_lock_async: Optional[asyncio.Lock] = PrivateAttr(default=None)
    _is_closed_async: bool = PrivateAttr(default=False)
    
    _logger: Any = PrivateAttr(default=None) 
    _server_project_root_resolved: Optional[Path] = PrivateAttr(default=None) # Store resolved path

    _client_info: MCPImplementation = PrivateAttr(default_factory=lambda: MCPImplementation(
        name="AlpyAsyncFileSystemToolClient", version="0.3.1" # Incremented version
    ))

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def _initialize_internal_fields_validator(self) -> 'MCPFileSystemTool':
        if self._logger is None:
            # Use a more specific logger name for the instance
            self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}.{id(self)}")
            if not self._logger.hasHandlers(): # Add basic handler if not configured upstream
                handler = logging.StreamHandler(sys.stdout) # Log to stdout for visibility
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s')
                handler.setFormatter(formatter)
                self._logger.addHandler(handler)
                self._logger.propagate = False 
        
        if str(self.node_executable) == "node" and not self.node_executable.exists():
            resolved_node = shutil.which("node")
            if resolved_node: self.node_executable = Path(resolved_node)
            else: self._logger.critical("Node.js executable 'node' not found. MCPFileSystemTool will fail.")
        self._logger.info(f"Using node_executable: {self.node_executable}")

        # Resolve server_project_root based on this file's location
        current_dir_of_this_file = Path(__file__).parent
        self._server_project_root_resolved = (current_dir_of_this_file / self.server_project_root_relative).resolve()
        self._logger.info(f"Resolved server_project_root: {self._server_project_root_resolved}")

        resolved_allowed_dirs = []
        for adir_str in self.allowed_dirs: # Iterate over original config
            p = Path(adir_str) # Convert str to Path
            if not p.is_absolute():
                # Resolve relative paths against a sensible default, e.g., current working directory of Alpy
                # Or decide they must be absolute. For now, resolve against CWD.
                p_resolved = Path.cwd() / p
                self._logger.warning(f"Allowed directory '{adir_str}' was relative, resolved to '{p_resolved}' against CWD.")
            else:
                p_resolved = p.resolve()
            
            if not p_resolved.exists() or not p_resolved.is_dir():
                self._logger.error(f"Configured allowed directory does not exist or is not a directory: {p_resolved}")
                # Optionally raise an error or allow tool to start but fail operations for this dir
            resolved_allowed_dirs.append(p_resolved)
        self.allowed_dirs = resolved_allowed_dirs # Update with resolved Path objects
        self._logger.info(f"Final allowed_dirs: {[str(d) for d in self.allowed_dirs]}")
        
        return self

    # --- Async Core Methods (for _arun) ---
    async def _initialize_async_primitives_if_needed_async(self):
        if self._init_lock_async is None: self._init_lock_async = asyncio.Lock()
        if self._session_ready_event_async is None: self._session_ready_event_async = asyncio.Event()
        if self._shutdown_event_async is None: self._shutdown_event_async = asyncio.Event()

    def _get_server_params_async(self) -> StdioServerParameters:
        if not self._server_project_root_resolved: # Should be set by validator
            raise McpFileSystemToolError("Server project root not resolved during initialization.")
            
        server_script_abs_path = (self._server_project_root_resolved / self.server_script_name_in_project).resolve()

        if not server_script_abs_path.exists():
            msg = f"Node.js server script not found: {server_script_abs_path}. Build server in {self._server_project_root_resolved}."
            self._logger.error(msg)
            raise McpFileSystemToolError(msg)
        
        if not self.allowed_dirs:
            msg = "No allowed_dirs configured. Node.js server requires allowed directory arguments."
            self._logger.error(msg)
            raise McpFileSystemToolError(msg)

        command_str = str(self.node_executable.resolve())
        args_list = [str(server_script_abs_path)] + [str(p.resolve()) for p in self.allowed_dirs]
        cwd_str = str(self._server_project_root_resolved)
        server_env = os.environ.copy()

        self._logger.debug(f"Async Server params: command='{command_str}', args={args_list}, cwd='{cwd_str}'")
        return StdioServerParameters(command=command_str, args=args_list, cwd=cwd_str, env=server_env)

    async def _mcp_logging_callback_async(self, params: LoggingMessageNotificationParams) -> None:
        level = getattr(logging, params.level.upper(), logging.INFO)
        self._logger.log(level, f"[MCP Server Async Log - {params.scope}]: {params.message}")

    async def _manage_session_lifecycle_async(self):
        self._logger.info("Starting ASYNC MCP FileSystem session lifecycle...")
        try:
            server_params = self._get_server_params_async()
            async with mcp.client.stdio.stdio_client(server_params, errlog=sys.stderr) as (rs, ws):
                self._logger.info("ASYNC Stdio streams obtained.")
                async with ClientSession(
                    rs, ws, client_info=self._client_info, logging_callback=self._mcp_logging_callback_async
                ) as session:
                    self._logger.info("ASYNC ClientSession created. Initializing...")
                    init_result: InitializeResult = await asyncio.wait_for(session.initialize(), timeout=self.session_init_timeout)
                    self._logger.info(f"ASYNC MCP session initialized. Server capabilities: {init_result.capabilities}")
                    self._session_async = session
                    self._session_ready_event_async.set()
                    self._logger.info("ASYNC FileSystem session ready. Waiting for shutdown signal...")
                    await self._shutdown_event_async.wait()
                    self._logger.info("ASYNC Shutdown signal received.")
        except asyncio.TimeoutError:
            self._logger.error(f"ASYNC Timeout ({self.session_init_timeout}s) during FileSystem session init.")
        except asyncio.CancelledError:
            self._logger.info("ASYNC FileSystem session lifecycle task cancelled.")
        except Exception as e:
            self._logger.error(f"Error in ASYNC FileSystem session lifecycle: {e}", exc_info=True)
        finally:
            self._logger.info("ASYNC FileSystem session lifecycle task finished.")
            self._session_async = None
            if self._session_ready_event_async and not self._session_ready_event_async.is_set():
                self._session_ready_event_async.set()

    async def _ensure_session_ready_async(self):
        if self._is_closed_async: raise McpFileSystemToolError(f"{self.name} is closed (async).")
        await self._initialize_async_primitives_if_needed_async()
        if self._session_async and self._session_ready_event_async.is_set(): return

        async with self._init_lock_async:
            if self._session_async and self._session_ready_event_async.is_set(): return
            if self._is_closed_async: raise McpFileSystemToolError(f"{self.name} closed during async readiness check.")

            if self._lifecycle_task_async is None or self._lifecycle_task_async.done():
                self._logger.info("No active ASYNC FS lifecycle task or task is done. Starting new one.")
                self._session_ready_event_async.clear()
                self._shutdown_event_async.clear()
                self._lifecycle_task_async = asyncio.create_task(self._manage_session_lifecycle_async())
            
            self._logger.debug("Waiting for ASYNC FileSystem session to become ready...")
            try:
                await asyncio.wait_for(self._session_ready_event_async.wait(), timeout=self.session_init_timeout + 10.0)
            except asyncio.TimeoutError:
                self._logger.error("Timeout waiting for ASYNC MCP FileSystem session.")
                if self._lifecycle_task_async and not self._lifecycle_task_async.done():
                    self._lifecycle_task_async.cancel()
                    try: await self._lifecycle_task_async
                    except: pass # Absorb cancellation/other errors
                raise McpFileSystemToolError("Timeout establishing ASYNC MCP FileSystem session.")

            if not self._session_async or not self._session_ready_event_async.is_set():
                raise McpFileSystemToolError("Failed to establish a valid ASYNC MCP FileSystem session.")
            self._logger.info("ASYNC MCP FileSystem session is ready.")

    def _uri_to_path_str_validated(self, uri_str: str) -> str:
        if not uri_str.startswith("file://"): raise ValueError(f"Invalid URI: {uri_str}. Must be file://")
        path_part = uri_str[len("file://"):]
        if os.name == 'nt' and path_part.startswith('/') and len(path_part) > 1 and path_part[2] == ':':
            path_part = path_part[1:]
        
        # Use os.path.normpath and then resolve for robustness
        normalized_path = os.path.normpath(path_part)
        abs_path = Path(normalized_path).resolve() # Resolve symlinks and make absolute

        if not self.allowed_dirs: raise ValueError("Security check failed: No allowed_dirs configured.")
        
        is_allowed = any(abs_path == allowed_p or abs_path.is_relative_to(allowed_p) for allowed_p in self.allowed_dirs)
        if not is_allowed:
            self._logger.warning(f"Access denied: Path '{abs_path}' (from URI '{uri_str}') is not within allowed: {[str(d) for d in self.allowed_dirs]}")
            raise ValueError(f"Access to path '{abs_path}' is denied.")
        return str(abs_path)

    async def _arun(
        self, 
        action: str, 
        uri: Optional[str] = None, 
        content: Optional[str] = None, recursive: Optional[bool] = False,
        paths_uris: Optional[List[str]] = None, edits: Optional[List[Dict[str, str]]] = None,
        dry_run: Optional[bool] = False, source_uri: Optional[str] = None, destination_uri: Optional[str] = None,
        pattern: Optional[str] = None, exclude_patterns: Optional[List[str]] = None,
        path: Optional[str] = None, source: Optional[str] = None, destination: Optional[str] = None,
        run_manager: Optional[Any] = None # Langchain specific
    ) -> str:
        if self._is_closed_async:
            self._logger.warning(f"ASYNC: Tool {self.name} is closed. Action '{action}' aborted.")
            return json.dumps({"error": "Tool is closed."})

        self._logger.info(f"ASYNC _arun: Action='{action}', URI='{uri}', Path='{path}', Source='{source_uri or source}', Dest='{destination_uri or destination}'")
        
        try:
            await self._ensure_session_ready_async()
            if not self._session_async:
                return json.dumps({"error": "ASYNC MCP FileSystem session not available."})

            actual_tool_name_on_server: str = ""
            server_tool_params: Dict[str, Any] = {} # Parameters for the *specific* server tool

            # Prepare parameters based on the action
            # Convert URIs to paths and validate them first
            path_arg: Optional[str] = None
            if uri: path_arg = self._uri_to_path_str_validated(uri)
            elif path: path_arg = self._uri_to_path_str_validated(Path(path).resolve().as_uri())

            if action == "list_dir":
                actual_tool_name_on_server = "list_directory"
                if not path_arg: return json.dumps({"error": "'uri' or 'path' required for list_dir"})
                server_tool_params = {"path": path_arg}
            elif action == "read_file":
                actual_tool_name_on_server = "read_file"
                if not path_arg: return json.dumps({"error": "'uri' or 'path' required for read_file"})
                server_tool_params = {"path": path_arg}
            elif action == "stat_file":
                actual_tool_name_on_server = "get_file_info"
                if not path_arg: return json.dumps({"error": "'uri' or 'path' required for stat_file"})
                server_tool_params = {"path": path_arg}
            elif action == "create_dir":
                actual_tool_name_on_server = "create_directory"
                if not path_arg: return json.dumps({"error": "'uri' or 'path' required for create_dir"})
                server_tool_params = {"path": path_arg}
            elif action == "write_file":
                actual_tool_name_on_server = "write_file"
                if not path_arg: return json.dumps({"error": "'uri' or 'path' required for write_file"})
                if content is None: return json.dumps({"error": "'content' required for write_file"})
                server_tool_params = {"path": path_arg, "content": content}
            elif action == "delete_file":
                actual_tool_name_on_server = "delete_file"
                if not path_arg: return json.dumps({"error": "'uri' or 'path' required for delete_file"})
                server_tool_params = {"path": path_arg}
            elif action == "delete_dir":
                actual_tool_name_on_server = "delete_directory"
                if not path_arg: return json.dumps({"error": "'uri' or 'path' required for delete_dir"})
                server_tool_params = {"path": path_arg, "recursive": bool(recursive)}
            elif action == "read_multiple_files":
                actual_tool_name_on_server = "read_multiple_files"
                if not paths_uris: return json.dumps({"error": "'paths_uris' required for read_multiple_files"})
                server_tool_params = {"paths": [self._uri_to_path_str_validated(p_uri) for p_uri in paths_uris]}
            elif action == "edit_file":
                actual_tool_name_on_server = "edit_file"
                if not path_arg: return json.dumps({"error": "'uri' or 'path' required for edit_file"})
                if edits is None: return json.dumps({"error": "'edits' are required for edit_file"})
                server_tool_params = {"path": path_arg, "edits": edits, "dryRun": bool(dry_run)}
            elif action == "directory_tree":
                actual_tool_name_on_server = "directory_tree"
                if not path_arg: return json.dumps({"error": "'uri' or 'path' required for directory_tree"})
                server_tool_params = {"path": path_arg}
            elif action == "move_file":
                actual_tool_name_on_server = "move_file"
                src_p = self._uri_to_path_str_validated(source_uri) if source_uri else (self._uri_to_path_str_validated(Path(source).resolve().as_uri()) if source else None)
                dst_p = self._uri_to_path_str_validated(destination_uri) if destination_uri else (self._uri_to_path_str_validated(Path(destination).resolve().as_uri()) if destination else None)
                if not src_p or not dst_p: return json.dumps({"error": "Both source and destination URIs/paths are required for 'move_file'"})
                server_tool_params = {"source": src_p, "destination": dst_p}
            elif action == "search_files":
                actual_tool_name_on_server = "search_files"
                if not path_arg: return json.dumps({"error": "'uri' or 'path' (for search root) required for search_files"})
                if pattern is None: return json.dumps({"error": "'pattern' required for search_files"})
                server_tool_params = {"path": path_arg, "pattern": pattern}
                if exclude_patterns is not None: server_tool_params["excludePatterns"] = exclude_patterns
            elif action == "list_allowed_directories":
                actual_tool_name_on_server = "list_allowed_directories"
                server_tool_params = {} 
            else:
                self._logger.error(f"Unknown action '{action}' requested for FileSystem tool.")
                return json.dumps({"error": f"Unknown or unsupported action: {action}"})

        except ValueError as ve: # From _uri_to_path_str_validated
            self._logger.error(f"Path validation error for action '{action}': {ve}")
            return json.dumps({"error": str(ve)})

        self._logger.debug(f"ASYNC Calling MCP tool '{actual_tool_name_on_server}' with server_tool_params: {server_tool_params}")
        try:
            response: CallToolResult = await asyncio.wait_for(
                self._session_async.call_tool(name=actual_tool_name_on_server, arguments=server_tool_params),
                timeout=self.tool_call_timeout + 15.0 
            )
            
            self._logger.debug(f"ASYNC Response for '{actual_tool_name_on_server}': isError={response.isError}, content_type={type(response.content)}")

            if response.isError:
                err_detail = f"Server error for action '{action}' with tool '{actual_tool_name_on_server}'."
                if response.content and isinstance(response.content, list) and len(response.content) > 0:
                    item = response.content[0]
                    if isinstance(item, MCPErrorData) and item.message: 
                        err_detail = item.message
                    elif isinstance(item, TextContent) and item.text: 
                        err_detail = item.text 
                    elif hasattr(item, 'message') and isinstance(getattr(item, 'message'), str): 
                        err_detail = getattr(item, 'message')
                    else:
                        err_detail += f" Raw error content: {str(item)[:200]}"
                self._logger.error(err_detail)
                return json.dumps({"error": err_detail})

            # Handle successful response
            if response.content and isinstance(response.content, list) and len(response.content) > 0:
                first_content_item = response.content[0]
                if isinstance(first_content_item, TextContent) and first_content_item.text is not None:
                    text_payload = first_content_item.text
                    
                    if action == "read_file":
                        self._logger.debug(f"Action '{action}' returning raw file content wrapped in JSON.")
                        return json.dumps({"content": text_payload})
                    elif action in ["list_dir", "stat_file", "search_files", "read_multiple_files", "edit_file", "directory_tree"]:
                        # These actions from the Node.js server are expected to return a JSON string directly in text_payload.
                        self._logger.debug(f"Action '{action}' returning TextContent as is (expected JSON string from server).")
                        # Validate if it's actually JSON before returning, or let the caller handle parse error
                        try:
                            json.loads(text_payload) # Test if it's valid JSON
                            return text_payload
                        except json.JSONDecodeError:
                            self._logger.error(f"Action '{action}' expected JSON string from server, but got non-JSON: {text_payload[:200]}")
                            return json.dumps({"error": "Received non-JSON response from server for an action expecting JSON.", "raw_response": text_payload})
                    elif action == "list_allowed_directories":
                        lines = [line.strip() for line in text_payload.splitlines() if line.strip()]
                        if lines and lines[0].lower().startswith("allowed directories:"):
                            lines.pop(0) 
                        self._logger.debug(f"Action '{action}' parsed lines to JSON list: {lines}")
                        return json.dumps(lines) 
                    else: 
                        # For actions like create_dir, write_file, delete_file, delete_dir, move_file
                        # where server sends a simple success message string.
                        self._logger.debug(f"Action '{action}' returning simple success message wrapped in JSON.")
                        return json.dumps({"status": "success", "message": text_payload})
                else: 
                    self._logger.warning(f"Unexpected successful content structure for '{actual_tool_name_on_server}': {type(first_content_item)}. Content: {str(first_content_item)[:200]}")
                    return json.dumps({"status": "success", "action": action, "message": "Operation completed with non-standard content format.", "raw_content": str(first_content_item)})
            
            self._logger.debug(f"Action '{action}' successful with no specific content items in response from server.")
            return json.dumps({"status": "success", "action": action, "message": "Operation completed successfully (no specific content items returned)."})

        except asyncio.TimeoutError:
            self._logger.error(f"Timeout executing ASYNC action '{action}' for tool '{actual_tool_name_on_server}'.")
            return json.dumps({"error": f"Timeout executing action '{action}'."})
        except McpFileSystemToolError as e: 
            self._logger.error(f"McpFileSystemToolError during ASYNC action '{action}': {e}")
            return json.dumps({"error": str(e)})
        except Exception as e:
            self._logger.error(f"Unexpected ASYNC error for action '{action}', tool '{actual_tool_name_on_server}': {e}", exc_info=True)
            return json.dumps({"error": f"Unexpected ASYNC error: {str(e)}"})

    async def close(self): # Async close
        if self._is_closed_async and (self._lifecycle_task_async is None or self._lifecycle_task_async.done()):
            self._logger.debug(f"ASYNC {self.name} already closed.")
            return
        self._logger.info(f"ASYNC Closing {self.name} (FileSystem)...")
        self._is_closed_async = True
        await self._initialize_async_primitives_if_needed_async()
        if self._shutdown_event_async: self._shutdown_event_async.set()

        if self._lifecycle_task_async and not self._lifecycle_task_async.done():
            self._logger.debug("ASYNC Waiting for FileSystem lifecycle task...")
            try:
                await asyncio.wait_for(self._lifecycle_task_async, timeout=15.0)
            except asyncio.TimeoutError:
                self._logger.warning("ASYNC Timeout waiting for FS lifecycle task. Cancelling.")
                self._lifecycle_task_async.cancel()
                try: await self._lifecycle_task_async
                except: pass # Absorb cancellation/errors
            except Exception as e: self._logger.error(f"ASYNC Error during FS task shutdown: {e}")
        
        if self._session_ready_event_async: self._session_ready_event_async.clear()
        if self._shutdown_event_async: self._shutdown_event_async.clear()
        self._session_async = None
        self._logger.info(f"ASYNC {self.name} (FileSystem) closed.")

    def _run(self, *args: Any, **kwargs: Any) -> str:
        # This method is called by LangChain's BaseTool if a sync interface is used.
        # Since Alpy is async, this should ideally not be called directly by the agent.
        # If it must be implemented, it should bridge to the async version.
        # However, for a purely async agent, we can state it's not supported.
        self._logger.warning(
            "_run (synchronous) called on async-native MCPFileSystemTool. This is not the intended execution path for an async agent."
        )
        raise NotImplementedError(
            "MCPFileSystemTool is async-native and its synchronous _run method is not supported in this async context. Use _arun."
        )

    def close_sync_bridge(self) -> None: # Renamed for clarity
        """Closes the synchronous bridge's event loop and thread."""
        self._logger.info(f"Closing SYNC bridge for {self.name}.")
        if self._event_loop_thread_sync and self._event_loop_thread_sync.is_alive():
            self._shutdown_event_sync.set() # Signal the sync lifecycle manager
            if self._loop_sync and self._loop_sync.is_running():
                self._loop_sync.call_soon_threadsafe(self._loop_sync.stop)
            
            self._event_loop_thread_sync.join(timeout=10.0)
            if self._event_loop_thread_sync.is_alive():
                self._logger.warning(f"SYNC bridge event loop thread for {self.name} did not join.")
        self._event_loop_thread_sync = None
        self._loop_sync = None
        self._session = None # Clear session potentially used by sync path
        self._logger.info(f"SYNC bridge for {self.name} closed.")

    def __del__(self):
        if not self._is_closed_async and hasattr(self, '_logger') and self._logger: # Check if logger exists
            self._logger.warning(
                f"ASYNC MCPFileSystemTool instance {id(self)} deleted without explicit "
                f"async close. Active lifecycle task: {self._lifecycle_task_async is not None and not self._lifecycle_task_async.done()}. "
                f"This might lead to resource leaks."
            )
        # Cannot reliably call async close() from __del__

async def main_fs_tool_test_async(): # Renamed main test to be async
    log_format = "%(asctime)s - %(levelname)s - %(name)s - [%(threadName)s] - %(module)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format) # Use INFO for less verbose test output
    logging.getLogger("asyncio").setLevel(logging.WARNING) # Quieten asyncio debug logs
    
    # Get a more specific logger for tests if needed, or use module logger directly
    test_logger = logging.getLogger(__name__ + ".FS_Test_Async") # Specific logger for this test run
    test_logger.info("Starting MCPFileSystemTool ASYNC example usage...")

    with tempfile.TemporaryDirectory(prefix="alpy_fs_test_async_") as tmpdir:
        test_logger.info(f"Using temporary directory for tests: {tmpdir}")
        allowed_path = Path(tmpdir).resolve()
        
        fs_tool = MCPFileSystemTool(
            allowed_dirs=[str(allowed_path)],
            # logger=test_logger.getChild("ToolInstance") # Pass a child logger if desired
        )
        # Access the tool's own logger after initialization
        tool_instance_logger = fs_tool._logger 
        tool_instance_logger.setLevel(logging.DEBUG) # Set tool's own logger to DEBUG

        test_dir_path = allowed_path / "my_async_test_folder"
        test_file_path = test_dir_path / "test_async_file.txt"
        file_content = "Hello from Alpy ASYNC! This is a test file.\nLine 2 for async edits."

        async def run_async_test_op(action, **params):
            test_logger.info(f"\n--- ASYNC Test Op: Action: {action}, Params: {params} ---")
            try:
                result_str = await fs_tool._arun(action=action, **params)
                test_logger.info(f"ASYNC Result Str: {result_str}")
                result_json = json.loads(result_str) # Assume server returns JSON string
                test_logger.info(f"ASYNC Result JSON: {json.dumps(result_json, indent=2)}")
                if isinstance(result_json, dict) and result_json.get("error"):
                    test_logger.error(f"ASYNC Operation reported an error: {result_json['error']}")
                return result_json
            except json.JSONDecodeError:
                test_logger.error(f"ASYNC Result was not valid JSON: {result_str}")
                return {"error": "Result was not valid JSON", "raw": result_str}
            except Exception as e:
                test_logger.error(f"ASYNC Unexpected error: {e}", exc_info=True)
                return {"error": str(e)}

        try:
            # Test 1: Create Directory
            await run_async_test_op(action="create_dir", uri=test_dir_path.as_uri())
            assert test_dir_path.is_dir()

            # Test 2: Write File
            await run_async_test_op(action="write_file", uri=test_file_path.as_uri(), content=file_content)
            assert test_file_path.is_file() and test_file_path.read_text() == file_content
            
            # Test 3: Read File
            read_res = await run_async_test_op(action="read_file", uri=test_file_path.as_uri())
            assert read_res.get("content") == file_content # Node server wraps content in {"content": ...}

            # Test 4: List Allowed Directories (should list the temp dir)
            list_allowed_res = await run_async_test_op(action="list_allowed_directories")
            assert isinstance(list_allowed_res, list) and str(allowed_path) in list_allowed_res, "Allowed dir not listed"


            test_logger.info("\n--- ASYNC tests completed. Check logs for details. ---")

        finally:
            test_logger.info("ASYNC Closing MCPFileSystemTool...")
            await fs_tool.close() # Use async close
            # fs_tool.close_sync_bridge() # Close sync bridge if it was used (not in this async test)
            test_logger.info("ASYNC MCPFileSystemTool closed.")

if __name__ == "__main__":
    # To test the _run (synchronous bridge), you'd need a separate test function
    # that doesn't use asyncio.run directly on main_fs_tool_test_async.
    # For now, just running the async test.
    asyncio.run(main_fs_tool_test_async())