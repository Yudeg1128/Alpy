import asyncio
import json
import logging
import os
import shutil # For shutil.which
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type

from langchain_core.tools import BaseTool, ToolException
from langchain_core.callbacks.manager import AsyncCallbackManagerForToolRun # Optional
from pydantic import BaseModel, Field, model_validator, PrivateAttr

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

class MCPDocumentParserToolError(ToolException):
    """Custom exception for the MCPDocumentParserTool."""
    pass

# --- Pydantic Schema for LangChain Tool Input (Input from LLM) ---
# This mirrors the server's DocumentParseInput, excluding max_pages_to_process_for_testing
# as that is a server-internal test feature.
class MCPDocumentParserToolInput(BaseModel):
    document_path: str = Field(description="Absolute path to the document file (e.g., PDF, DOCX).")
    ocr_mode: Literal["auto", "force_images", "force_full_pages", "none"] = Field(
        default="auto",
        description="OCR mode: 'auto' (OCR images if low text on page, or if page seems image-based), "
                    "'force_images' (OCR all extracted images from pages regardless of text content), "
                    "'force_full_pages' (render each page as an image and OCR it; can be slow), "
                    "'none' (disable OCR)."
    )
    ocr_lang: str = Field(
        default="mon",
        description="Tesseract language string for OCR (e.g., 'eng' for English, 'mon' for Mongolian). "
                    "Ensure the corresponding language pack is installed on the server."
    )
    min_chars_for_text_page_auto_ocr: int = Field(
        default=100,
        description="In 'auto' OCR mode, if a page has fewer than this many characters (extracted by direct text methods like PyMuPDF), "
                    "the tool will attempt OCR on discrete images found on that page. This helps decide if a page is primarily image-based."
    )
    dpi_for_full_page_ocr: int = Field(
        default=300,
        description="DPI (dots per inch) to use when rendering a page to an image for 'force_full_pages' OCR mode. Higher DPI means better quality but slower processing."
    )
    # max_pages_to_process_for_testing: Not exposed to LLM, server handles this internally for its own testing.


# --- Main Tool Class ---
class MCPDocumentParserTool(BaseTool, BaseModel):
    name: str = "MCPDocumentParser"
    description: str = (
        "Parses documents (currently PDFs) to extract text, tables, and perform OCR (Optical Character Recognition) on images or full pages if needed. "
        "Useful for processing scanned documents or PDFs where text is not directly extractable.\n"
        "**Inputs**:\n"
        "- `document_path`: REQUIRED. The absolute local file system path to the document.\n"
        "- `ocr_mode` (Optional, default='auto'): Controls how OCR is applied. \n"
        "  - 'auto': Attempts OCR on images on a page if the page has little extractable text. Recommended for mixed documents.\n"
        "  - 'force_images': Forces OCR on all discrete images found in the document.\n"
        "  - 'force_full_pages': Renders each page as an image and then OCRs the entire page image. Useful for fully scanned/image-based PDFs but is slower.\n"
        "  - 'none': Disables OCR entirely.\n"
        "- `ocr_lang` (Optional, default='mon'): Language for OCR (e.g., 'mon' for Mongolian, 'eng' for English). Server must have the language pack installed.\n"
        "- `min_chars_for_text_page_auto_ocr` (Optional, default=100): Threshold for 'auto' OCR mode to decide if a page is text-heavy enough to skip image OCR.\n"
        "- `dpi_for_full_page_ocr` (Optional, default=300): Resolution for 'force_full_pages' OCR.\n\n"
        "**Output**:\n"
        "A JSON string containing:\n"
        "- `status`: 'success' or 'error'.\n"
        "- `message`: Overall message.\n"
        "- `file_name`: Name of the parsed file.\n"
        "- `total_pages`: Total pages in the document.\n"
        "- `processed_pages_count`: Number of pages actually processed (might be less than total_pages if a test limit was hit on server).\n"
        "- `pages_content`: List of objects, one per page:\n"
        "  - `page_number`: The page number.\n"
        "  - `text_pymupdf`: Text extracted directly by PyMuPDF.\n"
        "  - `page_full_ocr_text`: Text from OCR if 'force_full_pages' was used.\n"
        "  - `page_full_ocr_error`: Any error during full page OCR.\n"
        "- `all_tables`: List of tables found across all processed pages:\n"
        "  - `page_number`, `table_index_on_page`, `rows` (list of lists of cell strings), `extraction_method`, `notes`.\n"
        "- `all_images_ocr_results`: List of OCR results for discrete images:\n"
        "  - `page_number`, `image_index_on_page`, `ocr_text`, `ocr_lang_used`, `ocr_error`.\n"
        "- `tesseract_available`: Boolean indicating if Tesseract OCR was usable during this run.\n"
        "- `tesseract_initial_check_message`: Message from Tesseract's initial availability check on the server."
    )

    args_schema: Type[BaseModel] = MCPDocumentParserToolInput
    return_direct: bool = False
    handle_tool_error: bool = True

    server_executable: Path = Field(default_factory=lambda: Path(shutil.which("python") or "python3" or sys.executable))
    server_script: Path = Field(default_factory=lambda: (Path(__file__).resolve().parent.parent.parent / "mcp_servers" / "document_parser_server.py"))
    server_cwd_path: Optional[Path] = Field(default=None)

    session_init_timeout: float = 45.0
    tool_call_timeout: float = 300.0 # Document parsing can be long

    _session_async: Optional[ClientSession] = PrivateAttr(default=None)
    _lifecycle_task_async: Optional[asyncio.Task] = PrivateAttr(default=None)
    _session_ready_event_async: Optional[asyncio.Event] = PrivateAttr(default=None)
    _shutdown_event_async: Optional[asyncio.Event] = PrivateAttr(default=None)
    _init_lock_async: Optional[asyncio.Lock] = PrivateAttr(default=None)
    _is_closed_async: bool = PrivateAttr(default=False)
    _logger_instance: logging.Logger = PrivateAttr(default=None)
    _client_info: MCPImplementation = PrivateAttr(default_factory=lambda: MCPImplementation(name="AlpyAsyncDocumentParserClient", version="0.1.0"))

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def _tool_post_init_validator(self) -> 'MCPDocumentParserTool':
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
             self._logger_instance.warning(f"MCP Document Parser server script DOES NOT EXIST: {self.server_script}")
        return self

    async def _initialize_async_primitives(self):
        if self._init_lock_async is None: self._init_lock_async = asyncio.Lock()
        if self._session_ready_event_async is None: self._session_ready_event_async = asyncio.Event()
        if self._shutdown_event_async is None: self._shutdown_event_async = asyncio.Event()

    def _get_server_params(self) -> StdioServerParameters:
        if not self.server_executable.exists(): raise MCPDocumentParserToolError(f"Server exec not found: {self.server_executable}")
        if not self.server_script.exists(): raise MCPDocumentParserToolError(f"Server script not found: {self.server_script}")
        if not self.server_cwd_path or not self.server_cwd_path.is_dir(): raise MCPDocumentParserToolError(f"Server CWD invalid: {self.server_cwd_path}")
        server_log_file = self.server_cwd_path.parent / "logs" / "mcp_document_parser_server.log"
        server_log_file.parent.mkdir(parents=True, exist_ok=True) # Ensure logs directory exists

        # Ensure executable and script paths are strings and properly quoted if they contain spaces (though unlikely for these paths)
        # Using absolute paths for executable and script is generally safer.
        # The paths self.server_executable and self.server_script are already absolute.
        command_str = f"{str(self.server_executable)} {str(self.server_script)}"
        
        self._logger_instance.info(f"MCPDocumentParserServer log will be at: {server_log_file}")

        return StdioServerParameters(
            command="/bin/bash", # Using /bin/bash for explicit path
            args=["-c", command_str],
            cwd=str(self.server_cwd_path),
            env=os.environ.copy()
        )

    async def _mcp_server_log_callback(self, params: LoggingMessageNotificationParams) -> None:
        level = getattr(logging, params.level.upper(), logging.INFO)
        self._logger_instance.log(level, f"[MCP Server ({self.server_script.name}) - {params.scope}]: {params.message}")

    async def _manage_session_lifecycle(self):
        self._logger_instance.info(f"Starting {self.name} (server: {self.server_script.name}) MCP session lifecycle...")
        try:
            server_params = self._get_server_params()
            
            log_file_path = Path("logs") / "mcp_document_parser_server.log"
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file_path, 'a', encoding='utf-8') as ferr:
                async with stdio_client(server_params, errlog=ferr) as (rs, ws):
                    self._logger_instance.info(f"Stdio streams obtained for {self.server_script.name}.")
                    async with ClientSession(
                        rs, 
                        ws, 
                        client_info=self._client_info,
                        logging_callback=self._mcp_server_log_callback
                    ) as session:
                        self._logger_instance.info(f"ClientSession created. Initializing {self.name} session...")
                        init_result: InitializeResult = await asyncio.wait_for(
                            session.initialize(), 
                            timeout=self.session_init_timeout
                        )
                        self._logger_instance.info(f"{self.name} MCP session initialized. Server caps: {init_result.capabilities}")
                        
                        self._session_async = session
                        self._session_ready_event_async.set()
                        
                        self._logger_instance.info(f"{self.name} session ready. Waiting for shutdown signal...")
                        await self._shutdown_event_async.wait()
                        self._logger_instance.info(f"{self.name} shutdown signal received.")

        except asyncio.TimeoutError:
            self._logger_instance.error(f"Timeout ({self.session_init_timeout}s) during {self.name} session init.")
        except asyncio.CancelledError:
            self._logger_instance.info(f"{self.name} session lifecycle task cancelled.")
        except MCPDocumentParserToolError as e:
            self._logger_instance.error(f"Lifecycle error (setup or general): {e}", exc_info=True) # Added exc_info for more detail
        except Exception as e:
            self._logger_instance.error(f"Error in {self.name} session lifecycle: {e}", exc_info=True)
        finally:
            self._logger_instance.info(f"{self.name} session lifecycle task finished.")
            self._session_async = None # Ensure session is cleared
            # If the lifecycle task ends (normally or due to error before ready_event was set),
            # and ready_event wasn't set, set it now to prevent indefinite blocking in _ensure_session_ready.
            # Callers of _ensure_session_ready must check if self._session_async is None.
            if self._session_ready_event_async and not self._session_ready_event_async.is_set():
                self._session_ready_event_async.set()
            # Clear the shutdown event in case it was set and we are exiting due to an error
            if self._shutdown_event_async and self._shutdown_event_async.is_set():
                self._shutdown_event_async.clear()

    async def _ensure_session_ready(self):
        if self._is_closed_async: raise MCPDocumentParserToolError(f"{self.name} is closed.")
        await self._initialize_async_primitives()
        if self._session_async and self._session_ready_event_async.is_set(): return
        async with self._init_lock_async:
            if self._session_async and self._session_ready_event_async.is_set(): return
            if self._is_closed_async: raise MCPDocumentParserToolError(f"{self.name} closed during readiness check.")
            if self._lifecycle_task_async is None or self._lifecycle_task_async.done():
                self._logger_instance.info(f"Starting new {self.name} session lifecycle task.")
                self._session_ready_event_async.clear(); self._shutdown_event_async.clear()
                self._lifecycle_task_async = asyncio.create_task(self._manage_session_lifecycle())
            else: self._logger_instance.info(f"Waiting for existing {self.name} session setup.") # Should not happen if logic is correct
            
            self._logger_instance.debug(f"About to wait for _session_ready_event_async for {self.name} (timeout: {self.session_init_timeout + 5.0}s). Event state: {self._session_ready_event_async.is_set()}")
            try: 
                await asyncio.wait_for(self._session_ready_event_async.wait(), timeout=self.session_init_timeout + 5.0) # Shorter subsequent wait
                self._logger_instance.debug(f"_session_ready_event_async for {self.name} was set or already set.")
            except asyncio.TimeoutError:
                self._logger_instance.error(f"Timeout waiting for {self.name} _session_ready_event_async after {self.session_init_timeout + 5.0}s.")
                if self._lifecycle_task_async and not self._lifecycle_task_async.done():
                    self._logger_instance.warning(f"Attempting to cancel the lifecycle task for {self.name} due to session ready timeout.")
                    self._lifecycle_task_async.cancel()
                raise MCPDocumentParserToolError(f"Timeout establishing {self.name} MCP session.")
            
            if not self._session_async or not self._session_ready_event_async.is_set():
                self._logger_instance.error(f"Session for {self.name} is not valid after wait. Session obj: {'present' if self._session_async else 'None'}. Event set: {self._session_ready_event_async.is_set()}.")
                raise MCPDocumentParserToolError(f"Failed to establish valid {self.name} MCP session even after event was supposedly set or wait completed.")
            self._logger_instance.info(f"{self.name} MCP session is ready.")

    def _process_mcp_response(self, response: CallToolResult, action_name: str) -> str:
        # self._logger_instance.debug(f"Raw MCP Response for '{action_name}': isError={response.isError}, Content: {response.content}")
        if response.isError or not response.content or not isinstance(response.content[0], TextContent):
            err_msg = f"Error from MCP server for '{action_name}'."
            details = "Protocol error or unexpected response format."
            if response.content and isinstance(response.content[0], MCPErrorData): err_msg = response.content[0].message or err_msg
            elif response.content and isinstance(response.content[0], TextContent): err_msg = response.content[0].text
            self._logger_instance.error(f"MCP Server Error for '{action_name}': {err_msg}")
            if response.content and isinstance(response.content[0], TextContent) and response.content[0].text.startswith('{"status": "error"'):
                return response.content[0].text
            return json.dumps({"status": "error", "error": "MCP_CLIENT_PROCESSING_ERROR", "message": err_msg, "details": details})

        try:
            json.loads(response.content[0].text) 
            return response.content[0].text 
        except (json.JSONDecodeError, AttributeError, IndexError) as e:
            self._logger_instance.error(f"Invalid/unexpected JSON from server for '{action_name}': {e}. Content: {str(response.content[0])[:500]}...")
            return json.dumps({"status": "error", "error": "INVALID_SERVER_JSON_RESPONSE", "message": "Server response was not valid JSON.", "details": str(response.content[0])[:200]})

    async def _arun(self, run_manager: Optional[AsyncCallbackManagerForToolRun] = None, **kwargs: Any) -> str:
        if self._is_closed_async: return json.dumps({"status": "error", "error": f"{self.name} is closed."})

        try:
            tool_input = MCPDocumentParserToolInput(**kwargs)
        except Exception as e:
            self._logger_instance.error(f"Invalid input arguments for {self.name}: {kwargs}. Error: {e}", exc_info=True)
            return json.dumps({"status": "error", "error": "INVALID_INPUT_ARGUMENTS", "message": str(e)})
        
        self._logger_instance.info(f"Executing {self.name} via MCP with input: {tool_input.model_dump_json(exclude_none=True)}")
        
        # The server's tool expects the arguments directly under "input_data"
        mcp_arguments_for_server = {"input_data": tool_input.model_dump(exclude_none=True)}
        server_tool_name = "parse_document" # Matches the @mcp_app.tool name on the server

        try:
            await self._ensure_session_ready()
            if not self._session_async: return json.dumps({"status": "error", "error": f"{self.name} session not available."})

            # --- DIAGNOSTIC DELAY --- 
            self._logger_instance.debug("MCP session ready. Adding 1s delay before first tool call to server as a diagnostic step.")
            await asyncio.sleep(1) # Potential race condition mitigation/diagnosis
            # --- END DIAGNOSTIC DELAY ---

            self._logger_instance.debug(f"Calling MCP server tool '{server_tool_name}' with arguments: {mcp_arguments_for_server}")

            try:
                response: CallToolResult = await asyncio.wait_for(
                    self._session_async.call_tool(name=server_tool_name, arguments=mcp_arguments_for_server),
                    timeout=self.tool_call_timeout
                )
            except asyncio.TimeoutError:
                self._logger_instance.error(f"Timeout calling MCP server tool '{server_tool_name}'.")
                return json.dumps({"status": "error", "error": "TOOL_CALL_TIMEOUT", "message": f"Tool call to '{server_tool_name}' timed out after {self.tool_call_timeout}s."})

            return self._process_mcp_response(response, server_tool_name)

        except MCPDocumentParserToolError as e:
            return json.dumps({"status": "error", "error": "MCP_TOOL_CLIENT_ERROR", "message": str(e)})
        except Exception as e:
            self._logger_instance.error(f"Unexpected error during {self.name} action: {e}", exc_info=True)
            return json.dumps({"status": "error", "error": "UNEXPECTED_CLIENT_ERROR", "message": f"Unexpected error: {str(e)}"})

    async def close(self): # Standard close logic
        if self._is_closed_async and (self._lifecycle_task_async is None or self._lifecycle_task_async.done()): return
        self._logger_instance.info(f"Closing {self.name} (server: {self.server_script.name})...")
        self._is_closed_async = True
        await self._initialize_async_primitives() 
        if self._shutdown_event_async: self._shutdown_event_async.set()
        if self._lifecycle_task_async and not self._lifecycle_task_async.done():
            self._logger_instance.info(f"Waiting for {self.name} lifecycle task shutdown...")
            try: await asyncio.wait_for(self._lifecycle_task_async, timeout=10.0) # Shorter timeout for parser server
            except asyncio.TimeoutError:
                self._logger_instance.warning(f"Timeout waiting for {self.name} task. Cancelling.")
                self._lifecycle_task_async.cancel(); await asyncio.sleep(0) 
                try: await self._lifecycle_task_async
                except asyncio.CancelledError: self._logger_instance.info(f"{self.name} task cancelled.")
            except Exception: pass 
        if self._session_ready_event_async: self._session_ready_event_async.clear()
        self._session_async = None; self._lifecycle_task_async = None
        self._logger_instance.info(f"{self.name} (server: {self.server_script.name}) closed.")

    def _run(self, *args: Any, **kwargs: Any) -> str: # Sync version not implemented
        raise NotImplementedError(f"{self.name} is async-native. Use `_arun` or `ainvoke`.")

    def __del__(self):
         if not self._is_closed_async and hasattr(self, '_logger_instance') and self._logger_instance:
              self._logger_instance.warning(f"{self.name} (ID: {id(self)}) instance deleted without explicit .close().")

# --- Example Usage (for standalone testing) ---
async def main_test_mcp_document_parser_tool():
    log_level_str = os.getenv("LOG_LEVEL_TOOL_TEST", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    Path("logs").mkdir(parents=True, exist_ok=True)
    if not logging.getLogger().hasHandlers(): logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s [%(threadName)s] - %(message)s', stream=sys.stdout)
    if log_level > logging.DEBUG: logging.getLogger("asyncio").setLevel(logging.WARNING)

    test_logger = logging.getLogger(__name__ + ".MCPDocumentParserTool_Test")
    test_logger.setLevel(log_level)
    
    test_logger.info("Starting MCPDocumentParserTool standalone test with multiple PDFs...")
    test_logger.warning("Ensure 'document_parser_server.py' is running in mcp_servers/ and its env has dependencies (PyMuPDF, pdfplumber, pytesseract + lang packs).")

    tool: Optional[MCPDocumentParserTool] = None
    
    pdf_base_dir = Path("/home/me/CascadeProjects/Alpy/otcmn_tool_test_output/current/secondary_board_B/MN0LNDB68390/documents").resolve()
    pdf_filenames = [
        "Албан бичиг ба 2001-2003 маягт_1.pdf",
        "Зээлжих зэрэглэлийн үнэлгээний тайлан (1).pdf",
        "Мэргэжлийн байгууллагын дүгнэлтүүд (3).pdf",
        "хавсралт (1).pdf"
    ]

    # Enhanced print_summary with more specific error handling
    def print_summary(json_string: str, test_case_name: str, pdf_file_name: str):
        print(f"\nAttempting to print summary for {test_case_name} (PDF: {pdf_file_name}):")
        try:
            parsed_output = json.loads(json_string)
            print(f"  Successfully parsed JSON for {test_case_name} (PDF: {pdf_file_name}).")

            print(f"  Status: {parsed_output.get('status')}, Message: {parsed_output.get('message')}")
            print(f"  File: {parsed_output.get('file_name')}, Total Pages: {parsed_output.get('total_pages')}, Processed: {parsed_output.get('processed_pages_count')}")
            print(f"  Tesseract Available: {parsed_output.get('tesseract_available')}, Initial Check: {parsed_output.get('tesseract_initial_check_message')}")

            pages_content = parsed_output.get('pages_content', [])
            print(f"  Pages Content Extracted: {len(pages_content)}")
            for i, page in enumerate(pages_content[:3]): # Show first 3 pages summary
                text_pymupdf_summary = (page.get('text_pymupdf') or '')[:50] + ('...' if len(page.get('text_pymupdf') or '') > 50 else '')
                page_ocr_summary = (page.get('page_full_ocr_text') or '')[:50] + ('...' if len(page.get('page_full_ocr_text') or '') > 50 else '')
                ocr_error_summary = page.get('page_full_ocr_error')
                print(f"    Page {page.get('page_number')}: PyMuPDF Text: '{text_pymupdf_summary}', Full Page OCR: '{page_ocr_summary}'" + (f", OCR Error: {ocr_error_summary}" if ocr_error_summary else ""))
            if len(pages_content) > 3:
                print(f"    ... and {len(pages_content) - 3} more pages content.")

            all_tables = parsed_output.get('all_tables', [])
            print(f"  Tables Extracted: {len(all_tables)}")
            for i, table in enumerate(all_tables[:2]): # Show first 2 tables summary
                print(f"    Table on Page {table.get('page_number')} (Idx {table.get('table_index_on_page')}): {len(table.get('rows', []))} rows. Method: {table.get('extraction_method')}. Notes: {table.get('notes')}")
            if len(all_tables) > 2:
                print(f"    ... and {len(all_tables) - 2} more tables.")

            all_images_ocr = parsed_output.get('all_images_ocr_results', [])
            print(f"  Images OCR Results: {len(all_images_ocr)}")
            for i, img_ocr in enumerate(all_images_ocr[:3]): # Show first 3 image OCR summaries
                ocr_text_summary = (img_ocr.get('ocr_text') or '')[:50] + ('...' if len(img_ocr.get('ocr_text') or '') > 50 else '')
                print(f"    Image on Page {img_ocr.get('page_number')} (Idx {img_ocr.get('image_index_on_page')}): OCR Text: '{ocr_text_summary}', Lang: {img_ocr.get('ocr_lang_used')}, Error: {img_ocr.get('ocr_error')}")
            if len(all_images_ocr) > 3:
                print(f"    ... and {len(all_images_ocr) - 3} more image OCR results.")
            print(f"  Successfully completed print_summary for {test_case_name} (PDF: {pdf_file_name}).")
            return parsed_output
        except json.JSONDecodeError as e_json:
            print(f"  Error: Could not parse JSON response for {test_case_name} (PDF: {pdf_file_name}): {e_json}")
            print(f"  Problematic JSON string (first 1000 chars): {json_string[:1000]}...")
            return None
        except Exception as e_summary:
            print(f"  Error: Unexpected error during print_summary for {test_case_name} (PDF: {pdf_file_name}): {e_summary}", exc_info=True)
            return None

    try:
        tool = MCPDocumentParserTool()
        if tool._logger_instance and log_level <= logging.DEBUG: tool._logger_instance.setLevel(logging.DEBUG)

        for pdf_file_name in pdf_filenames:
            pdf_to_test = pdf_base_dir / pdf_file_name
            if not pdf_to_test.exists():
                test_logger.error(f"PDF path does not exist: {pdf_to_test}. Skipping...")
                continue
            
            test_logger.info(f"\n{'='*20} PROCESSING PDF: {pdf_file_name} {'='*20}")

            tc1_name = f"Test Case 1 (auto/mon) for {pdf_file_name}"
            print(f"\n--- [{tc1_name}] ---")
            try:
                result_json_tc1 = await tool._arun(
                    document_path=str(pdf_to_test),
                    ocr_mode="auto",
                    ocr_lang="mon" 
                )
                parsed_result_tc1 = print_summary(result_json_tc1, tc1_name, pdf_file_name)
                if parsed_result_tc1:
                    assert parsed_result_tc1.get("status") == "success", f"{tc1_name} Parsing failed: {parsed_result_tc1.get('message')}"
            except Exception as e_tc1:
                 test_logger.error(f"Error during {tc1_name}: {e_tc1}", exc_info=True)

            tc2_name = f"Test Case 2 (force_full_pages/eng/150dpi) for {pdf_file_name}"
            print(f"\n--- [{tc2_name}] ---")
            result_json_tc2 = ""
            try:
                result_json_tc2 = await tool._arun(
                    document_path=str(pdf_to_test),
                    ocr_mode="force_full_pages",
                    ocr_lang="eng", 
                    dpi_for_full_page_ocr=150 
                )
                test_logger.info(f"{tc2_name}: Received JSON string of length {len(result_json_tc2)}.")
                test_logger.info(f"{tc2_name}: JSON snippet (first 500 chars): {result_json_tc2[:500]}...")
                
                parsed_result_tc2 = print_summary(result_json_tc2, tc2_name, pdf_file_name)
                
                if parsed_result_tc2:
                    assert parsed_result_tc2.get("status") == "success", f"{tc2_name} Force OCR parsing failed: {parsed_result_tc2.get('message')}"
                else:
                    test_logger.error(f"{tc2_name}: print_summary returned None, indicating an issue.")

            except Exception as e_tc2_arun:
                test_logger.error(f"{tc2_name}: Error during _arun or initial processing: {e_tc2_arun}", exc_info=True)
                test_logger.error(f"{tc2_name}: Potentially problematic JSON (if received before error): {result_json_tc2[:500]}...")

    except Exception as e:
        test_logger.error(f"Error during MCPDocumentParserTool test: {e}", exc_info=True)
    finally:
        if tool:
            test_logger.info("Closing MCPDocumentParserTool...")
            await tool.close()
            test_logger.info("MCPDocumentParserTool closed.")
        test_logger.info("MCPDocumentParserTool standalone test finished for all PDFs.")

if __name__ == "__main__":
    # Ensure logs directory exists for client-side view of server logs
    Path("logs").mkdir(parents=True, exist_ok=True)
    asyncio.run(main_test_mcp_document_parser_tool())