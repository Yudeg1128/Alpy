"""
Image Parser Tool

Parse images using Google Gemini to extract markdown formatted text and tables.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, model_validator, PrivateAttr

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import Implementation as MCPImplementation

logger = logging.getLogger(__name__)

class ImageParserInput(BaseModel):
    """Input schema for the image parser tool."""
    image_path: str = Field(description="Path to the image file to parse")
    prompt: Optional[str] = Field(
        default=None,
        description="Optional custom prompt for image analysis. If not provided, a default prompt will be used."
    )

class ImageParserTool(BaseTool, BaseModel):
    name: str = "image_parser"
    description: str = """Tool for parsing images using Google Gemini.
    Input should be a path to an image file. The tool will analyze the image and return a markdown formatted description.
    If the image contains tables, they will be formatted as markdown tables.
    You can optionally provide a custom prompt to guide the analysis."""
    
    args_schema: type[BaseModel] = ImageParserInput
    return_direct: bool = False

    # Server configuration
    server_executable: str = Field(
        default=sys.executable,
        description="Python executable for running the MCP server"
    )
    server_script: Path = Field(
        default=Path(__file__).parent.parent / "mcp_servers" / "mcp_image_parser" / "server.py",
        description="Path to the MCP server script"
    )
    server_cwd_path: Path = Field(
        default=None,
        description="Working directory for the MCP server"
    )

    # Private attributes
    _session: Optional[ClientSession] = PrivateAttr(default=None)
    _init_lock: Optional[asyncio.Lock] = PrivateAttr(default=None)
    _session_ready_event: Optional[asyncio.Event] = PrivateAttr(default=None)
    _shutdown_event: Optional[asyncio.Event] = PrivateAttr(default=None)
    _lifecycle_task: Optional[asyncio.Task] = PrivateAttr(default=None)
    _is_closed: bool = PrivateAttr(default=False)
    _logger_instance: logging.Logger = PrivateAttr(default=None)
    _client_info: MCPImplementation = PrivateAttr(
        default_factory=lambda: MCPImplementation(
            name="ImageParserToolClient",
            version="1.0.0"
        )
    )

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def _tool_post_init(self) -> 'ImageParserTool':
        if self._logger_instance is None:
            self._logger_instance = logger.getChild(f"{self.name}.{id(self)}")
            if not self._logger_instance.handlers:
                self._logger_instance.addHandler(logging.StreamHandler())
                self._logger_instance.setLevel(logging.INFO)
        
        # Resolve paths
        if isinstance(self.server_script, str):
            self.server_script = Path(self.server_script)
        self.server_script = self.server_script.resolve()
        
        if self.server_cwd_path is None:
            self.server_cwd_path = self.server_script.parent
        else:
            self.server_cwd_path = self.server_cwd_path.resolve()
        self._logger_instance.info(f"Using server_cwd_path: {self.server_cwd_path}")
        
        return self

    async def _initialize_async_primitives(self):
        """Initialize asyncio primitives if they don't exist."""
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()
        if self._session_ready_event is None:
            self._session_ready_event = asyncio.Event()
        if self._shutdown_event is None:
            self._shutdown_event = asyncio.Event()

    def _get_server_params(self) -> StdioServerParameters:
        """Get MCP server parameters."""
        return StdioServerParameters(
            command=self.server_executable,
            args=[str(self.server_script)],
            cwd=str(self.server_cwd_path)
        )

    async def _mcp_server_log_callback(self, params):
        """Handle logging messages from the MCP server."""
        level = getattr(logging, params.level.upper(), logging.INFO)
        self._logger_instance.log(level, f"[MCP Server - {params.scope}]: {params.message}")

    async def _manage_session_lifecycle(self):
        """Manage MCP session lifecycle."""
        self._logger_instance.info("Starting MCP session lifecycle...")
        
        try:
            # Initial checks
            if self._is_closed:
                self._logger_instance.debug("Tool is closed, cannot start session.")
                raise RuntimeError("Tool has been closed and cannot be reused")
                
            if self._session is not None:
                self._logger_instance.debug("Session already exists, skipping initialization.")
                return

            # Start MCP server
            server_params = self._get_server_params()
            self._logger_instance.info(f"Starting MCP server with: {server_params.command} {' '.join(server_params.args)}")
            
            # Create and manage session
            async with stdio_client(server_params, errlog=sys.stderr) as (rs, ws):
                self._logger_instance.info("Stdio streams obtained.")
                async with ClientSession(
                    rs, ws,
                    client_info=self._client_info,
                    logging_callback=self._mcp_server_log_callback
                ) as session:
                    self._logger_instance.info("ClientSession created. Initializing...")
                    init_result = await asyncio.wait_for(
                        session.initialize(),
                        timeout=30.0  # 30 second timeout for initialization
                    )
                    self._logger_instance.info(f"MCP session initialized. Server caps: {init_result.capabilities}")
                    self._session = session
                    self._session_ready_event.set()
                    self._logger_instance.info("Session ready. Waiting for shutdown signal...")
                    await self._shutdown_event.wait()
                    self._logger_instance.info("Shutdown signal received.")
        except asyncio.TimeoutError:
            self._logger_instance.error("Timeout during session initialization.")
            raise
        except asyncio.CancelledError:
            self._logger_instance.info("Session lifecycle task cancelled.")
            raise
        except Exception as e:
            self._logger_instance.error(f"Error in session lifecycle: {e}", exc_info=True)
            raise
        finally:
            self._logger_instance.info("Session lifecycle task finished.")
            self._session = None
            if self._session_ready_event and not self._session_ready_event.is_set():
                self._session_ready_event.set()  # Unblock waiters

    async def _ensure_session_ready(self):
        """Ensure MCP session is ready."""
        if self._is_closed:
            raise RuntimeError("Tool has been closed and cannot be reused")
        
        await self._initialize_async_primitives()
        
        if self._session and self._session_ready_event.is_set():
            return
            
        async with self._init_lock:  # Prevent concurrent initializations
            if self._session and self._session_ready_event.is_set():
                return
            if self._is_closed:
                raise RuntimeError("Tool closed during readiness check")
                
            if self._lifecycle_task is None or self._lifecycle_task.done():
                self._logger_instance.info("Starting new session lifecycle task")
                self._session_ready_event.clear()
                self._shutdown_event.clear()
                self._lifecycle_task = asyncio.create_task(self._manage_session_lifecycle())
            
            try:
                await asyncio.wait_for(
                    self._session_ready_event.wait(),
                    timeout=35.0  # 35 second timeout (slightly longer than session init)
                )
            except asyncio.TimeoutError:
                self._logger_instance.error("Timeout waiting for session to become ready")
                if self._lifecycle_task and not self._lifecycle_task.done():
                    self._lifecycle_task.cancel()
                raise RuntimeError("Timeout establishing MCP session")
            
            if not self._session or not self._session_ready_event.is_set():
                raise RuntimeError("Failed to establish a valid MCP session")
            
            self._logger_instance.info("MCP session is ready")

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("ImageParserTool is async only - use _arun")

    async def _arun(self, image_path: str, prompt: Optional[str] = None) -> str:
        """Parse an image using Google Gemini.

        Args:
            image_path: Path to the image file
            prompt: Optional custom prompt for analysis

        Returns:
            Markdown formatted analysis of the image

        Raises:
            RuntimeError: If image parsing fails or server returns an error
            FileNotFoundError: If image file does not exist
            ValueError: If image path is invalid
        """
        if not image_path:
            raise ValueError("Image path cannot be empty")
            
        image_path_obj = Path(image_path)

        if not image_path_obj.is_file():
            self._logger_instance.error(f"Image file not found at path: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Validate image extension
        allowed_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
        if image_path_obj.suffix.lower() not in allowed_extensions:
            self._logger_instance.error(f"Invalid image file extension: {image_path_obj.suffix}. Allowed: {allowed_extensions}")
            raise ValueError(
                f"Invalid image file extension: {image_path_obj.suffix}. "
                f"Allowed extensions are: {', '.join(sorted(list(allowed_extensions)))}"
            )

        # Ensure the session is ready (this will start the server if needed)
        await self._ensure_session_ready()
        self._logger_instance.info(f"Parsing image: {image_path}")

        # Prepare arguments for MCP call
        args = {
            "input_data": {
                "image_path": str(image_path_obj.resolve()),
                "prompt": prompt
            }
        }
        self._logger_instance.debug(f"Calling MCP server with args: {args}")
        
        try:
            # Call MCP server with timeout
            result = await asyncio.wait_for(
                self._session.call_tool(name="parse_image", arguments=args),
                timeout=60.0  # 60 second timeout for image parsing
            )
        except asyncio.TimeoutError:
            self._logger_instance.error(f"Timeout calling MCP server for image: {image_path}")
            raise RuntimeError(f"Timeout parsing image: {image_path}")
        except Exception as e:
            self._logger_instance.error(f"Error calling MCP server: {e}", exc_info=True)
            raise RuntimeError(f"Failed to call image parsing server: {e}")

        if result.isError:
            error_msg = result.content if isinstance(result.content, str) else str(result.content)
            self._logger_instance.error(f"MCP server error: {error_msg}")
            raise RuntimeError(f"MCP server error: {error_msg}")
        
        if not result.content or len(result.content) != 1:
            self._logger_instance.error(f"Unexpected response format from MCP server: {result}")
            raise RuntimeError(f"Unexpected response format from MCP server: {result}")
            
        try:
            response_text = result.content[0].text
            response = json.loads(response_text)
        except (json.JSONDecodeError, AttributeError, IndexError) as e:
            self._logger_instance.error(f"Failed to parse server response: {e}. Response text: '{response_text if 'response_text' in locals() else 'N/A'}'")
            raise RuntimeError(f"Invalid response format from server: {e}") from e
        
        if not isinstance(response, dict):
            self._logger_instance.error(f"Expected dict response from server, got {type(response)}")
            raise RuntimeError(f"Expected dict response from server, got {type(response)}")
            
        if response.get("status") == "error":
            error_msg = response.get("message", "Unknown error from image parsing server")
            self._logger_instance.error(f"Image parsing failed on server: {error_msg}")
            raise RuntimeError(f"Image parsing failed: {error_msg}")
            
        content = response.get("structured_content")
        if content is None:
            self._logger_instance.error("No structured_content in successful server response")
            raise RuntimeError("No structured_content in server response")
        return content

    async def close(self):
        """Close any open resources."""
        if self._is_closed and (self._lifecycle_task is None or self._lifecycle_task.done()):
            return
            
        self._logger_instance.info(f"Closing {self.name}...")
        self._is_closed = True
        await self._initialize_async_primitives()  # Ensure events exist
        
        if self._shutdown_event:
            self._shutdown_event.set()
            
        # First wait for lifecycle task to finish since it uses the session
        if self._lifecycle_task and not self._lifecycle_task.done():
            try:
                await asyncio.wait_for(self._lifecycle_task, timeout=10.0)
            except asyncio.TimeoutError:
                self._lifecycle_task.cancel()
                try:
                    await asyncio.wait_for(self._lifecycle_task, timeout=2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
            except Exception as e:
                self._logger_instance.warning(f"Error in lifecycle task shutdown: {e}")
        
        # Now close the session
        if self._session:
            try:
                await self._session.shutdown()
            except Exception as e:
                self._logger_instance.warning(f"Error closing MCP session: {e}")
            self._session = None
        
        if self._session_ready_event:
            self._session_ready_event.clear()
        if self._shutdown_event:
            self._shutdown_event.clear()
            
        self._logger_instance.info(f"{self.name} closed.")


# --- Example Usage (for standalone testing) ---
async def test_mongolian_doc(image_path=None):
    """Test the image parser with a Mongolian financial document."""
    test_logger = logging.getLogger("ImageParserToolTest")
    test_logger.setLevel(logging.INFO)
    test_logger.addHandler(logging.StreamHandler(sys.stdout))
    
    test_logger.info("Starting Mongolian document test...")
    
    tool = ImageParserTool()
    try:
        test_logger.info(f"Testing with image: {image_path}")
        result = await tool._arun(image_path=image_path)
        print("\n=== Parsing Result (Structured JSON) ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        # Check output file
        image_path_obj = Path(image_path)
        expected_output = image_path_obj.with_suffix(".json")
        if expected_output.exists():
            test_logger.info(f"Successfully saved result to: {expected_output}")
            print("\n=== Output JSON Content ===")
            with open(expected_output, "r", encoding="utf-8") as f:
                print(f.read())
            # Optionally, validate structure
            if not (isinstance(result, dict) and "content_blocks" in result):
                test_logger.error("Output JSON missing required 'content_blocks' key!")
            else:
                test_logger.info("Output JSON structure looks correct.")
        else:
            test_logger.error(f"Failed to find output file at: {expected_output}")
    finally:
        await tool.close()
        test_logger.info("Test finished.")

async def main_test_image_parser_tool():
    # Configure logging
    log_level_str = os.getenv("LOG_LEVEL_TOOL_TEST", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s [%(threadName)s] - %(message)s',
            stream=sys.stdout
        )
    test_logger = logging.getLogger(__name__ + ".ImageParserTool_Test")
    test_logger.setLevel(log_level)
    
    test_logger.info("Starting ImageParserTool standalone test...")
    
    try:
        await test_image_parser()
    except Exception as e:
        test_logger.error(f"Error during ImageParserTool test: {e}", exc_info=True)
    finally:
        test_logger.info("ImageParserTool standalone test finished.")

if __name__ == "__main__":
    import asyncio
    # Use the specific image path provided by the user for testing
    test_image_path = "/home/me/CascadeProjects/Alpy/otcmn_tool_test_output/current/secondary_board_B/MN0LNDB68390/images/Мэргэжлийн байгууллагын дүгнэлтүүд (3)_page_4.png"
    asyncio.run(test_mongolian_doc(image_path=test_image_path))
