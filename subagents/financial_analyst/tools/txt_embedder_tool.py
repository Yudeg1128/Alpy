"""
Txt Embedder Tool: Given a security_id, parses all images in its images folder to txt using the image parser tool, then builds a FAISS vector store from the txts using Gemini embeddings.
"""
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, List, Any, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, model_validator, PrivateAttr
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import Implementation as MCPImplementation

logger = logging.getLogger(__name__)

class ParseImagesInput(BaseModel):
    security_id: str = Field(description="Security identifier")

class ParseImagesOutput(BaseModel):
    status: str
    message: str
    parsed_txt_files: Optional[List[str]] = None

class BuildVectorStoreInput(BaseModel):
    security_id: str = Field(description="Security identifier")

class BuildVectorStoreOutput(BaseModel):
    status: str
    message: str
    vector_store_path: Optional[str] = None
    txt_files: Optional[List[str]] = None

class TxtEmbedderTool(BaseTool, BaseModel):
    name: str = "txt_embedder"
    description: str = (
        "Tool for text embedding workflows on security images. Provides two actions callable via the LangChain tool/action schema.\n\n"
        "Actions:\n"
        "1. parse_security_images_to_txts\n"
        "   Input: { 'security_id': <str> }\n"
        "   - Parses all images for the given security_id to .txt files. Skips images with existing txts.\n"
        "   Output: { 'status': <str>, 'message': <str>, 'parsed_txt_files': [<str>, ...] }\n\n"
        "2. build_faiss_vector_store\n"
        "   Input: { 'security_id': <str> }\n"
        "   - Builds a FAISS vector store from all .txt files for the given security_id (must be run after parsing).\n"
        "   Output: { 'status': <str>, 'message': <str>, 'vector_store_path': <str>, 'txt_files': [<str>, ...] }\n\n"
        "Arguments:\n"
        "    Only 'security_id' is required for both actions. Do NOT provide image paths or extra fields.\n\n"
        "Examples:\n"
        "    Action: parse_security_images_to_txts\n"
        "    Input: { 'security_id': 'MN0LNDB68390' }\n\n"
        "    Action: build_faiss_vector_store\n"
        "    Input: { 'security_id': 'MN0LNDB68390' }\n\n"
        "Returns JSON objects as described above."
    )
    return_direct: bool = False

    # Server configuration
    server_executable: str = Field(
        default=sys.executable,
        description="Python executable for running the MCP server"
    )
    server_script: Path = Field(
        default=Path(__file__).parent.parent / "mcp_servers" / "mcp_txt_embedder" / "server.py",
        description="Path to the MCP server script"
    )
    server_cwd_path: Path = Field(
        default=None,
        description="Working directory for the MCP server"
    )

    _session: Optional[ClientSession] = PrivateAttr(default=None)
    _init_lock: Optional[asyncio.Lock] = PrivateAttr(default=None)
    _session_ready_event: Optional[asyncio.Event] = PrivateAttr(default=None)
    _shutdown_event: Optional[asyncio.Event] = PrivateAttr(default=None)
    _lifecycle_task: Optional[asyncio.Task] = PrivateAttr(default=None)
    _is_closed: bool = PrivateAttr(default=False)
    _logger_instance: logging.Logger = PrivateAttr(default=None)
    _client_info: MCPImplementation = PrivateAttr(
        default_factory=lambda: MCPImplementation(
            name="TxtEmbedderToolClient",
            version="1.0.0"
        )
    )

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def _tool_post_init(self) -> 'TxtEmbedderTool':
        if self._logger_instance is None:
            self._logger_instance = logger.getChild(f"{self.name}.{id(self)}")
            if not self._logger_instance.handlers:
                self._logger_instance.addHandler(logging.StreamHandler())
                self._logger_instance.setLevel(logging.INFO)
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
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()
        if self._session_ready_event is None:
            self._session_ready_event = asyncio.Event()
        if self._shutdown_event is None:
            self._shutdown_event = asyncio.Event()

    def _get_server_params(self) -> StdioServerParameters:
        return StdioServerParameters(
            command=self.server_executable,
            args=[str(self.server_script)],
            cwd=str(self.server_cwd_path)
        )

    def _run(self, *args, **kwargs):
        import asyncio
        if args and isinstance(args[0], dict):
            input_dict = args[0]
        elif kwargs:
            input_dict = kwargs
        else:
            raise ValueError("No input provided to TxtEmbedderTool._run")
        action = input_dict.get('action')
        security_id = input_dict.get('security_id')
        if not action or not security_id:
            raise ValueError("Both 'action' and 'security_id' must be specified in the input dict.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            if action == 'parse_security_images_to_txts':
                return loop.run_until_complete(self.parse_security_images_to_txts(security_id=security_id))
            elif action == 'build_faiss_vector_store':
                return loop.run_until_complete(self.build_faiss_vector_store(security_id=security_id))
            else:
                raise ValueError(f"Unknown action: {action}")
        finally:
            loop.close()


    async def parse_security_images_to_txts(self, security_id: str) -> str:
        await self._initialize_async_primitives()
        server_params = self._get_server_params()
        self._logger_instance.info(f"Launching MCP server for txt embedder (parse_security_images_to_txts): {server_params}")
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream, client_info=self._client_info) as session:
                await session.initialize()
                args = {"input_data": {"security_id": security_id}}
                tool_res = await session.call_tool(name="parse_security_images_to_txts", arguments=args)
                if getattr(tool_res, 'isError', False):
                    msg = getattr(tool_res, 'message', None) or str(tool_res)
                    raise RuntimeError(f"Server error: {msg}")
                content = getattr(tool_res, 'content', None)
                if not content:
                    raise RuntimeError("No content in server response")
                # Ensure JSON serializability
                if hasattr(content, "dict"):
                    return json.dumps(content.dict())
                elif isinstance(content, list):
                    # If it's a list of Pydantic models
                    return json.dumps([item.dict() if hasattr(item, "dict") else item for item in content])
                elif isinstance(content, dict):
                    return json.dumps(content)
                else:
                    return str(content)

    async def build_faiss_vector_store(self, security_id: str) -> str:
        await self._initialize_async_primitives()
        server_params = self._get_server_params()
        self._logger_instance.info(f"Launching MCP server for txt embedder (build_faiss_vector_store): {server_params}")
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream, client_info=self._client_info) as session:
                await session.initialize()
                args = {"input_data": {"security_id": security_id}}
                tool_res = await session.call_tool(name="build_faiss_vector_store", arguments=args)
                if getattr(tool_res, 'isError', False):
                    msg = getattr(tool_res, 'message', None) or str(tool_res)
                    raise RuntimeError(f"Server error: {msg}")
                content = getattr(tool_res, 'content', None)
                if not content:
                    raise RuntimeError("No content in server response")
                # Ensure JSON serializability
                if hasattr(content, "dict"):
                    return json.dumps(content.dict())
                elif isinstance(content, dict):
                    return json.dumps(content)
                else:
                    return str(content)

    async def close(self):
        if self._is_closed and (self._lifecycle_task is None or self._lifecycle_task.done()):
            return
        self._logger_instance.info(f"Closing {self.name}...")
        self._is_closed = True
        await self._initialize_async_primitives()
        if self._shutdown_event:
            self._shutdown_event.set()
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
async def test_parse_security_images_to_txts(security_id=None):
    test_logger = logging.getLogger("TxtEmbedderToolTest.Parse")
    test_logger.setLevel(logging.INFO)
    test_logger.addHandler(logging.StreamHandler(sys.stdout))
    test_logger.info("Starting parse_security_images_to_txts test...")
    tool = TxtEmbedderTool()
    try:
        result = await tool.parse_security_images_to_txts(security_id=security_id)
        print("\n=== parse_security_images_to_txts Result ===")
        print(result)
    finally:
        await tool.close()
        test_logger.info("Test finished.")

async def test_build_faiss_vector_store(security_id=None):
    test_logger = logging.getLogger("TxtEmbedderToolTest.Faiss")
    test_logger.setLevel(logging.INFO)
    test_logger.addHandler(logging.StreamHandler(sys.stdout))
    test_logger.info("Starting build_faiss_vector_store test...")
    tool = TxtEmbedderTool()
    import json as _json
    try:
        result = await tool.build_faiss_vector_store(security_id=security_id)
        print("\n=== build_faiss_vector_store Result ===")
        print(result)
        # Try to print the metadata file if present
        try:
            result_dict = _json.loads(result)
            metadata_path = result_dict.get("metadata_path")
            if metadata_path and os.path.exists(metadata_path):
                print(f"\n=== vector_metadata.json ({metadata_path}) ===")
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    print(f.read())
            else:
                print("No metadata file found.")
        except Exception as e:
            print(f"Could not load metadata file: {e}")
    finally:
        await tool.close()
        test_logger.info("Test finished.")

if __name__ == "__main__":
    import asyncio
    # Test both parsing and faiss vector store
    asyncio.run(test_parse_security_images_to_txts(security_id="MN0LNDB68390"))
    asyncio.run(test_build_faiss_vector_store(security_id="MN0LNDB68390"))
