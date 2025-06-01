"""
Image Embedder Tool

Creates FAISS vector stores from images using Gemini embeddings via MCP server.
Supports intelligent batching and error recovery.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, model_validator, PrivateAttr
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import Implementation as MCPImplementation

logger = logging.getLogger(__name__)

class ImageEmbedderInput(BaseModel):
    security_id: str = Field(..., description="Security identifier")
    image_paths: List[str] = Field(..., description="List of image paths to embed")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata for each image")
    
    @model_validator(mode='after')
    def validate_paths(self) -> 'ImageEmbedderInput':
        # Validate that all image paths exist
        for path in self.image_paths:
            if not Path(path).exists():
                raise ValueError(f"Image file not found: {path}")
            if not any(path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
                raise ValueError(f"File is not an image: {path}")
        return self

class ImageEmbedderTool(BaseTool, BaseModel):
    name: str = "image_embedder"
    description: str = (
        "Create FAISS vector store from images using Gemini embeddings.\n"
        "Args:\n"
        "    security_id: Security identifier\n"
        "    image_paths: List of image paths to embed\n"
        "    metadata: Optional metadata for each image\n"
        "Returns:\n"
        "    Path to created vector store"
    )
    args_schema: Type[BaseModel] = ImageEmbedderInput
    return_direct: bool = False
    _session: ClientSession = PrivateAttr(default=None)
    _logger_instance: logging.Logger = PrivateAttr(default=None)
    
    @model_validator(mode='after')
    def _tool_post_init(self):
        if self._logger_instance is None:
            self._logger_instance = logger.getChild(f"{self.name}.{id(self)}")
        return self
    
    async def _arun(self, security_id: str, image_paths: List[str], metadata: Optional[Dict] = None):
        # Validate security_id format
        if not security_id or not isinstance(security_id, str):
            raise ValueError(f"Invalid security_id: {security_id}")
            
        # Ensure all image paths are absolute
        image_paths = [str(Path(p).resolve()) for p in image_paths]
        
        # Start the MCP server process
        server_path = Path(__file__).parent.parent
        server_module = server_path / "mcp_servers" / "mcp_image_embedder" / "server.py"
        
        # Set up environment with PYTHONPATH
        env = dict(os.environ)
        env["PYTHONPATH"] = str(server_path)
        
        params = StdioServerParameters(
            command="python3",
            args=[
                str(server_module)
            ],
            cwd=str(server_path),
            env=env
        )
        
        client_info = MCPImplementation(name="ImageEmbedderToolClient", version="0.1.0")
        try:
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream, client_info=client_info) as session:
                    await session.initialize()
                    
                    # Prepare arguments with optional metadata
                    arguments = {
                        "security_id": security_id,
                        "image_paths": image_paths
                    }
                    if metadata:
                        arguments["metadata"] = metadata
                    
                    # Call the embedding tool
                    result = await session.call_tool(name="embed_images", arguments=arguments)
                    
                    if not hasattr(result, 'content'):
                        raise ValueError(f"Unexpected result format from server: {result}")
                        
                    # Parse JSON response
                    if isinstance(result.content, list) and len(result.content) > 0:
                        # Extract JSON from TextContent
                        text_content = result.content[0]
                        if hasattr(text_content, 'text'):
                            import json
                            response = json.loads(text_content.text)
                        else:
                            raise ValueError(f"Unexpected text content format: {text_content}")
                    else:
                        response = result.content
                    
                    if isinstance(response, dict):
                        if response.get('status') == 'error':
                            raise ValueError(f"Server error: {response.get('message', 'Unknown error')}")
                        if not response.get('vector_store_path'):
                            raise ValueError("Server response missing 'vector_store_path' field")
                            
                        return f"Created vector store at: {response['vector_store_path']}"
                    else:
                        raise ValueError(f"Unexpected response format: {response}")
                        
        except Exception as e:
            self._logger_instance.error(f"Error in image_embedder tool: {e}", exc_info=True)
            raise
    
    async def close(self):
        if hasattr(self, '_session') and self._session:
            try:
                await self._session.close()
                self._session = None
            except Exception as e:
                self._logger_instance.error(f"Error closing session: {e}")
                raise
    
    def _run(self, *args, **kwargs):
        raise NotImplementedError("This tool only supports async execution.")

async def find_security_path(security_id: str) -> Path:
    """Find the security path in any board directory."""
    base_dir = Path('/home/me/CascadeProjects/Alpy/otcmn_tool_test_output/current')
    
    # Search in all board directories
    for board in os.listdir(base_dir):
        board_path = base_dir / board
        if not board_path.is_dir():
            continue
            
        security_path = board_path / security_id
        if not security_path.exists():
            continue
            
        return security_path
    
    raise ValueError(f"Security ID {security_id} not found in any board")

async def find_images_for_security(security_id: str) -> List[str]:
    """Find all images in the images folder for a given security ID."""
    security_path = await find_security_path(security_id)
    images_path = security_path / 'images'
    
    if not images_path.exists() or not images_path.is_dir():
        raise ValueError(f"No images folder found for security {security_id}")
    
    # Find all images in the images directory
    images = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        images.extend([str(f) for f in images_path.glob(ext)])
    
    if not images:
        raise ValueError(f"No images found for security {security_id}")
    
    logger.info(f"Found {len(images)} images for security {security_id}")
    return sorted(images)  # Sort to ensure consistent order

async def test_image_embedder():
    """Test the image embedder with a real security ID."""
    security_id = "MN0LNDB68390"
    
    try:
        # Find images for the security
        image_paths = await find_images_for_security(security_id)
        logger.info(f"Found {len(image_paths)} images")
        
        # Create and initialize the tool
        tool = ImageEmbedderTool()
        
        # Run the embedding - metadata will be extracted from filenames
        result = await tool._arun(
            security_id=security_id,
            image_paths=image_paths
        )
        logger.info(f"Embedding result: {result}")
        
        # Clean up
        await tool.close()
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s][%(name)s] %(message)s'
    )
    
    logger.info("Running standalone Image Embedder tool test...")
    asyncio.run(test_image_embedder())
