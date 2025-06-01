import asyncio
import logging
import os
from typing import List, Type
from pathlib import Path
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, model_validator, PrivateAttr
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import Implementation as MCPImplementation

logger = logging.getLogger(__name__)

class PDFToImageInput(BaseModel):
    security_id: str = Field(..., description="Security identifier")
    pdf_paths: List[str] = Field(..., description="List of PDF file paths")
    
    @model_validator(mode='after')
    def validate_paths(self) -> 'PDFToImageInput':
        # Validate that all PDF paths exist
        for path in self.pdf_paths:
            if not Path(path).exists():
                raise ValueError(f"PDF file not found: {path}")
            if not path.lower().endswith('.pdf'):
                raise ValueError(f"File is not a PDF: {path}")
        return self

class PDFToImageTool(BaseTool, BaseModel):
    name: str = "pdf_to_image"
    description: str = (
        "Convert all pages of given PDFs for a security into images for downstream embedding.\n"
        "Args:\n"
        "    security_id: Security identifier\n"
        "    pdf_paths: List of PDF file paths\n"
        "Returns:\n"
        "    Status message indicating completion or error"
    )
    args_schema: Type[BaseModel] = PDFToImageInput
    return_direct: bool = False
    _session: ClientSession = PrivateAttr(default=None)
    _logger_instance: logging.Logger = PrivateAttr(default=None)

    @model_validator(mode='after')
    def _tool_post_init(self):
        if self._logger_instance is None:
            self._logger_instance = logger.getChild(f"{self.name}.{id(self)}")
        return self

    async def _arun(self, security_id: str, pdf_paths: List[str]):
        # Validate security_id format
        if not security_id or not isinstance(security_id, str):
            raise ValueError(f"Invalid security_id: {security_id}")
            
        # Ensure all PDF paths are absolute
        pdf_paths = [str(Path(p).resolve()) for p in pdf_paths]
        
        # Start the MCP server process
        params = StdioServerParameters(
            command="python3",
            args=[str(Path(__file__).parent.parent / "mcp_pdf_to_image" / "server.py")],
            cwd=str(Path(__file__).parent.parent / "mcp_pdf_to_image")
        )
        
        client_info = MCPImplementation(name="PDFToImageToolClient", version="0.1.0")
        try:
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream, client_info=client_info) as session:
                    await session.initialize()
                    arguments = {"security_id": security_id, "pdf_paths": pdf_paths}
                    result = await session.call_tool(name="pdf_to_image", arguments=arguments)
                    
                    if not hasattr(result, 'content'):
                        raise ValueError(f"Unexpected result format from server: {result}")
                        
                    response = result.content
                    if not isinstance(response, dict):
                        response = {'status': 'success', 'message': str(response)}
                        
                    if response.get('status') == 'error':
                        raise ValueError(f"Server error: {response.get('message', 'Unknown error')}")
                    
                    return f"Successfully processed PDFs. {response.get('message', '')}"
                    
        except Exception as e:
            self._logger_instance.error(f"Error in pdf_to_image tool: {e}", exc_info=True)
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

async def find_security_pdfs(security_id: str) -> List[str]:
    """Find all PDFs in the documents folder for a given security ID."""
    base_dir = Path('/home/me/CascadeProjects/Alpy/otcmn_tool_test_output/current')
    
    # Search in all board directories
    for board in os.listdir(base_dir):
        board_path = base_dir / board
        if not board_path.is_dir():
            continue
            
        security_path = board_path / security_id
        if not security_path.exists():
            continue
            
        docs_path = security_path / 'documents'
        if not docs_path.exists() or not docs_path.is_dir():
            continue
            
        # Find all PDFs in the documents directory
        pdfs = []
        for file in docs_path.glob('*.pdf'):
            pdfs.append(str(file))
        
        if pdfs:
            logger.info(f"Found {len(pdfs)} PDFs for security {security_id} in {board}")
            return pdfs
    
    raise ValueError(f"No PDFs found for security {security_id}")

async def test_pdf_to_image():
    """Test the PDF to image conversion with a real security ID."""
    security_id = "MN0LNDB68390"
    
    try:
        # Find PDFs for the security
        pdf_paths = await find_security_pdfs(security_id)
        logger.info(f"Found PDFs: {pdf_paths}")
        
        # Create and initialize the tool
        tool = PDFToImageTool()
        
        # Run the conversion
        result = await tool._arun(security_id=security_id, pdf_paths=pdf_paths)
        logger.info(f"Conversion result: {result}")
        
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
    
    logger.info("Running standalone PDF to Image tool test...")
    asyncio.run(test_pdf_to_image())
