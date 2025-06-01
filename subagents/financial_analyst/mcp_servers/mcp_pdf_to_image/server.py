"""
MCP PDF-to-Image Server

Receives a security ID and a list of PDF files, outputs images (one per page, per PDF) for downstream vector embedding.
Optimizes images for Gemini model compatibility with proper size and format.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from pdf2image import convert_from_path
from PIL import Image
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s][MCPPDFToImage] %(message)s')
logger = logging.getLogger(__name__)

# Constants for image optimization
MAX_IMAGE_SIZE = (512, 512)  # Optimal size for Gemini
IMAGE_FORMAT = 'PNG'  # Lossless format
IMAGE_DPI = 200  # Good balance of quality and size
OUTPUT_BASE_DIR = Path('/home/me/CascadeProjects/Alpy/otcmn_tool_test_output/current')

# --- Pydantic Models ---
class PDFToImageInput(BaseModel):
    security_id: str = Field(..., description="Security identifier")
    pdf_paths: List[str] = Field(..., description="List of PDF file paths")

class PDFToImageOutput(BaseModel):
    status: str = Field(..., description="Success or error status")
    message: str = Field(..., description="Status message or error details")
    image_paths: Optional[List[str]] = Field(None, description="List of generated image paths")

# --- MCP Server ---
mcp_app = FastMCP(
    name="MCPPDFToImageServer",
    version="1.0.0",
    description="MCP server for converting PDFs to Gemini-optimized images"
)

def get_board_path(security_id: str) -> Path:
    """Find the board directory containing the security ID."""
    for board in os.listdir(OUTPUT_BASE_DIR):
        board_path = OUTPUT_BASE_DIR / board
        if not board_path.is_dir():
            continue
        if (board_path / security_id).exists():
            return board_path
    raise ValueError(f"Security ID {security_id} not found in any board directory")

def optimize_image(image: Image.Image) -> Image.Image:
    """Optimize image for Gemini model input."""
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize maintaining aspect ratio
    image.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    
    # Create new image with white background
    new_image = Image.new('RGB', MAX_IMAGE_SIZE, (255, 255, 255))
    
    # Paste resized image centered
    offset = ((MAX_IMAGE_SIZE[0] - image.size[0]) // 2,
             (MAX_IMAGE_SIZE[1] - image.size[1]) // 2)
    new_image.paste(image, offset)
    
    return new_image

@mcp_app.tool(name="pdf_to_image")
async def pdf_to_image(security_id: str, pdf_paths: List[str]) -> PDFToImageOutput:
    try:
        # Find correct board path
        board_path = get_board_path(security_id)
        security_path = board_path / security_id
        
        # Create images directory if it doesn't exist
        images_dir = security_path / "images"
        images_dir.mkdir(exist_ok=True)
        
        all_image_paths = []
        
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                logger.warning(f"PDF not found: {pdf_path}")
                continue
                
            try:
                # Convert PDF pages to images
                images = convert_from_path(
                    pdf_path,
                    dpi=IMAGE_DPI,
                    fmt=IMAGE_FORMAT.lower(),
                    thread_count=os.cpu_count() or 1
                )
                
                # Process each page
                for i, image in enumerate(images):
                    # Optimize for Gemini
                    optimized = optimize_image(image)
                    
                    # Generate output path
                    pdf_name = Path(pdf_path).stem
                    image_path = images_dir / f"{pdf_name}_page_{i+1}.{IMAGE_FORMAT.lower()}"
                    
                    # Save optimized image
                    optimized.save(
                        image_path,
                        format=IMAGE_FORMAT,
                        optimize=True
                    )
                    
                    all_image_paths.append(str(image_path))
                    logger.info(f"Generated image: {image_path}")
                    
            except Exception as e:
                logger.error(f"Error processing PDF {pdf_path}: {e}")
                continue
        
        if not all_image_paths:
            return PDFToImageOutput(
                status="error",
                message="No images were generated from any PDFs",
                image_paths=[]
            )
        
        return PDFToImageOutput(
            status="success",
            message=f"Generated {len(all_image_paths)} images",
            image_paths=all_image_paths
        )
        
    except Exception as e:
        logger.error(f"Error in pdf_to_image: {e}", exc_info=True)
        return PDFToImageOutput(
            status="error",
            message=f"Server error: {str(e)}",
            image_paths=[]
        )

# --- Run Server ---
if __name__ == "__main__":
    logger.info("Starting MCP PDF-to-Image Server...")
    try:
        mcp_app.run()
    except KeyboardInterrupt:
        logger.info("Server shutting down (KeyboardInterrupt).")
    except Exception as e:
        logger.critical(f"Server exited with error: {e}", exc_info=True)
