"""MCP Server for parsing images using Google Gemini."""

import base64
import json
import logging
import os
import mimetypes # Added for MIME type detection
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import sys
from pathlib import Path

# Add parent directory to path to import config
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from financial_analyst.config import GOOGLE_API_KEY, GOOGLE_MODEL, LLM_TEMPERATURE
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP

# --- Configuration ---
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s][%(levelname)s][MCPImageParser] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mcp_image_parser.log')
    ]
)
logger = logging.getLogger("MCPImageParser")
logger.info("Starting MCP Image Parser server")

# --- Constants ---

# Configure Gemini
llm = ChatGoogleGenerativeAI(
    model=GOOGLE_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.0,  # Force deterministic output regardless of config
    top_p=1.0,  # No sampling
    max_output_tokens=2000,
    convert_system_message_to_human=True,
    cache=False  # Disable caching
)

# --- Pydantic Models ---
class ImageParserInput(BaseModel):
    image_path: str
    prompt: Optional[str] = (
        "Your primary task is to transcribe all text from the provided image. "
        "Focus solely on extracting the textual content. "
        "Do NOT return bounding boxes, labels, confidence scores, or any other form of image analysis metadata. "
        "The only expected output is the transcribed text itself, formatted as plain text. "
        "Following this transcription task, adhere to the specific formatting rules below for financial data extraction.\n\n"
        "You are a financial data extractor. Your task is to extract EXACT text and data from financial documents.\n\n"
        "CRITICAL REQUIREMENTS:\n"
        "1. CHARACTER ACCURACY:\n"
        "   - Copy each character EXACTLY as shown\n"
        "   - Maintain Mongolian character encoding\n"
        "   - Never substitute similar-looking characters\n"
        "   - Preserve ALL special characters\n\n"
        "2. DATA STRUCTURE:\n"
        "   Title: [document title]\n"
        "   Date: [YYYY-MM-DD]\n\n"
        "   Section: [heading]\n"
        "   [content]\n\n"
        "   Table: [title]\n"
        "   [tab-separated headers]\n"
        "   [tab-separated values]\n\n"
        "   Text: [paragraph]\n\n"
        "3. CONTENT RULES:\n"
        "   - No formatting or markup\n"
        "   - Keep exact numbers and signs\n"
        "   - Preserve whitespace in tables\n"
        "   - Join wrapped lines in paragraphs\n\n"
        "Remember: Your primary goal is 100% accurate character-for-character transcription."
    )

class ImageParserOutput(BaseModel):
    status: str
    message: str
    markdown_content: Optional[str] = None


def read_image(image_path: str) -> str:
    """Read image file and convert to base64."""
    try:
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        raise RuntimeError(f"Failed to read image file: {e}")

def store_parsing_result(input_image_path_str: str, parsed_content: str, new_extension: str = ".txt") -> Path:
    """
    Store parsing result in a file with the same name as the input image,
    but with a new extension, in the same directory.
    """
    input_image_path = Path(input_image_path_str)
    output_path = input_image_path.with_suffix(new_extension)

    logger.debug(f"[store_parsing_result] Writing to: {output_path}")
    try:
        # output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists, though it should for same dir
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(parsed_content)
    except Exception as e:
        logger.error(f"[store_parsing_result] Error writing file {output_path}: {e}", exc_info=True)
        raise
    return output_path

# --- MCP App ---
mcp_app = FastMCP(
    name="MCPImageParserServer",
    version="1.0.0",
    description="MCP server for parsing images using Google Gemini"
)

@mcp_app.tool(name="parse_image")
async def parse_image(input_data: ImageParserInput) -> ImageParserOutput:
    logger.info(f"[parse_image] Handler START for image_path={input_data.image_path}")
    try:
        logger.debug(f"[parse_image] Received request: {input_data}")

        image_path_obj = Path(input_data.image_path)
        if not image_path_obj.exists():
            logger.error(f"[parse_image] Image file not found: {input_data.image_path}")
            raise FileNotFoundError(f"Image file not found: {input_data.image_path}")
        if not image_path_obj.is_file():
            logger.error(f"[parse_image] Path is not a file: {input_data.image_path}")
            raise ValueError(f"Path is not a file: {input_data.image_path}")

        # Read image
        image_data = read_image(str(image_path_obj))
        
        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(image_path_obj)
        if mime_type is None:
            mime_type = "application/octet-stream" # Fallback
            logger.warning(f"[parse_image] Could not determine MIME type for {image_path_obj}. Using fallback: {mime_type}")
        logger.debug(f"[parse_image] image_data length={len(image_data)}, mime_type={mime_type}")
        
        # Create message with image
        # Get default prompt from ImageParserInput if none provided
        prompt_to_use = ImageParserInput.model_fields["prompt"].default
        
        # logger.info(f"[parse_image] PROMPT BEING SENT TO LLM: {prompt_to_use}")

        message = HumanMessage(content=[
            {"type": "text", "text": prompt_to_use},
            {"type": "image_url", "image_url": f"data:{mime_type};base64,{image_data}"}
        ])
        logger.debug(f"[parse_image] HumanMessage created")
        
        # Generate response
        response = await llm.ainvoke([message])
        logger.debug(f"[parse_image] Gemini LLM response received (content logged above).")
        
        # Store result using the new logic
        result_file_path = store_parsing_result(
            input_image_path_str=str(image_path_obj),
            parsed_content=response.content,
            new_extension=".txt" # Save as .txt
        )
        
        logger.info(f"[parse_image] Handler SUCCESS, result_file={result_file_path}")
        return ImageParserOutput(
            status="success",
            message=f"Successfully parsed image: {image_path_obj.name}. Result stored at: {result_file_path}",
            markdown_content=response.content
        )
    except FileNotFoundError as e:
        error_message = str(e)
        logger.error(f"[parse_image] Handler ERROR - File Not Found: {error_message}", exc_info=False) # exc_info=False for common errors
        return ImageParserOutput(status="error", message=error_message)
    except ValueError as e:
        error_message = str(e)
        logger.error(f"[parse_image] Handler ERROR - Value Error: {error_message}", exc_info=False)
        return ImageParserOutput(status="error", message=error_message)
    except Exception as e:
        error_message = str(e) if str(e) else repr(e)
        logger.error(f"[parse_image] Handler ERROR: {error_message}", exc_info=True)
        return ImageParserOutput(
            status="error",
            message=f"An unexpected error occurred: {error_message}"
        )

if __name__ == "__main__":
    import sys
    import asyncio
    # Ensure UTF-8 output for stdio MCP protocol
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
    logger.info("[BOOT] MCP server main starting; stdio encoding set to utf-8")
    asyncio.run(mcp_app.run_stdio_async())
