"""MCP Server for parsing images using Google Gemini."""

import asyncio
import base64
import json
import logging
import os
import mimetypes # Added for MIME type detection
import re
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
    convert_system_message_to_human=True,
    cache=False  # Disable caching
)

# --- Pydantic Models ---
class ImageParserInput(BaseModel):
    image_path: str

class ImageParserOutput(BaseModel):
    status: str
    message: str
    structured_content: Optional[dict] = None


def read_image(image_path: str) -> str:
    """Read image file and convert to base64."""
    try:
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        raise RuntimeError(f"Failed to read image file: {e}")

def store_parsing_result(input_image_path_str: str, parsed_content: str, new_extension: str = ".md") -> Path:
    """
    Store parsing result in a file with the same name as the input image,
    but with a new extension, in the same directory.
    """
    input_image_path = Path(input_image_path_str)
    # Determine the parent directory of the image's folder
    image_dir = input_image_path.parent
    parent_dir = image_dir.parent
    parsed_images_dir = parent_dir / "parsed_images"
    parsed_images_dir.mkdir(parents=True, exist_ok=True)
    output_path = parsed_images_dir / (input_image_path.stem + new_extension)

    logger.debug(f"[store_parsing_result] Writing to: {output_path}")
    try:
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
        
        # Compose new prompt for structured extraction
        with open(os.path.join(os.path.dirname(__file__), "doc_schema.json"), "r", encoding="utf-8") as schema_file:
            doc_schema = json.load(schema_file)
        prompt_to_use = (
            "You are a financial data extractor. Your task is to extract EXACT text and data from financial documents. "
            "Output MUST be a single, valid JSON object. Do NOT include any explanatory text, comments, or markdown wrappers before or after the JSON. "
            "Pay EXTREME attention to JSON syntax: ensure all brackets [], braces {}, and quotes "" are correctly paired and placed. Ensure commas , are used correctly between elements in arrays and key-value pairs in objects, and NOT after the last element. "
            "The JSON object must have a top-level key 'content_blocks' which is an array of objects. "
            "Each object in 'content_blocks' must have 'type' (string, e.g., 'text', 'table', 'header') and 'content' (string for text/header, or an array of objects for tables). "
            "For tables, 'content' must be an array of objects, where each object represents a row and keys are column headers (strings). All cell values in tables must also be strings. "
            "Ensure all strings are properly escaped within the JSON (e.g., backslashes in text). "
            "Include the 'source_page' field (integer or null) for each content block if possible. "
            "Example of a valid, minimal structure:"
            "```json\n"
            "{\n"
            "  \"content_blocks\": [\n"
            "    {\n"
            "      \"type\": \"header\",\n"
            "      \"content\": \"Section Title\",\n"
            "      \"source_page\": 1\n"
            "    },\n"
            "    {\n"
            "      \"type\": \"table\",\n"
            "      \"content\": [\n"
            "        {\"Column1\": \"Row1Cell1\", \"Column2\": \"Row1Cell2\"},\n"
            "        {\"Column1\": \"Row2Cell1\", \"Column2\": \"Row2Cell2\"}\n"
            "      ],\n"
            "      \"source_page\": 1\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "```\n"
            "Strictly adhere to this JSON structure and ensure validity. Double-check your output for JSON correctness before returning."
        )
        message = HumanMessage(content=[
            {"type": "text", "text": prompt_to_use},
            {"type": "image_url", "image_url": f"data:{mime_type};base64,{image_data}"}
        ])
        logger.debug(f"[parse_image] HumanMessage created")
        # Generate response
        response = await llm.ainvoke([message])
        logger.debug(f"[parse_image] Gemini LLM response received (content logged above).")
        # Parse LLM output as JSON
        try:
            raw = response.content.strip()
            # Robustly remove markdown code block wrappers if present
            # Handles ```json ... ```, ``` ... ```, ````json ... ````, ```` ... ```` etc.
            # and optional language identifiers, and surrounding whitespace.
            match = re.match(r"^`{3,}(?:json)?\s*(.*?)\s*`{3,}$", raw, re.DOTALL | re.IGNORECASE)
            if match:
                raw = match.group(1).strip()
            # Fallback for cases where only one set of backticks might be an issue, 
            # e.g. if the LLM forgets the closing ones, or only uses opening ones.
            # This is less common if the prompt is good, but adds a layer of resilience.
            elif raw.startswith('```json') and raw.endswith('```'): # Explicitly check for both if re fails but structure is simple
                raw = raw[len('```json'):-len('```')].strip()
            elif raw.startswith('```') and raw.endswith('```'):
                raw = raw[len('```'):-len('```')].strip()
            try:
                structured = json.loads(raw)
            except Exception as e:
                # Attempt LLM self-repair with targeted instructions
                error_msg = str(e)
                logger.error(f"Initial JSON parse failed: {error_msg}. Attempting LLM repair.")
                repair_prompt = (
                    "The previous JSON output could not be parsed because: "
                    f"{error_msg}\n"
                    "Common issues include: duplicate keys in table rows, missing keys/values, or malformed objects. "
                    "Instructions:\n"
                    "1. For each table, ensure every row is a dict with unique keys. If keys are repeated, append a suffix (e.g., 'Дүн_1', 'Дүн_2').\n"
                    "2. Fill missing keys or values with null.\n"
                    "3. Do not drop any data.\n"
                    "4. Output only a single valid JSON object matching this schema: { 'content_blocks': [ ... ] }\n"
                    "Example:\n"
                    "{\n  \"content_blocks\": [\n    {\n      \"type\": \"table\",\n      \"content\": [\n        {\"Column1\": \"A\", \"Column2\": \"B\"},\n        {\"Column1\": \"C\", \"Column2\": null}\n      ],\n      \"source_page\": 1\n    }\n  ]\n}\n"
                    "Here is the previous output to repair:\n"
                    f"{raw}"
                )
                repair_message = HumanMessage(content=[{"type": "text", "text": repair_prompt}])
                repair_response = await llm.ainvoke([repair_message])
                repair_raw = repair_response.content.strip()
                # Remove markdown code block wrappers if present
                match = re.match(r"^`{3,}(?:json)?\s*(.*?)\s*`{3,}$", repair_raw, re.DOTALL | re.IGNORECASE)
                if match:
                    repair_raw = match.group(1).strip()
                elif repair_raw.startswith('```json') and repair_raw.endswith('```'):
                    repair_raw = repair_raw[len('```json'):-len('```')].strip()
                elif repair_raw.startswith('```') and repair_raw.endswith('```'):
                    repair_raw = repair_raw[len('```'):-len('```')].strip()
                try:
                    structured = json.loads(repair_raw)
                    logger.info("LLM repair successful.")
                except Exception as e2:
                    logger.error(f"LLM repair failed: {e2}. Repair raw: {repair_raw}")
                    raise ValueError("LLM did not return valid JSON after repair attempt.")
        except Exception as e:
            logger.error(f"Failed to parse LLM output as JSON: {e}. Output was: {response.content}")
            raise ValueError("LLM did not return valid JSON.")
        # Store result as .json
        result_file_path = store_parsing_result(
            input_image_path_str=str(image_path_obj),
            parsed_content=json.dumps(structured, ensure_ascii=False, indent=2),
            new_extension=".json"
        )
        logger.info(f"[parse_image] Handler SUCCESS, result_file={result_file_path}")
        return ImageParserOutput(
            status="success",
            message=f"Successfully parsed image: {image_path_obj.name}. Result stored at: {result_file_path}",
            structured_content=structured
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
