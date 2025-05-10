# mcp_servers/media_display_server.py

import asyncio
import logging
import subprocess # Keep for DEVNULL if needed
import webbrowser
from pathlib import Path
from typing import Literal, Optional
import shutil

# Corrected / Verified Imports
import sys 
import os  
from urllib.parse import urlparse, unquote
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP 
import codecs # Import codecs

# --- Configuration ---
# Use ISO format for better sorting/parsing if needed
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s][%(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S') 
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class MediaDisplayInput(BaseModel):
    source: str = Field(description="The URL or absolute local file path of the media to display.")
    media_type: Optional[Literal["image", "audio", "video", "document", "unknown"]] = Field(
        default="unknown",
        description="An optional hint about the media type (image, audio, video, document, unknown)."
    )

class MediaDisplayOutput(BaseModel):
    success: bool
    message: str

# --- Initialize FastMCP (Simplified) ---
media_display_mcp_server = FastMCP(
    name="MediaDisplayServer", 
    version="0.1.2", # Incremented version
    description="MCP Server for displaying media files using system defaults."
)

# --- Tool Implementation ---
@media_display_mcp_server.tool(
    name="display_media",
    description="Attempts to display media from a URL or local file path using the system's default application or web browser."
)

async def display_media_tool(input_params: MediaDisplayInput) -> MediaDisplayOutput:
    source_uri_or_path = input_params.source # Original input
    media_type_hint = input_params.media_type
    logger.info(f"Received request. Source: {source_uri_or_path}, Type: {media_type_hint}, Platform: {sys.platform}")

    # --- Initialize variables with defaults ---
    source_to_open: Optional[str] = None 
    is_local_file: bool = False
    path_validation_error: Optional[str] = None
    decoded_path_str: str = "[Path not decoded yet]" 

    # --- Path/URI Validation and Normalization (Keep your existing, corrected logic here) ---
    if source_uri_or_path.startswith("file://"):
        is_local_file = True
        try:
            parsed_uri = urlparse(source_uri_or_path)
            decoded_path_str = unquote(parsed_uri.path, encoding='utf-8', errors='replace')
            if os.name == 'nt' and len(decoded_path_str) > 1 and decoded_path_str.startswith('/') and decoded_path_str[2] == ':':
                 decoded_path_str = decoded_path_str[1:]
            logger.debug(f"Attempting Path resolution for: {decoded_path_str}")
            local_file = Path(decoded_path_str).resolve(strict=True) 
            if not local_file.is_file():
                 path_validation_error = f"Error: Resolved path '{local_file}' exists but is not a file."
            else:
                source_to_open = str(local_file) # Assign only on success
                logger.debug(f"Assigned source_to_open (file URI): {source_to_open}")
        except FileNotFoundError:
            path_validation_error = f"Error: Local file not found. URI: '{source_uri_or_path}', Path Attempted: '{decoded_path_str}'."
        except Exception as e_path: 
            path_validation_error = f"Error resolving path from URI '{source_uri_or_path}': {e_path}"
    elif source_uri_or_path.startswith("http://") or source_uri_or_path.startswith("https://"):
        is_local_file = False
        source_to_open = source_uri_or_path # Assign directly for URLs
        logger.debug(f"Assigned source_to_open (http URI): {source_to_open}")
    else: # Assume plain local path
        is_local_file = True
        try:
            local_file = Path(source_uri_or_path).resolve(strict=True)
            if not local_file.is_file():
                 path_validation_error = f"Error: Path '{local_file}' is not a file."
            else:
                source_to_open = str(local_file) # Assign only on success
                logger.debug(f"Assigned source_to_open (plain path): {source_to_open}")
        except FileNotFoundError:
             path_validation_error = f"Error: Local file not found at path '{source_uri_or_path}'."
        except Exception as e_path:
            path_validation_error = f"Error resolving local path '{source_uri_or_path}': {e_path}"

    # --- Early exit if path validation failed ---
    if path_validation_error:
        logger.error(path_validation_error)
        return MediaDisplayOutput(success=False, message=path_validation_error)
        
    # --- Safeguard Check & Re-Assurance ---
    if source_to_open is None:
         # This path *shouldn't* be reached if validation logic is sound
         msg = "Internal error: source_to_open was None after path validation."
         logger.error(msg)
         return MediaDisplayOutput(success=False, message=msg)
    
    # Explicitly ensure it's a string before proceeding (redundant but safe)
    source_to_open_str = str(source_to_open) 
    logger.debug(f"Validated source_to_open_str: {source_to_open_str}")

    # --- Attempt to Open Media ---
    opened_successfully = False
    method_used = "none"
    error_details = ""
    proceed_to_webbrowser = True 

    open_command_path: Optional[str] = None 
    platform_command_name: Optional[str] = None

    if sys.platform.startswith('linux'): platform_command_name = "xdg-open"
    elif sys.platform == 'darwin': platform_command_name = "open"
    
    if platform_command_name:
        resolved_path = shutil.which(platform_command_name)
        if resolved_path: open_command_path = resolved_path
        else: logger.warning(f"Command '{platform_command_name}' not found.")
    else: logger.debug("No specific platform command configured.")

    if open_command_path: 
        process = None
        try:
            logger.info(f"Attempting launch: '{open_command_path} {source_to_open_str}'") 
            process = await asyncio.create_subprocess_exec(
                open_command_path,    # Use resolved path
                source_to_open_str,   # Use definitely assigned string
                stdout=asyncio.subprocess.DEVNULL, 
                stderr=asyncio.subprocess.PIPE   
            )
            # ... (rest of the process.wait(), timeout, return code logic using source_to_open_str for logging) ...
            try:
                 stderr_bytes = None
                 return_code = await asyncio.wait_for(process.wait(), timeout=7.0) 
                 if process.stderr: stderr_bytes = await process.stderr.read()
                 error_message = stderr_bytes.decode(errors='replace').strip() if stderr_bytes else ""
                 if return_code == 0:
                     opened_successfully = True; method_used = platform_command_name; proceed_to_webbrowser = False
                 else: error_details = f"{platform_command_name} failed (RC:{return_code}): {error_message}"
            except asyncio.TimeoutError:
                error_details = f"{platform_command_name} timed out."; proceed_to_webbrowser = False; opened_successfully = True; method_used = f"{platform_command_name} (timed out)"
                if process and process.returncode is None: 
                    try: process.kill() 
                    except: pass
        except Exception as e_subproc:
             logger.error(f"Error launching '{open_command_path}' for '{source_to_open_str}': {e_subproc}", exc_info=True)
             error_details = f"Error launching '{platform_command_name}': {str(e_subproc)}"
             
    # Method 2: Fallback to webbrowser
    if not opened_successfully and proceed_to_webbrowser:
        try:
            logger.info(f"Attempting fallback with webbrowser: {source_to_open_str}") 
            target_for_webbrowser = Path(source_to_open_str).as_uri() if is_local_file else source_to_open_str
            loop = asyncio.get_running_loop()
            opened = await loop.run_in_executor(None, webbrowser.open, target_for_webbrowser)
            if opened:
                opened_successfully = True; method_used = "webbrowser"
            else:
                err_msg_wb = f"webbrowser.open() failed for '{target_for_webbrowser}'."
                logger.error(err_msg_wb)
                if error_details: error_details += f"; {err_msg_wb}"
                else: error_details = err_msg_wb
        except Exception as e_web:
             err_msg_wb_ex = f"Webbrowser error: {str(e_web)}"
             logger.error(f"Error using webbrowser for '{source_to_open_str}': {e_web}", exc_info=True)
             if error_details: error_details += f"; {err_msg_wb_ex}"
             else: error_details = err_msg_wb_ex

    # --- Return Final Result ---
    if opened_successfully:
        success_message = f"Successfully launched via {method_used} to open: '{source_to_open_str}'. Playback/display should start if app configured."
        if "(timed out)" in method_used: success_message += " (Note: Launcher process did not exit cleanly)."
        return MediaDisplayOutput(success=True, message=success_message)
    else:
        final_message = f"Failed to display media '{source_uri_or_path}'. Details: {error_details or 'Unknown reason.'}"
        return MediaDisplayOutput(success=False, message=final_message)

# --- Run Server ---
if __name__ == "__main__":
    logger.info("Starting FastMCP Media Display Server via media_display_mcp_server.run()...")
    try:
        media_display_mcp_server.run() 
    except KeyboardInterrupt:
         logger.info("FastMCP Media Display Server shutting down.")
    except Exception as e:
         logger.critical(f"FastMCP Media Display Server exited with critical error: {e}", exc_info=True)
         sys.exit(1) 