import asyncio
import logging
from typing import Dict, Any
import subprocess
from pydantic import Field, BaseModel

from mcp.server.fastmcp import FastMCP

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s][FastMCPBashServer] %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastMCP application
# The name "BashServer" will be part of the server_info in the initialize response.
mcp = FastMCP(
    name="BashServer", 
    version="0.2.0", 
    description="A server that executes bash commands."
)

# Define the input and output models for the tool for clarity and type hinting
# FastMCP will infer the schema from these type hints and Field descriptions.
class BashCommandInput(BaseModel):
    command: str = Field(..., description="The bash command to execute.")
    timeout: int = Field(60, description="Timeout in seconds for the command.")

class BashCommandOutput(BaseModel):
    stdout: str = Field(..., description="The standard output of the command.")
    stderr: str = Field(..., description="The standard error of the command.")
    exit_code: int = Field(..., description="The exit code of the command.")

@mcp.tool(
    name="execute_bash_command", 
    description="Executes a bash command or an entire bash script and returns its stdout, stderr, and exit code. Suitable for both single commands and multi-line scripts.",
)
async def execute_bash_command_tool(input: BashCommandInput) -> BashCommandOutput:
    command = input.command
    timeout = input.timeout
    logger.info(f"Executing bash command: '{command[:50]}...' with timeout {timeout}s")
    
    process = None # Define process here to ensure it's available in except asyncio.TimeoutError
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout)
        exit_code = process.returncode
        
        stdout_str = stdout_bytes.decode(errors='replace').strip()
        stderr_str = stderr_bytes.decode(errors='replace').strip()
        
        logger.info(f"Command '{command[:50]}...' finished. Exit code: {exit_code}")
        return BashCommandOutput(stdout=stdout_str, stderr=stderr_str, exit_code=exit_code)
    
    except asyncio.TimeoutError:
        logger.error(f"Command '{command[:50]}...' timed out after {timeout} seconds.")
        if process and process.returncode is None: # Check if process exists and hasn't terminated
            try:
                process.kill()
                await process.wait() # Ensure the process is cleaned up
                logger.info(f"Killed timed-out process for command: '{command[:50]}...'")
            except Exception as e_kill:
                logger.error(f"Error killing timed-out process for '{command[:50]}...': {e_kill}")
        return BashCommandOutput(
            stdout="", 
            stderr=f"Command timed out after {timeout} seconds.", 
            exit_code=-1 # Indicate timeout
        )
    except Exception as e:
        logger.error(f"Error executing command '{command[:50]}...': {e}", exc_info=True)
        return BashCommandOutput(
            stdout="", 
            stderr=str(e), 
            exit_code=-2 # Indicate general execution error
        )

def execute_bash_command_sync(command: str, timeout: float | None = 60.0) -> BashCommandOutput:
    """Synchronously executes a bash command using subprocess.run."""
    logger.info(f"Executing bash command: '{command[:100]}{'...' if len(command) > 100 else ''}' with timeout {timeout}s")
    try:
        # Use /bin/bash -c to execute the command string. This allows for complex commands,
        # pipelines, and multi-line scripts to be executed correctly.
        process = subprocess.run(
            ["/bin/bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,  # We handle the exit code manually
        )
        stdout = process.stdout
        stderr = process.stderr
        exit_code = process.returncode
        return BashCommandOutput(stdout=stdout, stderr=stderr, exit_code=exit_code)
    except subprocess.TimeoutExpired:
        logger.error(f"Command '{command[:100]}...' timed out after {timeout} seconds.")
        return BashCommandOutput(
            stdout="", 
            stderr=f"Command timed out after {timeout} seconds.", 
            exit_code=-1 # Indicate timeout
        )
    except Exception as e:
        logger.error(f"Error executing command '{command[:100]}...': {e}", exc_info=True)
        return BashCommandOutput(
            stdout="", 
            stderr=str(e), 
            exit_code=-2 # Indicate general execution error
        )

if __name__ == "__main__":
    logger.info("Starting FastMCP Bash Server with mcp.run()...")
    try:
        mcp.run() # This will handle stdio and serve the defined tools
    except KeyboardInterrupt:
        logger.info("FastMCP Bash Server shutting down (KeyboardInterrupt).")
    except Exception as e:
        logger.exception(f"FastMCP Bash Server failed to run: {e}")
