import asyncio
import logging
import subprocess
import sys

from pydantic import BaseModel, Field

class PipInstallInput(BaseModel):
    package: str = Field(description="The name of the package to install, e.g., 'requests' or 'numpy'.")
    version: str | None = Field(default=None, description="Optional version specifier, e.g., '1.2.3'.")

class PipInstallOutput(BaseModel):
    stdout: str
    stderr: str
    exit_code: int

from mcp.server.fastmcp import FastMCP

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s][FastMCPPythonServer] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

mcp = FastMCP()

class PythonCommandInput(BaseModel):
    code: str = Field(description="The Python code snippet to execute.")
    timeout: float | None = Field(default=60.0, description="Timeout in seconds for the Python code execution.")

class PythonCommandOutput(BaseModel):
    stdout: str
    stderr: str
    exit_code: int

def execute_python_code_sync(code: str, timeout: float | None = 60.0) -> PythonCommandOutput:
    """Synchronously executes Python code using a subprocess."""
    logger.info(f"Executing Python code (first 100 chars): '{code[:100]}{'...' if len(code) > 100 else ''}' with timeout {timeout}s")
    try:
        process = subprocess.run(
            [sys.executable, "-c", code], # sys.executable ensures we use the same Python interpreter
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False, # We handle the exit code manually
        )
        stdout = process.stdout
        stderr = process.stderr
        exit_code = process.returncode
        logger.info(f"Python code execution finished. Exit code: {exit_code}")
        return PythonCommandOutput(stdout=stdout, stderr=stderr, exit_code=exit_code)
    except subprocess.TimeoutExpired:
        logger.error(f"Python code '{code[:100]}...' timed out after {timeout} seconds.")
        return PythonCommandOutput(
            stdout="",
            stderr=f"Execution timed out after {timeout} seconds.",
            exit_code=-1 # Using -1 to indicate timeout for Python execution
        )
    except Exception as e:
        logger.error(f"Error executing Python code '{code[:100]}...': {e}", exc_info=True)
        return PythonCommandOutput(
            stdout="",
            stderr=str(e),
            exit_code=-2 # Using -2 for other general execution errors
        )

@mcp.tool(
    name="execute_python_code",
    description="Executes a Python code snippet and returns its stdout, stderr, and exit code.",
    # input_model and output_model are inferred from type hints by FastMCP
)
async def execute_python_code_tool(input: PythonCommandInput) -> PythonCommandOutput:
    """MCP tool to execute Python code."""
    # Run the synchronous function in a thread pool to avoid blocking the asyncio event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, # Uses the default_executor (a ThreadPoolExecutor)
        execute_python_code_sync,
        input.code,
        input.timeout
    )
    return result

@mcp.tool(
    name="install_python_package",
    description="Installs a Python package using pip in the current Python environment (sys.executable). Accepts a package name and optional version.",
)
async def install_python_package_tool(input: PipInstallInput) -> PipInstallOutput:
    """MCP tool to install a Python package using the correct Python environment."""
    package_spec = input.package if not input.version else f"{input.package}=={input.version}"
    logger.info(f"Installing Python package: {package_spec}")
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pip", "install", package_spec,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await process.communicate()
        stdout = stdout_bytes.decode()
        stderr = stderr_bytes.decode()
        exit_code = process.returncode
        logger.info(f"pip install finished. Exit code: {exit_code}")
        return PipInstallOutput(stdout=stdout, stderr=stderr, exit_code=exit_code)
    except Exception as e:
        logger.error(f"Error during pip install: {e}", exc_info=True)
        return PipInstallOutput(stdout="", stderr=str(e), exit_code=-2)

if __name__ == "__main__":

    logger.info("Starting FastMCP Python Server with mcp.run()...")
    try:
        mcp.run() # This will block and run the server
    except KeyboardInterrupt:
        logger.info("FastMCP Python Server shutting down due to KeyboardInterrupt.")
    except Exception as e:
        logger.critical(f"FastMCP Python Server exited with critical error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("FastMCP Python Server has stopped.")