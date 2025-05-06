import grpc
import logging
import sys
import os

from langchain.tools import Tool

# Adjust Python path to find mcp_servers. Assuming Alpy root is one level up from src.
alpy_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if alpy_root not in sys.path:
    sys.path.insert(0, alpy_root)

try:
    from mcp_servers.protos import python_service_pb2
    from mcp_servers.protos import python_service_pb2_grpc
except ImportError:
    print("ERROR: Cannot import MCP server protos. Ensure 'mcp_servers' directory is in the Python path.")
    print(f"Current sys.path includes: {sys.path}")
    raise

# Use relative import for config now that we are inside the 'src' package
from .. import config

logger = logging.getLogger(__name__)


def _run_python_code(code: str) -> str:
    """Connects to the Python MCP server and executes a code snippet."""
    logger.info(f"Attempting to execute python code via MCP: {code[:100]}...")
    address = config.PYTHON_MCP_SERVER_ADDRESS
    try:
        with grpc.insecure_channel(address) as channel:
            stub = python_service_pb2_grpc.PythonExecutorStub(channel)
            request = python_service_pb2.CodeRequest(code=code)
            # Use a reasonable timeout for potentially longer Python scripts
            reply = stub.ExecuteCode(request, timeout=120) # 120 seconds timeout
            logger.info(f"Python MCP reply received. RC: {reply.return_code}")
            # Format output clearly
            output = f"Return Code: {reply.return_code}\n"
            if reply.stdout:
                output += f"STDOUT:\n---\n{reply.stdout.strip()}\n---\n"
            # Include stderr even if return code is 0, as warnings might be printed
            if reply.stderr:
                 output += f"STDERR:\n---\n{reply.stderr.strip()}\n---\n"
            # Add a note if execution failed based on return code
            if reply.return_code != 0:
                output += "Note: Execution failed (non-zero return code or internal error).\n"
            return output
    except grpc.RpcError as e:
        status_code = e.code()
        logger.error(f"gRPC Error connecting to Python MCP server at {address}: {status_code} - {e.details()}")
        if status_code == grpc.StatusCode.UNAVAILABLE:
             return f"Error: Python execution server is unavailable at {address}. Is it running?"
        elif status_code == grpc.StatusCode.DEADLINE_EXCEEDED:
             return f"Error: Request to Python execution server timed out. The script might be taking too long (limit 120s)."
        else:
             return f"Error: Could not communicate with the Python server. Details: {status_code.name} - {e.details()}"
    except Exception as e:
        logger.error(f"Unexpected error in python tool: {e}", exc_info=True)
        return f"Error: An unexpected error occurred in the python tool: {e}"


python_tool = Tool(
    name="python_executor",
    func=_run_python_code,
    description=(
        "Executes a non-interactive Python code snippet on the local system and returns the stdout, stderr, and return code. "
        "Useful for calculations, data manipulation, or tasks requiring Python logic. "
        "Input MUST be a single string containing the complete Python code snippet. The code runs in a restricted environment. "
        "Example: `print(1 + 1)` or `import datetime; print(datetime.date.today())`. "
        "Do NOT attempt to use input(), GUI libraries, or long-running background tasks."
    ),
    return_direct=False, # Agent processes the output
)

# Example usage (for testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Ensure the python_server.py is running on localhost:50052 first
    print("--- Running Python Tool Test --- (Ensure python_server.py is running!)")

    test_code = "import platform\nprint(f'Hello from Python MCP on {platform.system()}!')\nprint(10 / 2)"
    print(f"\nTesting code:\n{test_code}")
    result = python_tool.run(test_code)
    print("\nResult:")
    print(result)

    test_code_error = "print(1/0)"
    print(f"\nTesting error code:\n{test_code_error}")
    result_err = python_tool.run(test_code_error)
    print("\nResult (Error):")
    print(result_err)

    print("\n--- Test Complete ---")
