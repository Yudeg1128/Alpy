import grpc
import logging
import sys
import os

from langchain.tools import Tool

# Adjust Python path to find mcp_servers. Assuming Alpy root is one level up from src.
# This might need refinement depending on how you run the main Alpy script.
alpy_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if alpy_root not in sys.path:
    sys.path.insert(0, alpy_root)

try:
    from mcp_servers.protos import bash_service_pb2
    from mcp_servers.protos import bash_service_pb2_grpc
except ImportError:
    print("ERROR: Cannot import MCP server protos. Ensure 'mcp_servers' directory is in the Python path.")
    print(f"Current sys.path includes: {sys.path}")
    # Optionally re-raise or exit if this is critical
    raise

# Use relative import for config now that we are inside the 'src' package
from .. import config

logger = logging.getLogger(__name__)


def _run_bash_command(command: str) -> str:
    """Connects to the Bash MCP server and executes a command."""
    logger.info(f"Attempting to execute bash command via MCP: {command}")
    address = config.BASH_MCP_SERVER_ADDRESS
    try:
        with grpc.insecure_channel(address) as channel:
            stub = bash_service_pb2_grpc.BashExecutorStub(channel)
            request = bash_service_pb2.CommandRequest(command=command)
            # Add a timeout to the gRPC call (e.g., 65 seconds, slightly more than server timeout)
            reply = stub.ExecuteCommand(request, timeout=65)
            logger.info(f"Bash MCP reply received. RC: {reply.return_code}")
            # Format output clearly
            output = f"Return Code: {reply.return_code}\n"
            if reply.stdout:
                output += f"STDOUT:\n---\n{reply.stdout.strip()}\n---\n"
            if reply.stderr:
                output += f"STDERR:\n---\n{reply.stderr.strip()}\n---\n"
            return output
    except grpc.RpcError as e:
        status_code = e.code()
        logger.error(f"gRPC Error connecting to Bash MCP server at {address}: {status_code} - {e.details()}")
        if status_code == grpc.StatusCode.UNAVAILABLE:
             return f"Error: Bash execution server is unavailable at {address}. Is it running?"
        elif status_code == grpc.StatusCode.DEADLINE_EXCEEDED:
             return f"Error: Request to Bash execution server timed out. The command might be taking too long."
        else:
             return f"Error: Could not communicate with the Bash server. Details: {status_code.name} - {e.details()}"
    except Exception as e:
        logger.error(f"Unexpected error in bash tool: {e}", exc_info=True)
        return f"Error: An unexpected error occurred in the bash tool: {e}"


bash_tool = Tool(
    name="bash_executor",
    func=_run_bash_command,
    description=(
        "Executes a non-interactive bash command on the local Linux system and returns the stdout, stderr, and return code. "
        "Use this for file system operations (ls, pwd, cat, mkdir, rm), checking processes, running simple scripts, or getting system info. "
        "Input MUST be a single string containing the complete bash command. "
        "Example: `ls -l /home/user` or `cat my_file.txt`. "
        "Do NOT ask the user for confirmation before using `rm` or other destructive commands unless absolutely necessary and clearly stated."
    ),
    return_direct=False, # Agent processes the output
)

# Example usage (for testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Ensure the bash_server.py is running on localhost:50051 first
    print("--- Running Bash Tool Test --- (Ensure bash_server.py is running!)")

    test_command = "echo 'Hello from Bash MCP!' && ls -l *.py"
    print(f"\nTesting command: {test_command}")
    result = bash_tool.run(test_command)
    print("\nResult:")
    print(result)

    test_command_error = "ls /non_existent_directory"
    print(f"\nTesting error command: {test_command_error}")
    result_err = bash_tool.run(test_command_error)
    print("\nResult (Error):")
    print(result_err)

    print("\n--- Test Complete ---")
