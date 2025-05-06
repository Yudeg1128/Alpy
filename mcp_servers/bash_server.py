from concurrent import futures
import subprocess
import grpc
import time
import logging

# Import the generated classes
from protos import bash_service_pb2
from protos import bash_service_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BashExecutorServicer(bash_service_pb2_grpc.BashExecutorServicer):
    """Provides methods that implement functionality of bash executor server."""

    def ExecuteCommand(self, request, context):
        """Executes a bash command and returns its output."""
        command = request.command
        logger.info(f"Executing bash command: {command}")
        try:
            process = subprocess.Popen(
                command, 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate(timeout=60) # 60-second timeout
            return_code = process.returncode
            logger.info(f"Command finished. RC: {return_code}, STDOUT: {stdout[:100]}, STDERR: {stderr[:100]}")
            return bash_service_pb2.CommandReply(stdout=stdout, stderr=stderr, return_code=return_code)
        except subprocess.TimeoutExpired:
            logger.error(f"Command '{command}' timed out.")
            process.kill()
            stdout, stderr = process.communicate()
            return bash_service_pb2.CommandReply(
                stdout=stdout,
                stderr="Error: Command timed out after 60 seconds.",
                return_code=-1 # Custom error code for timeout
            )
        except Exception as e:
            logger.error(f"Error executing command '{command}': {e}")
            return bash_service_pb2.CommandReply(stdout="", stderr=str(e), return_code=-2) # Custom error code for other errors

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    bash_service_pb2_grpc.add_BashExecutorServicer_to_server(BashExecutorServicer(), server)
    port = "50051"
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logger.info(f"Bash MCP Server started on port {port}")
    try:
        while True:
            time.sleep(86400)  # One day in seconds
    except KeyboardInterrupt:
        logger.info("Bash MCP Server shutting down.")
        server.stop(0)

if __name__ == '__main__':
    serve()
