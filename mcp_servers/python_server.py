from concurrent import futures
import grpc
import time
import logging
import io
import contextlib
import traceback

# Import the generated classes
from protos import python_service_pb2
from protos import python_service_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PythonExecutorServicer(python_service_pb2_grpc.PythonExecutorServicer):
    """Provides methods that implement functionality of python executor server."""

    def ExecuteCode(self, request, context):
        """Executes a Python code snippet and returns its output."""
        code = request.code
        logger.info(f"Executing Python code: {code[:200]}...")
        
        # Create dedicated stdout and stderr streams
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        return_code = 0
        
        try:
            # Redirect stdout and stderr using nested with statements
            with contextlib.redirect_stdout(stdout_capture):
                with contextlib.redirect_stderr(stderr_capture):
                    # Execute the code in a restricted scope
                    exec(code, {'__builtins__': __builtins__}, {})
        except Exception as e:
            # If exec itself fails or code within exec raises an unhandled exception
            # Ensure proper indentation here
            tb_str = traceback.format_exc()
            stderr_capture.write(tb_str) # Add traceback to stderr
            logger.error(f"Error during Python code execution: {e}\n{tb_str}")
            return_code = 1 # Indicate an error during execution
        finally:
            stdout_result = stdout_capture.getvalue()
            stderr_result = stderr_capture.getvalue()
            logger.info(f"Python code finished. RC: {return_code}, STDOUT: {stdout_result[:100]}, STDERR: {stderr_result[:100]}")

        return python_service_pb2.CodeReply(
            stdout=stdout_result,
            stderr=stderr_result,
            return_code=return_code
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    python_service_pb2_grpc.add_PythonExecutorServicer_to_server(PythonExecutorServicer(), server)
    port = "50052" # Different port from bash server
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logger.info(f"Python MCP Server started on port {port}")
    try:
        while True:
            time.sleep(86400)  # One day in seconds
    except KeyboardInterrupt:
        logger.info("Python MCP Server shutting down.")
        server.stop(0)

if __name__ == '__main__':
    serve()
