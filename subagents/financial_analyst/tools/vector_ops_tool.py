"""
Vector Operations Tool

Perform operations on image embedding vector stores like similarity search and clustering.
"""

import asyncio
import json
import logging
import os
import shutil
import sys
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Type, Union, Literal
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, field_validator, PrivateAttr, RootModel
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import Implementation as MCPImplementation

logger = logging.getLogger(__name__)

def store_operation_result(security_id: str, operation: str, result: Dict[str, Any], vector_store_type: str) -> str:
    """Store operation result in a JSON file, in the correct subfolder for image/text ops.
    
    Args:
        security_id: The security ID the operation was performed on
        operation: The type of operation (search_similar, batch_search, cluster)
        result: The operation result to store
        vector_store_type: 'image' or 'text'
    Returns:
        Path to the stored result file
    """
    # Get vector store path from security ID and type
    base_dir = Path(f"/home/me/CascadeProjects/Alpy/otcmn_tool_test_output/current/secondary_board_B/{security_id}")
    results_dir = base_dir / "operation_results" / vector_store_type
    results_dir.mkdir(parents=True, exist_ok=True)
    operation_map = {
        'search_similar': 'search',
        'batch_search': 'batch_search',
        'cluster': 'cluster'
    }
    file_prefix = operation_map.get(operation, operation)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = results_dir / f"{file_prefix}_{timestamp}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return str(result_file)

# --- Inner Action Models ---
class InnerVectorSearchAction(BaseModel):
    action: Literal["search_similar"] = Field(default="search_similar", frozen=True)
    security_id: str = Field(description="Security identifier")
    query: str = Field(description="Text query to search for in the documents")
    k: Optional[int] = Field(default=5, description="Number of similar sections to return")

class InnerVectorBatchSearchAction(BaseModel):
    action: Literal["batch_search"] = Field(default="batch_search", frozen=True)
    security_id: str = Field(description="Security identifier")
    queries: List[str] = Field(description="List of text queries to search for")
    k: Optional[int] = Field(default=5, description="Number of similar sections to return per query")

class InnerVectorClusterAction(BaseModel):
    action: Literal["cluster"] = Field(default="cluster", frozen=True)
    security_id: str = Field(description="Security identifier")
    n_clusters: Optional[int] = Field(default=10, description="Number of clusters to create")

# --- Main Input Schema ---
class VectorOpsActionInput(BaseModel):
    security_id: str = Field(description="Security identifier")
    vector_store_type: Literal["image", "text"] = Field(description="Type of vector store to operate on: 'image' or 'text'")
    action: Literal["search_similar", "batch_search", "cluster"] = Field(
        description="Type of operation to perform"
    )
    # Fields for search_similar
    query: Optional[str] = Field(
        default=None,
        description="Text query to search for in the documents (for search_similar)"
    )
    # Fields for batch_search
    queries: Optional[List[str]] = Field(
        default=None,
        description="List of queries for batch search (for batch_search)"
    )
    # Shared fields
    k: Optional[int] = Field(
        default=5,
        description="Number of results to return (for search_similar and batch_search)"
    )
    # Fields for cluster
    n_clusters: Optional[int] = Field(
        default=10,
        description="Number of clusters (for cluster operation)"
    )
    
    @field_validator('query')
    def validate_query(cls, v, info):
        if info.data.get('action') == 'search_similar' and not v:
            raise ValueError('query is required for search_similar action')
        return v
    
    @field_validator('queries')
    def validate_queries(cls, v, info):
        if info.data.get('action') == 'batch_search':
            if not v:
                raise ValueError('queries is required for batch_search action')
            if not isinstance(v, list) or not all(isinstance(q, str) for q in v):
                raise ValueError('queries must be a list of strings')
        return v
    
    def __getattr__(self, item: str) -> Any:
        try:
            root_obj = object.__getattribute__(self, 'root')
        except AttributeError:
            return super().__getattr__(item)
        if hasattr(root_obj, 'model_fields') and item in root_obj.model_fields:
            return getattr(root_obj, item)
        return super().__getattr__(item)

class VectorOpsTool(BaseTool, BaseModel):
    name: str = "vector_ops"

    class Config:
        arbitrary_types_allowed = True
    description: str = (
        "Tool for semantic search and analysis of financial documents.\n\n"
        "Each security (identified by security_id) has two types of vector stores: 'image' and 'text'.\n"
        "You must specify which type to operate on using the 'vector_store_type' argument.\n\n"
        "Available Actions (for both 'image' and 'text' vector stores):\n"
        "1. search_similar: Find document/image sections matching your query\n"
        "   Required: security_id, vector_store_type, action='search_similar', query\n"
        "2. batch_search: Search multiple queries at once\n"
        "   Required: security_id, vector_store_type, action='batch_search', queries\n"
        "3. cluster: Cluster document/image embeddings\n"
        "   Required: security_id, vector_store_type, action='cluster', n_clusters (optional)\n\n"
        "Arguments:\n"
        "  - security_id: The security identifier (e.g., 'MN0LNDB68390')\n"
        "  - vector_store_type: 'image' or 'text' (MUST be specified)\n"
        "  - action: One of 'search_similar', 'batch_search', 'cluster'\n"
        "  - query: Search query string (for search_similar)\n"
        "  - queries: List of query strings (for batch_search)\n"
        "  - n_clusters: Number of clusters (for cluster, default=10)\n\n"
        "Examples:\n"
        "  Single search (text):\n"
        "    {\n"
        "      'security_id': 'MN0LNDB68390',\n"
        "      'vector_store_type': 'text',\n"
        "      'action': 'search_similar',\n"
        "      'query': 'What is the ISIN?'\n"
        "    }\n"
        "  Batch search (image):\n"
        "    {\n"
        "      'security_id': 'MN0LNDB68390',\n"
        "      'vector_store_type': 'image',\n"
        "      'action': 'batch_search',\n"
        "      'queries': ['Revenue?', 'Key risks?']\n"
        "    }\n"
        "  Clustering (text):\n"
        "    {\n"
        "      'security_id': 'MN0LNDB68390',\n"
        "      'vector_store_type': 'text',\n"
        "      'action': 'cluster',\n"
        "      'n_clusters': 8\n"
        "    }\n"
        "\nReturns: JSON with operation status, result file path, and results.\n"
    )
    args_schema: Type[BaseModel] = VectorOpsActionInput
    return_direct: bool = False
    _session: ClientSession = PrivateAttr(default=None)
    _logger_instance: logging.Logger = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger_instance = logger

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError(f"{self.name} is async-native. Use `_arun` or `ainvoke`.")

    async def _arun(self, **kwargs):
        # Require explicit vector_store_type
        vector_store_type = kwargs.get('vector_store_type', None)
        if vector_store_type not in ('image', 'text'):
            raise ValueError("You must specify vector_store_type as either 'image' or 'text'. Got: {}".format(vector_store_type))
        root = None
        if kwargs.get('action') == 'search_similar':
            root = InnerVectorSearchAction(
                security_id=kwargs['security_id'],
                query=kwargs['query'],
                k=kwargs.get('k', 5)
            )
        elif kwargs.get('action') == 'batch_search':
            root = InnerVectorBatchSearchAction(
                security_id=kwargs['security_id'],
                queries=kwargs['queries'],
                k=kwargs.get('k', 5)
            )
        elif kwargs.get('action') == 'cluster':
            root = InnerVectorClusterAction(
                security_id=kwargs['security_id'],
                n_clusters=kwargs.get('n_clusters', 10)
            )
        else:
            raise ValueError(f"Invalid action: {kwargs.get('action')}")
        # Route to the correct MCP server
        server_script = Path(__file__).parent.parent / "mcp_servers" / "mcp_vector_ops" / "server.py"
        if not server_script.exists():
            raise RuntimeError(f"MCP server script not found at {server_script}")
        server_path = server_script.parent
        env = dict(os.environ)
        env["PYTHONPATH"] = str(server_path.parent.parent)
        params = StdioServerParameters(
            command="python3",
            args=[str(server_script)],
            cwd=str(server_path),
            env=env
        )
        client_info = MCPImplementation(
            name="VectorOpsToolClient",
            version="0.1.0"
        )
        try:
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream, client_info=client_info) as session:
                    await session.initialize()
                    model_data = root.model_dump()
                    # Patch: always include vector_store_type from kwargs
                    model_data['vector_store_type'] = vector_store_type
                    if 'action' in model_data:
                        model_data['operation'] = model_data.pop('action')
                    print(f"[_arun] Sending model_data to MCP server: {json.dumps(model_data, indent=2)}")
                    tool_arguments = {"input_data": model_data}
                    tool_result = await session.call_tool(
                        name="vector_ops", 
                        arguments=tool_arguments
                    )
                    if tool_result.isError:
                        error_msg = "Unknown error"
                        if tool_result.content and len(tool_result.content) > 0:
                            error_msg = tool_result.content[0].text
                        return json.dumps({
                            "status": "error", 
                            "error": error_msg
                        })
                    if tool_result.content and len(tool_result.content) > 0:
                        try:
                            result = json.loads(tool_result.content[0].text)
                            if not result:
                                print(f"[_arun] MCP server returned empty or null result: {tool_result.content[0].text}")
                                return json.dumps({
                                    "status": "error",
                                    "error": "MCP server returned null result",
                                    "raw_response": tool_result.content[0].text
                                })
                            if 'result' in result and result['result'] and 'metadata' in result['result']:
                                del result['result']['metadata']
                            operation = kwargs.get('action') if 'action' in kwargs else kwargs.get('operation', 'unknown')
                            if operation == 'batch_search' and 'queries' in kwargs:
                                result['queries'] = kwargs['queries']
                            security_id = kwargs.get('security_id')
                            result_file = store_operation_result(security_id, operation, result, vector_store_type)
                            result['result_file'] = result_file
                            return json.dumps(result)
                        except json.JSONDecodeError:
                            print(f"[_arun] Invalid JSON response from MCP server: {tool_result.content[0].text}")
                            return json.dumps({
                                "status": "error",
                                "error": "Invalid JSON response from server",
                                "raw_response": tool_result.content[0].text
                            })
                    return json.dumps({
                        "status": "error",
                        "error": "Empty response from server"
                    })                    
        except Exception as e:
            self._logger_instance.error(f"Error in vector_ops tool: {e}", exc_info=True)
            raise
            
    async def close(self):
        """Close any open resources."""
        if hasattr(self, '_session') and self._session:
            try:
                await self._session.close()
                self._session = None
            except Exception as e:
                self._logger_instance.error(f"Error closing session: {e}")
                raise

# --- Example Usage (for standalone testing) ---
async def test_cluster_action(security_id: str, vector_store_type: str):
    print(f"[test_cluster_action] Called with security_id={security_id}, vector_store_type={vector_store_type}")
    tool = VectorOpsTool()
    try:
        result_json = await tool._arun(
            security_id=security_id,
            vector_store_type=vector_store_type,
            action="cluster",
            n_clusters=5
        )
        result = json.loads(result_json)
        print(f"Clustering result: {result}")
    finally:
        await tool.close()

async def test_search_similar_action(security_id: str, vector_store_type: str):
    print(f"[test_search_similar_action] Called with security_id={security_id}, vector_store_type={vector_store_type}")
    tool = VectorOpsTool()
    try:
        result_json = await tool._arun(
            security_id=security_id,
            vector_store_type=vector_store_type,
            action="search_similar",
            query="What are the main revenue streams?",
            k=3
        )
        result = json.loads(result_json)
        print(f"Search result: {result}")
    finally:
        await tool.close()

async def test_batch_search_action(security_id: str, vector_store_type: str):
    print(f"[test_batch_search_action] Called with security_id={security_id}, vector_store_type={vector_store_type}")
    tool = VectorOpsTool()
    try:
        result_json = await tool._arun(
            security_id=security_id,
            vector_store_type=vector_store_type,
            action="batch_search",
            queries=[
                "What are the main revenue streams?",
                "What are the key risk factors?"
            ],
            k=3
        )
        result = json.loads(result_json)
        print(f"Batch search result: {result}")
    finally:
        await tool.close()

async def main_test_vector_ops_tool():
    # Configure logging
    log_level_str = os.getenv("LOG_LEVEL_TOOL_TEST", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s [%(threadName)s] - %(message)s',
            stream=sys.stdout
        )
    test_logger = logging.getLogger(__name__ + ".VectorOpsTool_Test")
    test_logger.setLevel(log_level)
    
    test_logger.info("Starting VectorOpsTool standalone test...")
    test_logger.warning("Ensure the relevant MCP server is in mcp_servers/ and a vector store exists for testing.")
    
    try:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--test-cluster", action="store_true", help="Run clustering test")
        parser.add_argument("--test-search", action="store_true", help="Run single vector search test")
        parser.add_argument("--test-batch", action="store_true", help="Run batch search test")
        parser.add_argument("--test-all", action="store_true", help="Run all tests")
        parser.add_argument("--security-id", type=str, default="MN0LNDB68390", help="Security ID to test with")
        parser.add_argument("--vector-store-type", type=str, default="text", choices=["image", "text"], help="Vector store type: image or text")
        args = parser.parse_args()
        
        security_id = args.security_id
        vector_store_type = args.vector_store_type
        print(f"[main_test_vector_ops_tool] CLI parsed: security_id={security_id}, vector_store_type={vector_store_type}")
        
        # Default to all tests if no specific test is selected
        if not (args.test_cluster or args.test_search or args.test_batch or args.test_all):
            args.test_all = True
        
        if args.test_all or args.test_cluster:
            print("\n--- [Test Case 1: Clustering] ---")
            await test_cluster_action(security_id, vector_store_type)
        
        if args.test_all or args.test_search:
            print("\n--- [Test Case 2: Single Vector Search] ---")
            await test_search_similar_action(security_id, vector_store_type)
        
        if args.test_all or args.test_batch:
            print("\n--- [Test Case 3: Batch Search] ---")
            await test_batch_search_action(security_id, vector_store_type)
            
    except Exception as e:
        test_logger.error(f"Error during VectorOpsTool test: {e}", exc_info=True)
    finally:
        test_logger.info("VectorOpsTool standalone test finished.")



if __name__ == "__main__":
    asyncio.run(main_test_vector_ops_tool())
