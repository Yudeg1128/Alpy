"""MCP Server for vector store operations."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Literal
import numpy as np
import faiss
from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP
import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel

# Initialize Google Cloud
project_id = "alpy-461606"
location = "us-central1"

# Re-initialize Vertex AI with fresh application default credentials
vertexai.init(
    project=project_id,
    location=location,
    credentials=None  # Force refresh of application default credentials
)

# --- Configuration ---
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more details
    format='[%(asctime)s][%(levelname)s][MCPVectorOps] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mcp_vector_ops.log')
    ]
)
logger = logging.getLogger("MCPVectorOps")
logger.info("Starting MCP Vector Ops server")

# --- Constants ---
VECTOR_STORE_NAME = "vector_store"
VECTOR_DIMENSION = 1408  # MultiModal embedding dimension

# --- Pydantic Models ---
class VectorOpsInput(BaseModel):
    operation: Literal["search_similar", "batch_search", "cluster"]
    security_id: str
    query: Optional[str] = None  # Text query for search_similar
    queries: Optional[List[str]] = None  # Text queries for batch_search
    k: Optional[int] = 5
    n_clusters: Optional[int] = 10

class VectorOpsOutput(BaseModel):
    status: str
    message: str
    result: Optional[Dict] = None

def get_security_path(security_id: str) -> Path:
    """Get path to security directory."""
    base_dir = Path('/home/me/CascadeProjects/Alpy/otcmn_tool_test_output/current')
    # Look in all board directories
    for board in base_dir.iterdir():
        if not board.is_dir():
            continue
        security_path = board / security_id
        if security_path.exists():
            return security_path
    raise ValueError(f"Security {security_id} not found in any board under {base_dir}")

def load_vector_store(security_id: str) -> tuple[Any, Dict]:
    """Load vector store for security."""
    security_path = get_security_path(security_id)
    store_dir = security_path / VECTOR_STORE_NAME
    
    if not store_dir.exists():
        raise ValueError(f"No vector store found for security {security_id} at {store_dir}")
        
    try:
        # Load index
        index_path = store_dir / "index.faiss"
        if not index_path.exists():
            raise ValueError(f"No FAISS index found at {index_path}")
        index = faiss.read_index(str(index_path))
        logger.debug(f"Loaded FAISS index for {security_id}. Index dimension (index.d): {index.d}, Total vectors (index.ntotal): {index.ntotal}")
        
        # Load metadata
        metadata_path = store_dir / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"No metadata found at {metadata_path}")
        with open(metadata_path) as f:
            metadata = json.load(f)
            
        return index, metadata
    except Exception as e:
        raise RuntimeError(f"Failed to load vector store for {security_id}: {e}")

def search_similar(index: Any, query_vector: List[float], k: int = 5) -> Dict:
    """Find k most similar images to query vector."""
    query = np.array([query_vector], dtype=np.float32)
    distances, indices = index.search(query, k)
    return {
        "distances": distances[0].tolist(),
        "indices": indices[0].tolist()
    }

def batch_search(index: Any, query_vectors: List[List[float]], k: int = 5) -> Dict:
    """Search multiple query vectors at once."""
    queries = np.array(query_vectors, dtype=np.float32)
    distances, indices = index.search(queries, k)
    return {
        "distances": distances.tolist(),
        "indices": indices.tolist()
    }

def cluster_images(index: Any, n_clusters: int = 10) -> Dict:
    """Cluster images into groups using k-means."""
    # Get all vectors
    vectors = np.array([index.reconstruct(i) for i in range(index.ntotal)], dtype=np.float32)
    
    # Create and train kmeans
    kmeans = faiss.Kmeans(vectors.shape[1], n_clusters)
    kmeans.train(vectors)
    
    # Get cluster assignments
    _, assignments = kmeans.index.search(vectors, 1)
    
    # Group by cluster
    clusters = [[] for _ in range(n_clusters)]
    for idx, cluster_id in enumerate(assignments):
        clusters[cluster_id[0]].append(idx)
        
    # Find centroids
    centroids = []
    for cluster in clusters:
        if not cluster:
            continue
        # Get vectors for cluster
        cluster_vectors = np.array([index.reconstruct(i) for i in cluster], dtype=np.float32)
        # Compute mean vector
        mean = cluster_vectors.mean(axis=0, keepdims=True)
        # Find closest to mean
        distances, _ = index.search(mean, 1)
        centroids.append(cluster[0])  # Use first image as representative
        
    return {
        "clusters": clusters,
        "centroids": centroids
    }

# --- MCP App ---
mcp_app = FastMCP(
    name="MCPVectorOpsServer",
    version="1.0.0",
    description="MCP server for vector operations like similarity search and clustering"
)

@mcp_app.tool(name="vector_ops")
async def vector_ops(input_data: VectorOpsInput) -> VectorOpsOutput:
    try:
        logger.debug(f"Received vector_ops request: {input_data.model_dump_json(indent=2)}")
        # Load vector store
        index, metadata = load_vector_store(input_data.security_id)
        
        # Get embeddings using Vertex AI
        model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
        
        # Perform operation
        if input_data.operation == "search_similar":
            if not input_data.query:
                raise ValueError("search_similar requires a query")
            # Convert query to vector
            result = model.get_embeddings(contextual_text=input_data.query)
            query_vector = result.text_embedding
            result = search_similar(index, query_vector, input_data.k)
            message = f"Found {input_data.k} similar items for query '{input_data.query}' in security {input_data.security_id}"
            
        elif input_data.operation == "batch_search":
            if not input_data.queries:
                raise ValueError("batch_search requires queries")
            # Convert queries to vectors
            query_vectors = []
            for query in input_data.queries:
                result = model.get_embeddings(contextual_text=query)
                query_vectors.append(result.text_embedding)
            result = batch_search(index, query_vectors, input_data.k)
            message = f"Found {input_data.k} similar items for {len(input_data.queries)} queries in security {input_data.security_id}"
            
        elif input_data.operation == "cluster":
            result = cluster_images(index, input_data.n_clusters)
            message = f"Created {len(result['clusters'])} clusters for security {input_data.security_id}"
            
        else:
            raise ValueError(f"Invalid operation: {input_data.operation}")
            
        # Add metadata to result
        if isinstance(result, dict):
            result["metadata"] = metadata
        else: # Should not happen with current logic, but good for safety
            result = {"data": result, "metadata": metadata}
            
        return VectorOpsOutput(
            status="success",
            message=message,
            result=result
        )
        
    except Exception as e:
        error_message = str(e) if str(e) else repr(e)
        logger.error(f"Vector ops error for security {input_data.security_id} with operation {input_data.operation}: {error_message}", exc_info=True)
        return VectorOpsOutput(
            status="error",
            message=error_message
        )

async def run_stdio_server():
    await mcp_app.run_stdio_async()

def main():
    # Initialize logging
    logger.info("Starting MCP Vector Ops server")

    try:
        # Initialize MCP app
        logger.debug("Creating MCPVectorOpsServer instance")
        import asyncio
        asyncio.run(run_stdio_server())
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
