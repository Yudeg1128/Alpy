"""MCP Server for vector store operations."""

import json
import logging
import os
from pathlib import Path
from financial_analyst.security_folder_utils import require_security_folder, get_subfolder, get_security_file
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
TEXT_EMBEDDING_DIMENSION = 768
IMAGE_EMBEDDING_DIMENSION = 768 # Common dimension for image part of multimodal embeddings

# --- Pydantic Models ---
class VectorOpsInput(BaseModel):
    operation: Literal["search_similar", "batch_search", "cluster"]
    security_id: str
    vector_store_type: Literal["image", "text"] = "image"
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
    return require_security_folder(security_id)

def load_vector_store(security_id: str, vector_store_type: str) -> tuple[Any, Dict]:
    """Load vector store for security."""
    security_path = get_security_path(security_id)
    if vector_store_type == "image":
        store_dir = get_subfolder(security_id, "vector_store_image")
    elif vector_store_type == "text":
        store_dir = get_subfolder(security_id, "vector_store_txt")
    else:
        raise ValueError(f"Unknown vector_store_type: {vector_store_type}")
    
    if not store_dir.exists():
        raise ValueError(f"No vector store found for security {security_id} at {store_dir}")
        
    try:
        # Load index (support both index.faiss and <security_id>_faiss.index)
        index_path1 = store_dir / "index.faiss"
        index_path2 = store_dir / f"{security_id}_faiss.index"
        if index_path1.exists():
            index_path = index_path1
        elif index_path2.exists():
            index_path = index_path2
        else:
            raise ValueError(f"No FAISS index found at {index_path1} or {index_path2}")
        index = faiss.read_index(str(index_path))
        logger.debug(f"Loaded FAISS index for {security_id} at {index_path}. Index dimension (index.d): {index.d}, Total vectors (index.ntotal): {index.ntotal}")
        
        # Load metadata (support both metadata.json and <security_id>_vector_metadata.json)
        metadata_path1 = store_dir / "metadata.json"
        metadata_path2 = store_dir / f"{security_id}_vector_metadata.json"
        if metadata_path1.exists():
            metadata_path = metadata_path1
        elif metadata_path2.exists():
            metadata_path = metadata_path2
        else:
            raise ValueError(f"No metadata found at {metadata_path1} or {metadata_path2}")
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Validate index dimension
        expected_d = -1
        if vector_store_type == "text":
            expected_d = TEXT_EMBEDDING_DIMENSION
        elif vector_store_type == "image":
            expected_d = IMAGE_EMBEDDING_DIMENSION

        if expected_d != -1 and index.d != expected_d:
            logger.error(f"FAISS index dimension mismatch for '{vector_store_type}' store at {index_path}. Expected {expected_d}, found {index.d}.")
            raise ValueError(
                f"FAISS index dimension mismatch for '{vector_store_type}' store. "
                f"Expected {expected_d}, but found {index.d} in {index_path}. "
                f"The index may need to be rebuilt with the correct embeddings."
            )
        
        logger.info(f"Successfully loaded '{vector_store_type}' FAISS index for {security_id} from {index_path} with dimension {index.d}.")
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

def batch_search(index: Any, query_vectors: List[List[float]], k: int = 5, expected_dim: int = None) -> Dict:
    """Search multiple query vectors at once."""
    queries = np.array(query_vectors, dtype=np.float32)
    if queries.size == 0:
        raise ValueError("No query vectors provided for batch_search.")
    target_dim = expected_dim if expected_dim is not None else index.d
    if len(queries.shape) != 2 or queries.shape[1] != target_dim:
        raise ValueError(f"Query vectors shape mismatch: got {queries.shape}, expected (?, {target_dim}). expected_dim was {expected_dim}, index.d was {index.d}")
    import logging
    logging.getLogger("MCPVectorOps").debug(f"[batch_search] queries.shape={queries.shape}, index.d={index.d}, expected_dim={expected_dim}")
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
        logger.info(f"[vector_ops] Using vector_store_type={input_data.vector_store_type} for security_id={input_data.security_id}")
        # Load vector store
        index, metadata = load_vector_store(input_data.security_id, input_data.vector_store_type)
        logger.info(f"[vector_ops] Loaded vector store directory for type={input_data.vector_store_type}, security_id={input_data.security_id}")
        
        # Get embeddings using Vertex AI
        model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
        
        # Import config API key for Gemini embedding
        from financial_analyst.config import GOOGLE_API_KEY
        # Perform operation
        if input_data.operation == "search_similar":
            if not input_data.query:
                raise ValueError("search_similar requires a query")
            if input_data.vector_store_type == "text":
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
                embedding_dim = TEXT_EMBEDDING_DIMENSION
                query_vector = embedder.embed_query(input_data.query)
            else:
                embedding_dim = index.d
                result = model.get_embeddings(contextual_text=input_data.query)
                query_vector = result.text_embedding
            result = search_similar(index, query_vector, input_data.k)
            message = f"Found {input_data.k} similar items for query '{input_data.query}' in security {input_data.security_id}"
            
        elif input_data.operation == "batch_search":
            if not input_data.queries:
                raise ValueError("batch_search requires queries")
            # Use correct embedding model and dimension for vector_store_type
            if input_data.vector_store_type == "text":
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
                embedding_dim = TEXT_EMBEDDING_DIMENSION
                query_vectors = embedder.embed_documents(input_data.queries)
            else:
                embedding_dim = index.d
                query_vectors = []
                for query in input_data.queries:
                    result = model.get_embeddings(contextual_text=query)
                    query_vectors.append(result.text_embedding)
            result = batch_search(index, query_vectors, input_data.k, expected_dim=embedding_dim)
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
