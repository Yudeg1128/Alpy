"""
MCP Image Embedder Server

Creates and manages FAISS vector stores from images using Gemini embeddings.
Supports per-security vector stores for efficient document retrieval.
"""

import json
import logging
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

import faiss
import numpy as np
from PIL import Image as PILImage
from io import BytesIO
from pydantic import BaseModel, Field
import vertexai
from vertexai.vision_models import Image as VertexImage, MultiModalEmbeddingModel

from mcp.server.fastmcp import FastMCP
from financial_analyst.config import GOOGLE_API_KEY

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
    format='[%(asctime)s][%(levelname)s][MCPImageEmbedder] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mcp_image_embedder.log')
    ]
)
logger = logging.getLogger(__name__)
logger.info("Starting MCP Image Embedder server")

# Constants
OUTPUT_BASE_DIR = Path('/home/me/CascadeProjects/Alpy/otcmn_tool_test_output/current')
VECTOR_DIMENSION = 1408
VECTOR_STORE_NAME = "vector_store"

# Rate limiting constants
REQUESTS_PER_MINUTE = 10  # API rate limit
BYTES_PER_MINUTE = 1.5 * 1024 * 1024 * 1024  # 1.5GB per minute
BATCH_SIZE = 5  # Process in batches of 5
BATCH_DELAY_SECONDS = 6  # 10 reqs/min = 1 req/6s

# --- Pydantic Models ---
class ImageEmbedderInput(BaseModel):
    security_id: str = Field(..., description="Security identifier")
    image_paths: List[str] = Field(..., description="List of image paths to embed")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata for each image")

class ImageEmbedderOutput(BaseModel):
    status: str
    message: str
    vector_store_path: Optional[str] = None

# --- MCP App ---
mcp_app = FastMCP(
    name="MCPImageEmbedderServer",
    version="1.0.0",
    description="MCP server for creating FAISS vector stores from images using Vertex AI's multimodal embedding model"
)

def get_board_path(security_id: str) -> Path:
    for board_path in OUTPUT_BASE_DIR.iterdir():
        if not board_path.is_dir():
            continue
        if (board_path / security_id).exists():
            return board_path
    raise ValueError(f"Security ID {security_id} not found in any board directory")

def load_image_as_blob(image_path: str) -> bytes:
    """Load an image from path and return it as bytes."""
    with Image.open(image_path) as img:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Calculate new size while maintaining aspect ratio
        target_size = 36000  # Max size in bytes
        ratio = 1
        while True:
            # Save to bytes
            img_bytes = BytesIO()
            resized = img.resize((int(img.width/ratio), int(img.height/ratio)), Image.Resampling.LANCZOS)
            resized.save(img_bytes, format='JPEG', quality=85)
            if len(img_bytes.getvalue()) <= target_size or ratio > 8:
                break
            ratio *= 1.5
        
        logger.info(f"Resized image {image_path} with ratio {ratio:.1f}, final size: {len(img_bytes.getvalue())} bytes")
        
        return img_bytes.getvalue()

def create_faiss_index(embeddings: List[List[float]]) -> Any:
    """Create a FAISS index from embeddings."""
    if not embeddings:
        raise ValueError("Cannot create index with empty embeddings")
    embeddings_array = np.array(embeddings, dtype=np.float32)
    if embeddings_array.shape[1] != VECTOR_DIMENSION:
        raise ValueError(f"Expected embeddings of dimension {VECTOR_DIMENSION}, got {embeddings_array.shape[1]}")
    index = faiss.IndexFlatL2(VECTOR_DIMENSION)
    index.add(embeddings_array)
    return index

def load_vector_store(base_path: Path) -> Tuple[Optional[Any], Optional[Dict], Set[str]]:
    """Load existing vector store and return index, metadata, and processed image paths."""
    if not isinstance(base_path, Path):
        base_path = Path(base_path)
    
    store_dir = base_path / VECTOR_STORE_NAME
    if not store_dir.exists():
        return None, None, set()
        
    try:
        # Load index if it exists
        index_path = store_dir / "index.faiss"
        index = faiss.read_index(str(index_path)) if index_path.exists() else None
        
        # Load metadata if it exists
        metadata_path = store_dir / "metadata.json"
        metadata = {}
        processed_paths = set()
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                processed_paths = {m['path'] for m in metadata.values()}
                
        return index, metadata, processed_paths
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        return None, None, set()

def save_vector_store(index: Any, metadata: Dict, base_path: Path) -> str:
    """Save FAISS index and metadata to disk."""
    if not isinstance(base_path, Path):
        base_path = Path(base_path)
    
    store_dir = base_path / VECTOR_STORE_NAME
    store_dir.mkdir(parents=True, exist_ok=True)
    
    # Save index
    index_path = store_dir / "index.faiss"
    try:
        faiss.write_index(index, str(index_path))
    except Exception as e:
        raise RuntimeError(f"Failed to save FAISS index: {e}")
    
    # Save metadata
    metadata_path = store_dir / "metadata.json"
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
    except Exception as e:
        # Clean up index if metadata save fails
        if index_path.exists():
            index_path.unlink()
        raise RuntimeError(f"Failed to save metadata: {e}")
    
    return str(store_dir)

@mcp_app.tool(name="embed_images")
async def embed_images(security_id: str, image_paths: List[str], metadata: Optional[Dict] = None) -> ImageEmbedderOutput:
    logger.info(f"Embedding images for security {security_id}")
    logger.debug(f"Image paths: {image_paths}")

    # Create vector store directory if it doesn't exist
    board_path = get_board_path(security_id)
    logger.debug(f"Board path: {board_path}")
    security_path = board_path / security_id
    logger.debug(f"Security path: {security_path}")

    # Get multimodal model
    logger.debug("Initializing Vertex AI model")
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    logger.debug("Model initialized successfully")
    embeddings = []
    image_metadata = {}

    for i, image_path in enumerate(image_paths):
        logger.debug(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
        if not Path(image_path).exists():
            logger.warning(f"Image not found: {image_path}")
            continue

        try:
            # Load and resize image if needed
            logger.debug(f"Loading and resizing image: {image_path}")
            img_bytes = load_image_as_blob(image_path)
            logger.debug(f"Image loaded, size: {len(img_bytes)} bytes")
            
            # Get embedding
            logger.debug("Getting embedding from model")
            prediction = model.predict([{
                'image_bytes': img_bytes
            }])
            logger.debug("Got prediction from model")
            embedding = prediction[0]

            embeddings.append(embedding)
            image_metadata[i] = {
                "path": image_path,
                "metadata": metadata or {}
            }

            logger.info(f"Successfully embedded image {i+1}/{len(image_paths)}")

        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            continue

    if not embeddings:
        return ImageEmbedderOutput(
            status="error",
            message="No images were successfully embedded"
        )

    index = create_faiss_index(embeddings)
    store_path = save_vector_store(index, image_metadata, security_path)

    return ImageEmbedderOutput(
        status="success",
        message=f"Created vector store with {len(embeddings)} embeddings",
        vector_store_path=store_path
    )

async def run_stdio_server():
    logger.debug("Starting stdio server")
    mcp_app = FastMCP("MCPImageEmbedderServer")
    
    @mcp_app.tool(name="embed_images")
    async def embed_images(security_id: str, image_paths: List[str], metadata: Optional[Dict] = None) -> ImageEmbedderOutput:
        try:
            board_path = get_board_path(security_id)
            security_path = board_path / security_id
            
            logger.debug(f"Processing images for security {security_id}")
            logger.debug(f"Image paths: {image_paths}")
            
            # Get multimodal model
            logger.debug("Initializing Vertex AI model")
            model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
            logger.debug("Model initialized successfully")
            
            # Load existing progress
            existing_index, existing_metadata, processed_paths = load_vector_store(security_path)
            
            # Check for store corruption
            if len(processed_paths) > len(image_paths):
                logger.warning(f"Vector store contains more images ({len(processed_paths)}) than exist ({len(image_paths)}). Clearing store.")
                existing_index = None
                existing_metadata = None
                processed_paths = set()
            
            # Initialize embeddings from existing store
            if existing_index is not None:
                logger.info(f"Found existing vector store with {existing_index.ntotal} embeddings")
                all_embeddings = [existing_index.reconstruct(i) for i in range(existing_index.ntotal)]
                all_metadata = existing_metadata
            else:
                all_embeddings = []
                all_metadata = {}
            
            # Filter out already processed images
            pending_images = [p for p in image_paths if p not in processed_paths]
            total_images = len(pending_images)
            if not pending_images:
                if len(processed_paths) != len(image_paths):
                    logger.warning(f"Vector store mismatch: {len(processed_paths)} stored vs {len(image_paths)} images. Clearing store.")
                    existing_index = None
                    existing_metadata = None
                    processed_paths = set()
                    pending_images = image_paths
                    total_images = len(pending_images)
                else:
                    logger.info(f"All {len(processed_paths)} images already embedded")
                    return ImageEmbedderOutput(
                    status="success",
                    message=f"All {len(processed_paths)} images already embedded",
                    vector_store_path=str(security_path / VECTOR_STORE_NAME)
                )
                
            logger.info(f"Processing {len(pending_images)} new images")
            store_path = None
            
            # Track data usage for rate limiting
            bytes_this_minute = 0
            minute_start = asyncio.get_event_loop().time()
            requests_this_minute = 0
            
            for batch_start in range(0, total_images, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, total_images)
                batch = pending_images[batch_start:batch_end]
                
                # Check and reset rate limits
                current_time = asyncio.get_event_loop().time()
                if current_time - minute_start >= 60:
                    bytes_this_minute = 0
                    requests_this_minute = 0
                    minute_start = current_time
                
                logger.info(f"Processing batch {batch_start//BATCH_SIZE + 1}/{(total_images + BATCH_SIZE - 1)//BATCH_SIZE}")
                
                # Load all images in batch first
                batch_images = []
                batch_paths = []
                batch_sizes = []
                
                for image_path in batch:
                    path = Path(image_path)
                    if not path.exists() or not path.is_file():
                        logger.warning(f"Image not found or not a file: {image_path}")
                        continue
                        
                    # Check size against rate limit
                    file_size = path.stat().st_size
                    if bytes_this_minute + file_size > BYTES_PER_MINUTE:
                        wait_time = 60 - (current_time - minute_start)
                        logger.info(f"Data limit reached, waiting {wait_time:.1f}s...")
                        await asyncio.sleep(wait_time)
                        bytes_this_minute = 0
                        minute_start = asyncio.get_event_loop().time()
                    
                    try:
                        image = VertexImage.load_from_file(str(path))
                        batch_images.append(image)
                        batch_paths.append(image_path)
                        batch_sizes.append(file_size)
                        bytes_this_minute += file_size
                    except Exception as e:
                        logger.error(f"Failed to load image {image_path}: {e}")
                        continue
                
                if not batch_images:
                    logger.warning("No valid images in batch, skipping")
                    continue
                
                try:
                    # Check request rate limit
                    if requests_this_minute >= REQUESTS_PER_MINUTE:
                        wait_time = 60 - (current_time - minute_start)
                        logger.info(f"Request limit reached, waiting {wait_time:.1f}s...")
                        await asyncio.sleep(wait_time)
                        requests_this_minute = 0
                        minute_start = asyncio.get_event_loop().time()
                    
                    # Get embeddings for batch
                    logger.debug(f"Getting embeddings for batch of {len(batch_images)} images")
                    results = [model.get_embeddings(image=img) for img in batch_images]
                    requests_this_minute += len(batch_images)
                    
                    # Process results
                    next_idx = len(all_embeddings)
                    for result, image_path in zip(results, batch_paths):
                        if not hasattr(result, 'image_embedding') or result.image_embedding is None:
                            logger.error(f"No embedding returned for {image_path}")
                            continue
                            
                        embedding = result.image_embedding
                        if len(embedding) != VECTOR_DIMENSION:
                            logger.error(f"Wrong embedding dimension for {image_path}: got {len(embedding)}, expected {VECTOR_DIMENSION}")
                            continue
                            
                        all_embeddings.append(embedding)
                        all_metadata[next_idx] = {
                            "path": image_path,
                            "metadata": metadata or {}
                        }
                        next_idx += 1
                        logger.info(f"Successfully embedded image {len(all_embeddings)}/{total_images} (total processed: {len(all_embeddings) + len(processed_paths)}/{len(image_paths)})")
                    
                    # Create/update index with current batch
                    if all_embeddings:  # Only create index if we have valid embeddings
                        try:
                            index = create_faiss_index(all_embeddings)
                            store_path = save_vector_store(index, all_metadata, security_path)
                            logger.info(f"Saved {len(all_embeddings)} embeddings to vector store")
                        except Exception as e:
                            logger.error(f"Failed to save vector store: {e}")
                            # Continue processing next batch even if save fails
                    
                except Exception as e:
                    logger.error(f"Failed to process batch: {e}")
                    continue
                
                # Adaptive rate limiting delay
                if batch_end < total_images:
                    delay = max(0, (60 / REQUESTS_PER_MINUTE) - (asyncio.get_event_loop().time() - current_time))
                    if delay > 0:
                        logger.info(f"Rate limiting: waiting {delay:.1f}s...")
                        await asyncio.sleep(delay)
            
            if not all_embeddings:
                return ImageEmbedderOutput(
                    status="error",
                    message="No images were successfully embedded"
                )

            if not all_embeddings:
                return ImageEmbedderOutput(
                    status="error",
                    message="No images were successfully embedded"
                )
            
            if store_path is None:
                return ImageEmbedderOutput(
                    status="error",
                    message="Failed to save vector store"
                )
                
            return ImageEmbedderOutput(
                status="success",
                message=f"Created vector store with {len(all_embeddings)} embeddings",
                vector_store_path=store_path
            )

        except Exception as e:
            logger.error("Unexpected error in embed_images", exc_info=True)
            return ImageEmbedderOutput(status="error", message=str(e))
    
    await mcp_app.run_stdio_async()

def main():
    # Initialize logging
    logger.info("Starting MCP Image Embedder server")

    try:
        # Initialize MCP app
        logger.debug("Creating MCPImageEmbedderServer instance")
        import asyncio
        asyncio.run(run_stdio_server())
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
