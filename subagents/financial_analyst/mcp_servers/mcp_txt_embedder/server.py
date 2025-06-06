"""MCP Server for txt embedder: parses all images for a security_id and builds a FAISS vector store from the resulting txts using Gemini embeddings."""

import sys
from pathlib import Path
from financial_analyst.security_folder_utils import require_security_folder, get_subfolder, get_security_file
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio
import logging
import os
from typing import Optional, List

from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP
from mcp.types import Implementation
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import faiss
import numpy as np
from financial_analyst.config import GOOGLE_API_KEY

# Hardcoded Gemini embedding model for FAISS vector store
EMBEDDING_MODEL = "models/embedding-001"

# --- Logging ---
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s][%(levelname)s][MCPTxtEmbedder] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("MCPTxtEmbedder")

# --- Helper: Find security images folder ---

# --- Pydantic Models ---
class ParseImagesInput(BaseModel):
    security_id: str = Field(description="Security identifier")

class ParseImagesOutput(BaseModel):
    status: str
    message: str
    parsed_txt_files: Optional[List[str]] = None

class BuildVectorStoreInput(BaseModel):
    security_id: str = Field(description="Security identifier")

class BuildVectorStoreOutput(BaseModel):
    status: str
    message: str
    vector_store_path: Optional[str] = None
    txt_files: Optional[List[str]] = None

# --- MCP App ---
mcp_app = FastMCP(
    name="MCPTxtEmbedderServer",
    version="1.0.0",
    description="MCP server for txt embedder: parses images and builds FAISS vector store"
)

@mcp_app.tool(name="parse_security_images_to_txts")
async def parse_security_images_to_txts(input_data: ParseImagesInput) -> ParseImagesOutput:
    """
    Parses all images for the given security_id to .json files (structured JSON). Skips images with existing .json files.
    """
    security_id = input_data.security_id
    logger.info(f"[parse_security_images_to_txts] START for security_id={security_id}")
    security_folder = require_security_folder(security_id)
    images_dir = get_subfolder(security_id, "images")
    parsed_images_dir = get_subfolder(security_id, "parsed_images")
    if not images_dir.exists() or not images_dir.is_dir():
        msg = f"Images directory not found for id: {security_id}"
        logger.error(msg)
        return ParseImagesOutput(status="error", message=msg)
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.types import Implementation as MCPImplementation
    import sys
    server_script = Path(__file__).parent.parent / "mcp_image_parser" / "server.py"
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_script)],
        cwd=str(server_script.parent)
    )
    client_info = MCPImplementation(name="FinancialAnalystImageParserClient", version="1.0.0")
    MAX_RPM = 20
    DELAY_BETWEEN = 60.0 / MAX_RPM
    image_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    logger.info(f"[parse_security_images_to_txts] Found {len(image_files)} images.")
    parsed_json_files = []
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream, client_info=client_info) as session:
            await session.initialize()
            for idx, image_file in enumerate(image_files):
                json_path = parsed_images_dir / (image_file.stem + ".json")
                if json_path.exists():
                    logger.info(f"[parse_security_images_to_txts] Skipping {image_file}, json exists.")
                    parsed_json_files.append(str(json_path))
                    continue
                logger.info(f"[parse_security_images_to_txts] Parsing image {image_file} ({idx+1}/{len(image_files)})")
                args = {"input_data": {"image_path": str(image_file)}}
                try:
                    tool_res = await session.call_tool(name="parse_image", arguments=args)
                    if tool_res.isError:
                        err = getattr(tool_res, 'error', None) or getattr(tool_res, 'message', None) or str(tool_res)
                        logger.error(f"[parse_security_images_to_txts] Error parsing {image_file}: {err}")
                        continue
                    if json_path.exists():
                        parsed_json_files.append(str(json_path))
                except Exception as e:
                    logger.error(f"[parse_security_images_to_txts] Exception parsing {image_file}: {e}")
                await asyncio.sleep(DELAY_BETWEEN)
    if not parsed_json_files:
        msg = "No json files generated for security."
        logger.error(msg)
        return ParseImagesOutput(status="error", message=msg)
    return ParseImagesOutput(status="success", message=f"Parsed {len(parsed_json_files)} images.", parsed_txt_files=parsed_json_files)

@mcp_app.tool(name="build_faiss_vector_store")
async def build_faiss_vector_store(input_data: BuildVectorStoreInput) -> BuildVectorStoreOutput:
    security_id = input_data.security_id
    logger.info(f"[build_faiss_vector_store] START for security_id={security_id}")
    security_folder = require_security_folder(security_id)
    parsed_images_dir = get_subfolder(security_id, "parsed_images")
    if not parsed_images_dir.exists() or not parsed_images_dir.is_dir():
        msg = f"Security or parsed_images folder not found for id: {security_id}"
        logger.error(msg)
        return BuildVectorStoreOutput(status="error", message=msg)
    # Collect all .json files from both parsed_images_dir and directly under the security_folder
    json_files = list(security_folder.glob('*.json')) + list(parsed_images_dir.glob('*.json'))
    all_files = [str(f) for f in json_files if f.name != "extracted_structured_bond_data.json"]
    texts = []
    metadata = []
    # Process json files (content_blocks)
    import json as _json
    def extract_text_from_content_blocks(data):
        blocks = data.get("content_blocks", [])
        texts = []
        for block in blocks:
            if block.get("type") == "table":
                # Flatten table rows
                table = block.get("content", [])
                for row in table:
                    if isinstance(row, list):
                        texts.append(" ".join(str(cell) for cell in row))
            else:
                content = block.get("content", "")
                if isinstance(content, str):
                    texts.append(content)
        return "\n".join(texts)
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = _json.load(f)
                if isinstance(data, dict):
                    extracted = extract_text_from_content_blocks(data)
                    texts.append(extracted)
                    metadata.append({
                        "file": str(json_file),
                        "type": "json"
                    })
                elif isinstance(data, list):
                    # Extract all string values from all dicts in the list
                    def extract_all_strings_from_list(lst):
                        result = []
                        for item in lst:
                            if isinstance(item, dict):
                                for v in item.values():
                                    if isinstance(v, str):
                                        result.append(v)
                            elif isinstance(item, str):
                                result.append(item)
                        return "\n".join(result)
                    extracted = extract_all_strings_from_list(data)
                    texts.append(extracted)
                    metadata.append({
                        "file": str(json_file),
                        "type": "json-list"
                    })
                else:
                    logger.warning(f"[build_faiss_vector_store] Skipping {json_file}: JSON root is not dict or list.")
        except Exception as e:
            logger.error(f"[build_faiss_vector_store] Failed to read/extract from {json_file}: {e}")
    if not texts:
        msg = "No text content loaded from txt or json files."
        logger.error(msg)
        return BuildVectorStoreOutput(status="error", message=msg)
    # all_files now contains both .txt and .json files for traceability
    embedder = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
    batch_size = 10
    vectors = []
    import numpy as np
    import faiss
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            batch_vecs = embedder.embed_documents(batch)
            vectors.extend(batch_vecs)
        except Exception as e:
            logger.error(f"[build_faiss_vector_store] Embedding batch failed: {e}")
            continue
        await asyncio.sleep(1.0)
    if not vectors:
        msg = "No vectors generated."
        logger.error(msg)
        return BuildVectorStoreOutput(status="error", message=msg)
    if not metadata or len(metadata) != len(vectors):
        msg = f"Metadata length ({len(metadata)}) does not match vectors ({len(vectors)}). Aborting metadata write."
        logger.error(msg)
        return BuildVectorStoreOutput(status="error", message=msg)
    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype(np.float32))
    vector_store_dir = get_subfolder(security_id, "vector_store_txt")
    vector_store_dir.mkdir(exist_ok=True)
    faiss_path = vector_store_dir / f"{security_id}_faiss.index"
    faiss.write_index(index, str(faiss_path))
    # Add explicit index to each metadata entry
    for idx, entry in enumerate(metadata):
        entry["index"] = idx
    # Save metadata mapping ONLY if valid
    metadata_path = vector_store_dir / f"{security_id}_vector_metadata.json"
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            _json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"[build_faiss_vector_store] Metadata saved to {metadata_path}")
    except Exception as e:
        logger.error(f"[build_faiss_vector_store] Failed to save metadata: {e}")
        metadata_path = None
    logger.info(f"[build_faiss_vector_store] Vector store saved to {faiss_path}")
    return BuildVectorStoreOutput(status="success", message=f"Built vector store from {len(all_files)} files (txt+json).", vector_store_path=str(faiss_path), txt_files=all_files, metadata_path=str(metadata_path) if metadata_path else None)


if __name__ == "__main__":
    import sys
    import asyncio
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
    logger.info("[BOOT] MCP txt embedder server main starting; stdio encoding set to utf-8")
    asyncio.run(mcp_app.run_stdio_async())
