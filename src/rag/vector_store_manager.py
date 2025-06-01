import json
import logging
import asyncio
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import numpy as np

# Attempt to import faiss. If not found, it means it's not installed.
try:
    import faiss
except ImportError:
    faiss = None
    # Logger might not be initialized yet, so print a warning for now.
    print("WARNING: FAISS library not found. FaissVectorStoreManager will not be functional. Please install faiss-cpu or faiss-gpu.")

# Attempt to import DocumentChunk from the sibling module.
try:
    from .chunking import DocumentChunk
    from .embedding_service import BaseEmbeddingService # For type hinting
except ImportError:
    # This allows the file to be parsed, but type hints will be unresolved
    # if chunking.py or embedding_service.py is not available.
    DocumentChunk = Any  # type: ignore
    BaseEmbeddingService = Any # type: ignore

logger = logging.getLogger(__name__)

class BaseVectorStoreManager(ABC):
    @abstractmethod
    async def add_to_vector_store(self, security_id: str, document_chunks: List[DocumentChunk], embedding_service: BaseEmbeddingService) -> None:
        pass

    @abstractmethod
    async def similarity_search_with_scores(self, security_id: str, query_embedding: List[float], k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[DocumentChunk, float]]:
        pass

    @abstractmethod
    async def load_vector_store(self, security_id: str, embedding_dim_if_new: Optional[int] = None) -> bool:
        pass

    @abstractmethod
    async def save_vector_store(self, security_id: str) -> None:
        pass

    @abstractmethod
    async def vector_store_exists(self, security_id: str) -> bool:
        pass

class FaissVectorStoreManager(BaseVectorStoreManager):
    def __init__(self, vector_stores_base_path: Path):
        if faiss is None:
            raise ImportError("FAISS library is not installed. Cannot use FaissVectorStoreManager.")
            
        self.logger = logging.getLogger(__name__)
        self.vector_stores_base_path = vector_stores_base_path
        self.vector_stores_base_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"FaissVectorStoreManager initialized with base path: {self.vector_stores_base_path}")

        # Active store attributes - these will be populated by load_vector_store for a specific security_id
        self.active_security_id: Optional[str] = None
        self.active_faiss_index: Optional[faiss.Index] = None
        self.active_chunks_metadata_list: List[DocumentChunk] = []
        self.active_embedding_dim: Optional[int] = None

    def _get_paths_for_security_id(self, security_id: str) -> Tuple[Path, Path, Path]:
        store_dir = self.vector_stores_base_path / security_id
        index_file = store_dir / "index.faiss"
        metadata_file = store_dir / "chunks_meta.jsonl"
        return store_dir, index_file, metadata_file

    def _sync_vector_store_exists(self, security_id: str) -> bool:
        _store_dir, index_file, metadata_file = self._get_paths_for_security_id(security_id)
        index_exists = index_file.exists() and index_file.stat().st_size > 0
        meta_exists = metadata_file.exists() and metadata_file.stat().st_size > 0
        self.logger.debug(f"VSM.sync_vector_store_exists for {security_id}: Index Exists: {index_exists}, Meta Exists: {meta_exists}")
        return index_exists and meta_exists

    async def vector_store_exists(self, security_id: str) -> bool:
        return await asyncio.to_thread(self._sync_vector_store_exists, security_id)

    def _sync_load_vector_store(self, security_id: str, embedding_dim_if_new: Optional[int] = None) -> bool:
        if self.active_security_id == security_id and self.active_faiss_index is not None:
            self.logger.info(f"VSM.sync_load_vector_store: Store for '{security_id}' already active.")
            return True

        store_dir, index_file, metadata_file = self._get_paths_for_security_id(security_id)
        store_dir.mkdir(parents=True, exist_ok=True)

        self.active_security_id = security_id
        self.active_chunks_metadata_list = []
        self.active_faiss_index = None
        self.active_embedding_dim = None # Reset active embedding dim

        if self._sync_vector_store_exists(security_id):
            try:
                self.logger.info(f"VSM.sync_load_vector_store: Loading FAISS index from: {index_file}")
                loaded_index = faiss.read_index(str(index_file))
                self.active_faiss_index = loaded_index
                self.active_embedding_dim = loaded_index.d # Get dimension from loaded index
                self.logger.info(f"VSM.sync_load_vector_store: FAISS index for '{security_id}' loaded. Dim: {self.active_embedding_dim}, NTotal: {loaded_index.ntotal}")

                with open(metadata_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        self.active_chunks_metadata_list.append(DocumentChunk(**data))
                self.logger.info(f"VSM.sync_load_vector_store: {len(self.active_chunks_metadata_list)} metadata items loaded for '{security_id}'.")
                return True
            except Exception as e:
                self.logger.error(f"VSM.sync_load_vector_store: Error loading store for '{security_id}' from {store_dir}. Error: {e}", exc_info=True)
                self.active_faiss_index = None
                self.active_chunks_metadata_list = []
                self.active_security_id = None
                self.active_embedding_dim = None
                return False
        else:
            self.logger.info(f"VSM.sync_load_vector_store: Store for '{security_id}' does not exist or is incomplete. Initializing empty store structure.")
            if embedding_dim_if_new:
                self.active_embedding_dim = embedding_dim_if_new
                # IndexIDMap supports adding IDs later. IndexFlatL2 is a common choice for dense vectors.
                self.active_faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(self.active_embedding_dim))
                self.logger.info(f"VSM.sync_load_vector_store: New FAISS index initialized for '{security_id}' with dim {self.active_embedding_dim}.")
            else:
                # Cannot initialize index without dimension. Store will be loaded as empty but unusable for adds until dim is known.
                self.logger.warning(f"VSM.sync_load_vector_store: Store for '{security_id}' not found and no embedding_dim_if_new provided. Index not initialized.")
            return True # Considered 'loaded' as an empty state

    async def load_vector_store(self, security_id: str, embedding_dim_if_new: Optional[int] = None) -> bool:
        return await asyncio.to_thread(self._sync_load_vector_store, security_id, embedding_dim_if_new)

    def _sync_save_vector_store(self, security_id: str) -> None:
        if self.active_security_id != security_id:
            self.logger.error(f"VSM.sync_save_vector_store: Attempted to save store for '{security_id}', but active store is '{self.active_security_id}'. Load store first.")
            # Potentially raise an error or try to load it, but for now, just log and return.
            # raise ValueError(f"Store for {security_id} is not active. Load it before saving.")
            return

        if self.active_faiss_index is None or self.active_embedding_dim is None:
            self.logger.warning(f"VSM.sync_save_vector_store: Attempted to save store for '{security_id}' but FAISS index or embedding_dim is not initialized. No action taken.")
            return

        store_dir, index_file, metadata_file = self._get_paths_for_security_id(security_id)
        store_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.logger.info(f"VSM.sync_save_vector_store: Saving FAISS index for '{security_id}' to {index_file}")
            faiss.write_index(self.active_faiss_index, str(index_file))
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                for chunk_meta in self.active_chunks_metadata_list:
                    f.write(json.dumps(chunk_meta.model_dump()) + '\n')
            self.logger.info(f"VSM.sync_save_vector_store: Store for '{security_id}' saved successfully to {store_dir}.")
        except Exception as e:
            self.logger.error(f"VSM.sync_save_vector_store: Failed to save store for '{security_id}' to {store_dir}. Error: {e}", exc_info=True)
            # Consider if partial save needs cleanup

    async def save_vector_store(self, security_id: str) -> None:
        await asyncio.to_thread(self._sync_save_vector_store, security_id)

    async def add_to_vector_store(self, security_id: str, document_chunks: List[DocumentChunk], embedding_service: BaseEmbeddingService) -> None:
        if not document_chunks:
            self.logger.info(f"VSM.add_to_vector_store: No document chunks provided for '{security_id}'. Nothing to add.")
            return

        # Ensure the target store is active or load/initialize it
        if self.active_security_id != security_id:
            self.logger.info(f"VSM.add_to_vector_store: Store for '{security_id}' not active. Attempting to load/initialize.")
            # Try to get embedding dimension from the first chunk's embedding if store is new
            # This assumes all embeddings for this call will have the same dimension.
            temp_first_embedding = await embedding_service.embed_documents([document_chunks[0].text_content])
            initial_embedding_dim = len(temp_first_embedding[0]) if temp_first_embedding else None
            
            if not await self.load_vector_store(security_id, embedding_dim_if_new=initial_embedding_dim):
                self.logger.error(f"VSM.add_to_vector_store: Failed to load or initialize store for '{security_id}'. Cannot add chunks.")
                return
        
        # If index is still None after load_vector_store (e.g. new store, no dim previously), try to initialize now.
        if self.active_faiss_index is None:
            if self.active_embedding_dim:
                 self.active_faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(self.active_embedding_dim))
                 self.logger.info(f"VSM.add_to_vector_store: FAISS index for '{security_id}' initialized with dim {self.active_embedding_dim} during add operation.")
            else: # Still no dimension, try to get from first chunk
                texts_to_embed_for_dim_check = [doc_chunk.text_content for doc_chunk in document_chunks[:1]]
                if texts_to_embed_for_dim_check:
                    embeddings_for_dim_check = await embedding_service.embed_documents(texts_to_embed_for_dim_check)
                    if embeddings_for_dim_check and embeddings_for_dim_check[0]:
                        self.active_embedding_dim = len(embeddings_for_dim_check[0])
                        self.active_faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(self.active_embedding_dim))
                        self.logger.info(f"VSM.add_to_vector_store: FAISS index for '{security_id}' initialized with derived dim {self.active_embedding_dim}.")
                    else:
                        self.logger.error(f"VSM.add_to_vector_store: Could not determine embedding dimension for '{security_id}'. Cannot add chunks.")
                        return
                else: # No chunks to derive dim from
                     self.logger.error(f"VSM.add_to_vector_store: No chunks to derive embedding dimension for '{security_id}'. Cannot add chunks.")
                     return

        texts_to_embed = [doc_chunk.text_content for doc_chunk in document_chunks]
        if not texts_to_embed:
            self.logger.info(f"VSM.add_to_vector_store: All provided document chunks have empty text_content for '{security_id}'.")
            return

        embeddings = await embedding_service.embed_documents(texts_to_embed)
        if not embeddings or len(embeddings[0]) != self.active_embedding_dim:
            self.logger.error(f"VSM.add_to_vector_store: Embedding dimension mismatch or no embeddings returned for '{security_id}'. Expected {self.active_embedding_dim}, got {len(embeddings[0]) if embeddings else 'None'}. Chunks not added.")
            return

        embeddings_np = np.array(embeddings, dtype=np.float32)
        
        # Generate unique IDs for FAISS. These could be based on chunk_id if they are unique and numeric,
        # or simply indices into the active_chunks_metadata_list.
        # For simplicity, using indices corresponding to the current length of metadata list.
        start_id = len(self.active_chunks_metadata_list)
        ids_for_faiss = np.arange(start_id, start_id + len(document_chunks), dtype=np.int64)

        self.active_faiss_index.add_with_ids(embeddings_np, ids_for_faiss)
        self.active_chunks_metadata_list.extend(document_chunks)
        self.logger.info(f"VSM.add_to_vector_store: Added {len(document_chunks)} chunks to store for '{security_id}'. Total chunks: {len(self.active_chunks_metadata_list)}, FAISS index total: {self.active_faiss_index.ntotal}.")

        await self.save_vector_store(security_id) # Persist changes

    def _sync_similarity_search(self, security_id: str, query_embedding: List[float], k: int, filter_metadata: Optional[Dict[str, Any]]) -> List[Tuple[DocumentChunk, float]]:
        if self.active_security_id != security_id:
            self.logger.warning(f"VSM.sync_similarity_search: Store for '{security_id}' not active. Attempting to load.")
            # Try to load with a common dimension or fail. For search, dim must be known.
            # This is tricky because we don't know the query embedding's dimension source here.
            # Best if load_vector_store is called explicitly before search if store isn't active.
            if not self._sync_load_vector_store(security_id, embedding_dim_if_new=len(query_embedding)):
                 self.logger.error(f"VSM.sync_similarity_search: Failed to load store for '{security_id}'. Cannot perform search.")
                 return []
        
        if self.active_faiss_index is None or self.active_faiss_index.ntotal == 0:
            self.logger.info(f"VSM.sync_similarity_search: FAISS index for '{security_id}' is empty or not initialized.")
            return []
        
        if len(query_embedding) != self.active_embedding_dim:
            self.logger.error(f"VSM.sync_similarity_search: Query embedding dimension ({len(query_embedding)}) does not match store dimension ({self.active_embedding_dim}) for '{security_id}'.")
            return []

        query_embedding_np = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.active_faiss_index.search(query_embedding_np, k)
        
        results: List[Tuple[DocumentChunk, float]] = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            dist = distances[0][i]
            if idx != -1: # FAISS returns -1 for non-existent neighbors
                if 0 <= idx < len(self.active_chunks_metadata_list):
                    chunk_candidate = self.active_chunks_metadata_list[idx]
                    # Implement metadata filtering if provided
                    if filter_metadata:
                        match = True
                        for filter_key, filter_value in filter_metadata.items():
                            # Assuming metadata is a flat dict in DocumentChunk or accessible via .metadata
                            chunk_meta_value = getattr(chunk_candidate, filter_key, None)
                            if chunk_meta_value is None and hasattr(chunk_candidate, 'metadata'): # check in .metadata dict
                                chunk_meta_value = chunk_candidate.metadata.get(filter_key)
                            
                            if chunk_meta_value != filter_value:
                                match = False
                                break
                        if match:
                            results.append((chunk_candidate, float(dist)))
                    else:
                        results.append((chunk_candidate, float(dist)))
                else:
                    self.logger.warning(f"VSM.sync_similarity_search: FAISS returned index {idx} out of bounds for metadata list (len {len(self.active_chunks_metadata_list)}) for '{security_id}'.")
        return results

    async def similarity_search_with_scores(self, security_id: str, query_embedding: List[float], k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[DocumentChunk, float]]:
        return await asyncio.to_thread(self._sync_similarity_search, security_id, query_embedding, k, filter_metadata)
