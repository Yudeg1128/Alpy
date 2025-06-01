import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .chunking import chunk_parsed_document, DocumentChunk
from .embedding_service import BaseEmbeddingService
from .vector_store_manager import BaseVectorStoreManager
from .retriever import RagRetriever
from .. import config as alpy_config

class RagManager:
    def __init__(self, 
                 embedding_service: BaseEmbeddingService, 
                 vector_store_manager: BaseVectorStoreManager, 
                 logger: Optional[logging.Logger] = None):
        self.embedding_service = embedding_service
        self.vector_store_manager = vector_store_manager
        self.logger = logger or logging.getLogger(__name__)
        self.alpy_config = alpy_config
        self.retriever = RagRetriever(embedding_service, vector_store_manager)
        if not logger: 
            logging.basicConfig(level=alpy_config.LOG_LEVEL)
            self.logger.setLevel(alpy_config.LOG_LEVEL)

    async def process_document_for_rag(self, security_id: str, parsed_doc_content: Dict[str, Any], document_name: str) -> int:
        """Chunks a single parsed document, embeds chunks, and adds to vector store.
        Returns the number of chunks processed.
        """
        self.logger.info(f"[RagManager] Processing document for RAG: {document_name} (Security ID: {security_id})")

        try:
            self.logger.debug(f"[RagManager] About to call chunk_parsed_document. Parsed_doc_content type: {type(parsed_doc_content)}, keys: {list(parsed_doc_content.keys()) if isinstance(parsed_doc_content, dict) else ('Not a dict or None' if parsed_doc_content is not None else 'None')}")
            chunks: List[DocumentChunk] = chunk_parsed_document(
                parsed_doc_content=parsed_doc_content,
                document_id=document_name,
                chunk_size=alpy_config.RAG_CHUNK_SIZE,
                overlap_size=int(alpy_config.RAG_CHUNK_SIZE * alpy_config.RAG_CHUNK_OVERLAP_PERCENTAGE),
                strategy=alpy_config.RAG_CHUNK_STRATEGY,
                min_chunk_size_chars=alpy_config.RAG_MIN_CHUNK_SIZE_CHARS
            )

            if not chunks:
                self.logger.warning(f"[RagManager] No chunks generated for document: {document_name}. It might be empty or too small.")
                return 0

            self.logger.info(f"[RagManager] Generated {len(chunks)} chunks for document: {document_name}. Adding to vector store.")
            
            # The document_name is already part of each DocumentChunk object in the 'chunks' list.
            # The FaissVectorStoreManager's add_to_vector_store expects List[DocumentChunk] and the embedding_service.
            await self.vector_store_manager.add_to_vector_store(
                security_id=security_id,
                document_chunks=chunks, 
                embedding_service=self.embedding_service 
            )
            self.logger.info(f"[RagManager] Successfully added {len(chunks)} chunks from {document_name} to vector store for {security_id}.")
            return len(chunks)

        except Exception as e:
            self.logger.error(f"[RagManager] Failed to process document {document_name} for RAG: {e}", exc_info=True)
            raise

    async def retrieve_context_chunks(
        self, 
        security_id: str, 
        query: str, 
        k_retrieved_chunks: Optional[int] = None,
        document_name_filter: Optional[str] = None 
    ) -> List[DocumentChunk]:
        """Retrieves relevant document chunks for a given query."""
        self.logger.info(f"[RagManager] Retrieving context chunks for query: '{query[:50]}...' (Security ID: {security_id})")
        
        if k_retrieved_chunks is None:
            k_retrieved_chunks = alpy_config.RAG_NUM_RETRIEVED_CHUNKS

        try:
            self.logger.info(f"[RagManager] Delegating retrieval of top {k_retrieved_chunks} context chunks for query: '{query[:50]}...' for security_id: {security_id} to RagRetriever.")

            retrieved_chunks = await self.retriever.retrieve_context_chunks(
                security_id=security_id,
                query_text=query, 
                k=k_retrieved_chunks,
                filter_metadata=None if document_name_filter is None else {'document_name': document_name_filter}
            )
            
            self.logger.info(f"[RagManager] RagRetriever returned {len(retrieved_chunks)} chunks for query: '{query[:50]}...'" )
            return retrieved_chunks

        except Exception as e:
            self.logger.error(f"[RagManager] Failed to retrieve context chunks: {e}", exc_info=True)
            return [] 
