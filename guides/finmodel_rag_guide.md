# Guide: Implementing RAG for Financial Data Extraction in Alpy

This guide outlines the steps to integrate a Retrieval Augmented Generation (RAG) system into Alpy's financial modeling `PhaseAOrchestrator`. This will provide more relevant context to the LLM, improving extraction accuracy for financial data from documents.

**Core Idea:** Instead of manually crafting context from page snippets, we will:
1.  **Chunk:** Break down parsed documents into smaller, manageable pieces.
2.  **Embed:** Convert these chunks into numerical representations (embeddings) that capture their semantic meaning.
3.  **Store:** Save these chunks and their embeddings in a searchable vector database.
4.  **Retrieve:** When information is needed (e.g., "Revenue for FY2023"), search the vector database for the most relevant chunks.
5.  **Augment & Generate:** Provide these retrieved chunks as context to the LLM along with the extraction prompt.

---

## I. Project Structure Changes

We'll introduce a new `rag` sub-directory within `src/` and a new top-level `data/` directory for persistent storage.

```
Alpy/
├── data/
│   └── vector_stores/          # To store vector DB indices and chunk metadata
│       └── <security_id_1>/    # Example: Stores for a specific security
│           ├── index.faiss     # FAISS index file
│           └── chunks_meta.jsonl # Metadata for chunks
│       └── <security_id_2>/
│           └── ...
├── src/
│   ├── financial_modeling/
│   │   ├── phase_a_orchestrator.py # Will be significantly modified
│   │   └── ...
│   ├── rag/                      # NEW: RAG components
│   │   ├── __init__.py
│   │   ├── chunking.py           # Document chunking strategies
│   │   ├── embedding_service.py  # Embedding model integration
│   │   ├── vector_store_manager.py # Vector database interaction
│   │   └── retriever.py          # Orchestrates query -> retrieval
│   ├── tools/
│   │   └── ...
│   ├── config.py                 # Add RAG related configurations
│   └── ...
├── mcp_servers/
├── prompts/
└── ...
```

---

## II. New RAG Components & Implementation Steps

### Step 1: Document Chunking (`src/rag/chunking.py`)

This module will be responsible for taking the parsed document content (text from pages, tables) and splitting it into smaller, meaningful chunks.

*   **Purpose:** Create appropriately sized text pieces for effective embedding and retrieval.
*   **Key File:** `src/rag/chunking.py`
*   **Implementation Details:**
    *   Define a Pydantic model for a `DocumentChunk`:
        ```python
        # src/rag/chunking.py (example Pydantic model)
        from pydantic import BaseModel
        from typing import Dict, Any, Optional

        class DocumentChunk(BaseModel):
            chunk_id: str # Unique ID for the chunk (e.g., f"{doc_name}_page_{page_num}_chunk_{idx}")
            text_content: str
            document_name: str
            page_number: Optional[int] = None
            table_index_on_page: Optional[int] = None # If chunk is from a table
            chunk_type: str # e.g., "text_snippet", "table_row", "full_table_text"
            metadata: Dict[str, Any] = {} # Other relevant info
        ```
    *   Implement chunking strategies:
        *   **Strategy 1: Page-wise text snippets (with overlap):** Split text from each page (PyMuPDF + OCR) into overlapping chunks of a certain character/token length.
        *   **Strategy 2: Table-aware chunking:**
            *   Represent each table as one or more chunks. A whole table as text, or perhaps row-by-row if tables are very large.
            *   Convert table data (list of lists) into a readable string format (e.g., Markdown, or simple "Header: Value" pairs per row).
        *   **Strategy 3 (Advanced): Semantic chunking:** Use NLP libraries or LLMs to split based on sections or topics (more complex).
    *   A primary function might look like:
        `def chunk_parsed_document(parsed_doc_content: Dict[str, Any], doc_name: str, config: Dict) -> List[DocumentChunk]:`
        This function would iterate through `parsed_doc_content["pages_content"]` and `parsed_doc_content["all_tables"]`, applying the chosen chunking strategies.

### Step 2: Embedding Service (`src/rag/embedding_service.py`)

This module will handle the conversion of text chunks into vector embeddings.

*   **Purpose:** Generate numerical representations that capture semantic meaning.
*   **Key File:** `src/rag/embedding_service.py`
*   **Implementation Details:**
    *   Define a base class:
        ```python
        # src/rag/embedding_service.py
        from abc import ABC, abstractmethod
        from typing import List
        # from .chunking import DocumentChunk # If DocumentChunk is defined in chunking.py

        class BaseEmbeddingService(ABC):
            @abstractmethod
            async def embed_documents(self, texts: List[str]) -> List[List[float]]:
                pass

            @abstractmethod
            async def embed_query(self, text: str) -> List[float]:
                pass
        ```
    *   Implement `GeminiEmbeddingService(BaseEmbeddingService)`:
        *   Uses `google.generativeai` library.
        *   Model name from `alpy_config.EMBEDDING_MODEL_NAME` (e.g., `text-embedding-004` or a newer Gemini embedding model).
        *   `__init__`: Configure `genai` with API key.
        *   `embed_documents`: Takes a list of chunk texts, calls `genai.embed_content()` (or equivalent for batch embeddings), handles batching if necessary for API limits.
        *   `embed_query`: Embeds a single query string.

### Step 3: Vector Store Setup & Management (`src/rag/vector_store_manager.py`)

This module manages interactions with the chosen vector database. For local use and simplicity, **FAISS** is a good starting point.

*   **Purpose:** Store, manage, and search chunk embeddings efficiently.
*   **Key File:** `src/rag/vector_store_manager.py`
*   **Directory for Stores:** `data/vector_stores/<security_id>/`
*   **Implementation Details (using FAISS):**
    *   Requires `faiss-cpu` (or `faiss-gpu`) and `numpy`. Add to `requirements.txt`.
    *   Define a base class:
        ```python
        # src/rag/vector_store_manager.py
        from abc import ABC, abstractmethod
        from typing import List, Tuple, Dict, Any, Optional
        from pathlib import Path
        # from .chunking import DocumentChunk

        class BaseVectorStoreManager(ABC):
            @abstractmethod
            def add_chunks_and_embeddings(self, chunks_with_meta: List[DocumentChunk], embeddings: List[List[float]]):
                pass

            @abstractmethod
            def similarity_search_with_scores(self, query_embedding: List[float], k: int = 5, filter_metadata: Optional[Dict] = None) -> List[Tuple[DocumentChunk, float]]: # Returns (chunk, score)
                pass
            
            @abstractmethod
            def load_store(self): # Load index and metadata from disk
                pass

            @abstractmethod
            def save_store(self): # Save index and metadata to disk
                pass

            @abstractmethod
            def store_exists(self) -> bool:
                pass
        ```
    *   Implement `FaissVectorStoreManager(BaseVectorStoreManager)`:
        *   `__init__(self, store_path_base: Path, security_id: str, embedding_dim: int)`:
            *   `self.index_file = store_path_base / security_id / "index.faiss"`
            *   `self.metadata_file = store_path_base / security_id / "chunks_meta.jsonl"`
            *   `self.faiss_index = None` (FAISS index object)
            *   `self.chunks_metadata_list: List[DocumentChunk] = []` (stores the actual chunks and their metadata, as FAISS only stores vectors and IDs).
            *   `self.embedding_dim = embedding_dim` (e.g., 768 for `text-embedding-004`).
        *   `store_exists()`: Checks if `index_file` and `metadata_file` exist.
        *   `add_chunks_and_embeddings()`:
            *   Converts `embeddings` (list of lists) to a NumPy array.
            *   If `self.faiss_index` is None, initialize it (`faiss.IndexFlatL2(embedding_dim)` or `faiss.IndexIDMap(faiss.IndexFlatL2(embedding_dim))`).
            *   Add embeddings to `self.faiss_index`. If using `IndexIDMap`, store an ID mapping to the index in `self.chunks_metadata_list`.
            *   Append `chunks_with_meta` to `self.chunks_metadata_list`.
        *   `similarity_search_with_scores()`:
            *   Converts `query_embedding` to a NumPy array.
            *   Searches `self.faiss_index.search(query_embedding_np, k)`.
            *   Returns distances and indices.
            *   Uses indices to look up the corresponding `DocumentChunk` from `self.chunks_metadata_list`.
            *   (Optional) Implement metadata filtering if needed (more complex with basic FAISS, easier with ChromaDB/LanceDB).
        *   `save_store()`:
            *   Ensures directory exists.
            *   Saves FAISS index: `faiss.write_index(self.faiss_index, str(self.index_file))`.
            *   Saves `self.chunks_metadata_list` to `chunks_meta.jsonl` (one JSON object per line for scalability).
        *   `load_store()`:
            *   Loads FAISS index: `self.faiss_index = faiss.read_index(str(self.index_file))`.
            *   Loads `chunks_meta.jsonl` into `self.chunks_metadata_list`.

### Step 4: Retriever (`src/rag/retriever.py`)

This module orchestrates the retrieval process.

*   **Purpose:** Convert a natural language query into relevant document chunks.
*   **Key File:** `src/rag/retriever.py`
*   **Implementation Details:**
    *   Class `RagRetriever`:
        *   `__init__(self, embedding_service: BaseEmbeddingService, vector_store_manager: BaseVectorStoreManager)`
        *   `async def retrieve_context_chunks(self, query_text: str, k: int = 5, filter_metadata: Optional[Dict] = None) -> List[DocumentChunk]`:
            1.  `query_embedding = await self.embedding_service.embed_query(query_text)`
            2.  `retrieved_items_with_scores = self.vector_store_manager.similarity_search_with_scores(query_embedding, k=k, filter_metadata=filter_metadata)`
            3.  Extract just the `DocumentChunk` objects from the tuples.
            4.  (Optional Advanced) Implement re-ranking here if desired (e.g., using a cross-encoder or LLM-based re-ranker).
            5.  Return the list of `DocumentChunk`s.

---

## III. Modifying `PhaseAOrchestrator`

The `src/financial_modeling/phase_a_orchestrator.py` will need significant changes.

1.  **New Imports:**
    ```python
    # In phase_a_orchestrator.py
    from ..rag.chunking import chunk_parsed_document, DocumentChunk # Assuming these are defined
    from ..rag.embedding_service import GeminiEmbeddingService # Or your chosen implementation
    from ..rag.vector_store_manager import FaissVectorStoreManager # Or your chosen implementation
    from ..rag.retriever import RagRetriever
    from .. import config as alpy_config # Ensure this is robustly imported
    ```

2.  **`__init__` Changes:**
    *   Accept `store_path_base` from config.
    *   Instantiate `self.embedding_service`, `self.vector_store_manager` (per security ID), and `self.rag_retriever`.
        ```python
        # In PhaseAOrchestrator.__init__
        # ... existing initializations ...
        self.store_path_base = Path(alpy_config.VECTOR_STORE_PATH) # New config needed
        self.security_id_for_store = output_log_path.stem.replace("financial_modeling_orchestrator_", "").replace("orchestrator_", "") # Infer from log path or pass explicitly

        self.embedding_service = GeminiEmbeddingService() # Uses API key from alpy_config internally
        
        # Determine embedding dimension (hardcode or get from embedding model)
        # Example: text-embedding-004 is 768
        embedding_dim = 768 # TODO: Make this configurable or get from embedding_service
        
        self.vector_store_manager = FaissVectorStoreManager(
            store_path_base=self.store_path_base,
            security_id=self.security_id_for_store,
            embedding_dim=embedding_dim
        )
        self.rag_retriever = RagRetriever(self.embedding_service, self.vector_store_manager)
        ```

3.  **New Method: `_prepare_and_index_documents` (or similar):**
    *   This method will be called once per document set (e.g., for a `security_id`).
    *   It checks if a vector store already exists for the `security_id` and document set version.
    *   If not, or if re-indexing is forced:
        1.  Takes `parsed_doc_content` (from `MCPDocumentParserTool`).
        2.  Calls `chunk_parsed_document` to get `List[DocumentChunk]`.
        3.  Extracts text from chunks: `texts = [chunk.text_content for chunk in document_chunks]`.
        4.  Gets embeddings: `embeddings = await self.embedding_service.embed_documents(texts)`.
        5.  Adds to vector store: `self.vector_store_manager.add_chunks_and_embeddings(document_chunks, embeddings)`.
        6.  Saves the store: `self.vector_store_manager.save_store()`.
    *   If store exists, it loads it: `self.vector_store_manager.load_store()`.

4.  **Major Overhaul of `_get_context_for_item`:**
    *   This function will be *completely replaced*.
    *   The new version will be simpler at this level:
        ```python
        # In PhaseAOrchestrator
        async def _get_rag_context_for_llm(
            self,
            query_elements: List[str], # e.g., [item_name_en, item_name_mn, period_label, section_hint]
            k_retrieved_chunks: int = 5,
            max_context_length: int = 10000 # Max chars for final context string
        ) -> str:
            # Filter out None or empty strings from query_elements
            valid_query_elements = [qe for qe in query_elements if qe and isinstance(qe, str) and qe.strip()]
            if not valid_query_elements:
                self.logger.warning("No valid query elements provided for RAG context retrieval.")
                return "No specific query terms provided to retrieve context."

            query_text = " ".join(valid_query_elements)
            self.logger.debug(f"RAG Retriever Query: '{query_text}'")

            try:
                retrieved_chunks = await self.rag_retriever.retrieve_context_chunks(query_text, k=k_retrieved_chunks)
            except Exception as e_retrieve:
                self.logger.error(f"Error during RAG retrieval for query '{query_text}': {e_retrieve}", exc_info=True)
                return f"Error retrieving context: {e_retrieve}"

            if not retrieved_chunks:
                self.logger.warning(f"No chunks retrieved by RAG for query: '{query_text}'")
                return "No relevant context chunks found by RAG for the query."

            context_parts = []
            current_length = 0
            for chunk in retrieved_chunks:
                chunk_header = f"\n--- Source: {chunk.document_name}, Page: {chunk.page_number or 'N/A'}, Type: {chunk.chunk_type} ---\n"
                snippet_to_add = chunk_header + chunk.text_content
                if current_length + len(snippet_to_add) > max_context_length:
                    remaining_space = max_context_length - current_length
                    if remaining_space > len(chunk_header) + 50: # Min space for header + some content
                        context_parts.append(chunk_header + chunk.text_content[:remaining_space - len(chunk_header)])
                    self.logger.warning(f"RAG context truncated for query '{query_text}'. Max length {max_context_length} reached.")
                    break
                context_parts.append(snippet_to_add)
                current_length += len(snippet_to_add)
            
            final_context = "".join(context_parts)
            self.logger.debug(f"Final RAG context for query '{query_text}' (len {len(final_context)}):\n{final_context[:300]}...")
            return final_context
        ```

5.  **Update Callers (`_extract_metadata`, `_identify_historical_periods`, `_extract_historical_line_item_data`):**
    *   These methods will no longer use `anchor_pages` or `task_type` directly for context.
    *   They will construct a list of query elements and call `_get_rag_context_for_llm`.
    *   Example change in `_extract_metadata`:
        ```python
        # In _extract_metadata
        # OLD: context = self._get_context_for_item(...)
        # NEW:
        query_elements_meta = [
            "company identification", "company name", "ticker symbol", "currency", 
            "fiscal year end", "reporting period", document_type,
            "компанийн танилцуулга", "валют", "санхүүгийн жил", "тайлант үе" # Add Mongolian hints
        ]
        context = await self._get_rag_context_for_llm(query_elements_meta, k_retrieved_chunks=3, max_context_length=8000)
        prompt_args = {"context": context, "document_type": document_type}
        # ... rest of the logic ...
        ```
    *   Similarly for `_identify_historical_periods`:
        ```python
        # In _identify_historical_periods
        query_elements_periods = [
            "financial statements column headers", "years", "periods", "table of contents for statements",
            "санхүүгийн тайлангийн баганын гарчиг", "он", "үе", "гарчиг" # Add Mongolian hints
        ]
        context = await self._get_rag_context_for_llm(query_elements_periods, k_retrieved_chunks=7, max_context_length=12000)
        prompt_args = {"context": context, "document_type": document_type}
        # ... rest of the logic ...
        ```
    *   And in `_extract_historical_line_item_data`:
        ```python
        # In _extract_historical_line_item_data
        # OLD: context = self._get_context_for_item(...)
        # NEW:
        query_elements_line_item = [
            item_name_en, item_name_mn, period_label, 
            f"{item_name_en} value", f"{item_name_mn} утга", # More specific query
            "financial table", "statement data" # General hints
        ]
        context = await self._get_rag_context_for_llm(query_elements_line_item, k_retrieved_chunks=5) # Default max_context_length
        prompt_args = {"period_label": period_label, ... , "context": context}
        # ... rest of the logic ...
        ```

6.  **`run_phase_a_extraction` Main Flow Update:**
    *   Call `_prepare_and_index_documents(parsed_doc_content, primary_doc_name_or_id)` at the beginning.
    *   Remove `anchor_pages` variable and its passing around, as the new context method doesn't need it directly from `run_phase_a_extraction`.

---

## IV. Configuration (`src/config.py`)

Add the following new configurations:

```python
# --- RAG Configuration ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-004") # Or a newer Gemini embedding model
VECTOR_STORE_BASE_PATH = os.getenv("VECTOR_STORE_BASE_PATH", str(Path(__file__).parent.parent / "data" / "vector_stores")) # Points to Alpy/data/vector_stores
RAG_NUM_RETRIEVED_CHUNKS = int(os.getenv("RAG_NUM_RETRIEVED_CHUNKS", 5))
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", 1000)) # Chars for text chunker
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", 150)) # Chars for text chunker

# Ensure the VECTOR_STORE_BASE_PATH directory exists
Path(VECTOR_STORE_BASE_PATH).mkdir(parents=True, exist_ok=True)
```
Update `GeminiEmbeddingService` and `FaissVectorStoreManager` to use these configs.

---

## V. Workflow Summary with RAG

1.  **`FinancialPhaseATool` receives request.**
2.  It calls `PhaseAOrchestrator.run_phase_a_extraction`.
3.  **Orchestrator:**
    a.  Gets `parsed_doc_content` from `MCPDocumentParserTool`.
    b.  **NEW RAG PREP:** Calls `_prepare_and_index_documents`:
        i.  Checks if vector store for `security_id` (+ document version/hash) exists.
        ii. If not:
            1.  `chunking.chunk_parsed_document()` creates `DocumentChunk`s.
            2.  `embedding_service.embed_documents()` gets embeddings.
            3.  `vector_store_manager.add_chunks_and_embeddings()` builds/updates the index.
            4.  `vector_store_manager.save_store()`.
        iii. If exists: `vector_store_manager.load_store()`.
    c.  Proceeds with `_extract_metadata`, `_identify_historical_periods`, etc.
    d.  Inside these extraction methods:
        i.  A query string is formed.
        ii. `_get_rag_context_for_llm` is called.
        iii. `_get_rag_context_for_llm` uses `self.rag_retriever` to fetch relevant chunks.
        iv.  The text from these chunks becomes the `context` for the LLM prompt.
    e.  LLM is called, response processed, model populated.
    f.  `QualityChecker` runs.

---

## VI. Important Considerations & Next Steps

*   **Chunking Strategy:** This is critical. Experiment with different chunk sizes, overlap, and methods (e.g., table-specific chunking).
*   **Embedding Model:** The quality of embeddings directly impacts retrieval.
*   **Query Formulation:** How you construct the `query_text` from item names, hints, etc., is key to good retrieval.
*   **Vector Store Choice:** FAISS is good for starting. For more features (metadata filtering, persistence, scalability), explore ChromaDB, LanceDB, or cloud-based options later.
*   **Re-indexing:** Implement a strategy for when to re-index documents (e.g., if the source PDF changes). This could involve hashing the document content.
*   **Cost (Embeddings):** Embedding large documents can incur API costs if using a paid embedding service.
*   **Evaluation:** Systematically evaluate the RAG pipeline's impact on extraction accuracy.

This guide provides a comprehensive roadmap. Start by implementing the core RAG components and then iteratively refine chunking, retrieval, and query strategies.