# src/rag/embedding_service.py
from abc import ABC, abstractmethod
from typing import List

# Attempt to import DocumentChunk, but make it optional for now
# as the file might be created in a different order or context.
try:
    from .chunking import DocumentChunk
except ImportError:
    DocumentChunk = None  # Or a placeholder type

import google.generativeai as genai

# Assuming your API key and model name will be managed via a config object or environment variables later.
# For now, these are placeholders. You'll need to integrate this with your actual config system.
# from src.config import alpy_config # Example of how you might import your config

class BaseEmbeddingService(ABC):
    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        pass

class GeminiEmbeddingService(BaseEmbeddingService):
    def __init__(self, api_key: str, model_name: str = "text-embedding-004"):
        """
        Initializes the RAG Gemini Embedding Service.
        Args:
            api_key: The Google AI API key.
            model_name: The name of the embedding model to use (e.g., "text-embedding-004").
        """
        self.model_name = model_name
        # Configure the genai library with the API key.
        # Ensure this is done securely and ideally only once in your application's lifecycle.
        genai.configure(api_key=api_key)

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of documents (text chunks).
        Args:
            texts: A list of strings, where each string is a document/chunk to embed.
        Returns:
            A list of embeddings, where each embedding is a list of floats.
        """
        if not texts:
            return []
        
        # The Gemini API for embeddings typically handles batching internally to some extent,
        # but for very large lists, you might need to batch them manually.
        # Example: genai.embed_content(model=self.model_name, content=texts, task_type="RETRIEVAL_DOCUMENT")
        # The 'content' parameter can be a list of strings.
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=texts,
                task_type="RETRIEVAL_DOCUMENT" # or "SEMANTIC_SIMILARITY" / "CLUSTERING" depending on use case
            )
            return result['embedding'] # The structure of the response might vary slightly
        except Exception as e:
            # Log the error appropriately
            print(f"Error embedding documents with Gemini: {e}")
            # Consider how to handle partial failures or retries
            raise # Re-raise the exception or handle it as per your error strategy

    async def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query string.
        Args:
            text: The query string to embed.
        Returns:
            The embedding for the query as a list of floats.
        """
        if not text:
            return [] # Or raise an error, depending on desired behavior
        
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="RETRIEVAL_QUERY"
            )
            return result['embedding']
        except Exception as e:
            # Log the error appropriately
            print(f"Error embedding query with Gemini: {e}")
            raise

# Example usage (for testing, to be removed or placed in a test file):
# async def main():
#     # This requires GOOGLE_API_KEY to be set in the environment or passed directly
#     # from src.config import alpy_config # Assuming alpy_config.GOOGLE_API_KEY exists
#     api_key_from_config = "YOUR_GOOGLE_API_KEY" # Replace with actual key loading
#     if not api_key_from_config:
#         print("GOOGLE_API_KEY not found in config. Please set it.")
#         return

#     embedder = GeminiEmbeddingService(api_key=api_key_from_config)
#     docs = ["This is a test document.", "Another document for testing embeddings."]
#     query = "What is this test about?"

#     doc_embeddings = await embedder.embed_documents(docs)
#     print(f"Document Embeddings (first 5 dims of first doc): {doc_embeddings[0][:5]}")
#     print(f"Number of document embeddings: {len(doc_embeddings)}")

#     query_embedding = await embedder.embed_query(query)
#     print(f"Query Embedding (first 5 dims): {query_embedding[:5]}")

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())
