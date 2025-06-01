# src/rag/chunking.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging

# Configure a logger for this module if not already configured by the application
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    # Basic configuration if no handlers are set up by the main application
    # This is helpful for standalone testing or if the module is used in different contexts
    # In a larger application, logging is typically configured centrally.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s')

class DocumentChunk(BaseModel):
    chunk_id: str  # Unique ID for the chunk (e.g., f"{doc_name}_page_{page_num}_chunk_{idx}")
    text_content: str
    document_name: str
    page_number: Optional[int] = None
    table_index_on_page: Optional[int] = None  # If chunk is from a table
    chunk_type: str  # e.g., "text_snippet", "table_row", "full_table_text"
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Placeholder for chunking strategies
def chunk_parsed_document(parsed_doc_content: Dict[str, Any], document_id: str, chunk_size: int, overlap_size: int, strategy: Optional[str] = "simple_split", min_chunk_size_chars: Optional[int] = None) -> List[DocumentChunk]:
    """Chunks parsed document content into DocumentChunk objects based on page text.

    Args:
        parsed_doc_content: The dictionary containing parsed PDF data, 
                              expected to have 'pages_content' with 'text_pymupdf'.
        doc_name: The name of the document being chunked.
        config: A dictionary containing 'chunk_size' and 'chunk_overlap'.

    Returns:
        A list of DocumentChunk objects.
    """
    if not parsed_doc_content or not isinstance(parsed_doc_content, dict):
        logger.warning(f"Invalid parsed_doc_content for {document_id}. Expected a dictionary.")
        return []
    logger.info(f"Starting chunk_parsed_document for doc: {document_id} with strategy: {strategy}")
    logger.debug(f"CHUNK_DEBUG Received parsed_doc_content type: {type(parsed_doc_content)}, keys: {list(parsed_doc_content.keys()) if isinstance(parsed_doc_content, dict) else ('Not a dict or None' if parsed_doc_content is not None else 'None')}")
    logger.debug(f"Chunker params: chunk_size={chunk_size}, chunk_overlap={overlap_size}")

    all_chunks: List[DocumentChunk] = []
    # Parameters chunk_size and chunk_overlap are now direct arguments

    if not parsed_doc_content or 'pages_content' not in parsed_doc_content:
        logger.warning(f"'pages_content' not found in parsed_doc_content for {document_id} or content is None. Returning empty list.")
        return []

    pages_content = parsed_doc_content.get('pages_content', [])
    all_images_ocr = parsed_doc_content.get('all_images_ocr_results', [])
    logger.info(f"Processing {len(pages_content)} pages for document {document_id} (and {len(all_images_ocr)} image OCR results).")
    logger.info(f"First 2 keys in parsed_doc_content: {list(parsed_doc_content.keys())[:2]}")
    
    # Debug the all_images_ocr data structure
    if all_images_ocr and isinstance(all_images_ocr, list) and len(all_images_ocr) > 0:
        first_item = all_images_ocr[0]
        logger.info(f"First image OCR item: page_number={first_item.get('page_number')}, has_text={bool(first_item.get('ocr_text', '').strip())}")
        logger.info(f"First 100 chars of OCR text: {first_item.get('ocr_text', '')[:100]}")
    else:
        logger.warning(f"No valid all_images_ocr data found. Type: {type(all_images_ocr)}, Empty: {len(all_images_ocr) == 0}")

    for i, page_data_from_pages_content in enumerate(pages_content):
        page_number = page_data_from_pages_content.get('page_number', i + 1)
        
        current_page_text_sources = []
        text_source_log_parts = [] # For logging which sources contributed

        # 1. Get text from pages_content[i].text_pymupdf
        pymupdf_text = page_data_from_pages_content.get('text_pymupdf', '')
        if isinstance(pymupdf_text, str) and pymupdf_text.strip():
            current_page_text_sources.append(pymupdf_text)
            text_source_log_parts.append("PyMuPDF")
        elif not isinstance(pymupdf_text, str) and pymupdf_text is not None:
            logger.warning(f"Page {page_number}: 'text_pymupdf' is not a string (type: {type(pymupdf_text)}). Value: '{str(pymupdf_text)[:50]}'.")

        # 2. Get text from pages_content[i].page_full_ocr_text
        page_ocr_text = page_data_from_pages_content.get('page_full_ocr_text', '')
        if isinstance(page_ocr_text, str) and page_ocr_text.strip():
            current_page_text_sources.append(page_ocr_text)
            text_source_log_parts.append("PageFullOCR")
        elif not isinstance(page_ocr_text, str) and page_ocr_text is not None:
            logger.warning(f"Page {page_number}: 'page_full_ocr_text' is not a string (type: {type(page_ocr_text)}). Value: '{str(page_ocr_text)[:50]}'.")

        # 3. Get text from all_images_ocr_results for this page
        img_ocr_text_list = []
        logger.info(f"Processing page {page_number}: searching in {len(all_images_ocr)} OCR results")
        if all_images_ocr and isinstance(all_images_ocr, list):
            for img_ocr_data in all_images_ocr:
                if isinstance(img_ocr_data, dict) and img_ocr_data.get('page_number') == page_number:
                    ocr_text = img_ocr_data.get('ocr_text', '')
                    if isinstance(ocr_text, str) and ocr_text.strip():
                        img_ocr_text_list.append(ocr_text)
                        text_source_log_parts.append(f"Image OCR ({img_ocr_data.get('image_index_on_page', 'unknown idx')})")
                    elif not isinstance(ocr_text, str) and ocr_text is not None:
                        logger.warning(f"Page {page_number}, Image OCR item: 'ocr_text' is not a string (type: {type(ocr_text)}). Value: '{str(ocr_text)[:50]}'")

        if img_ocr_text_list:
            current_page_text_sources.extend(img_ocr_text_list)

        # Combine all text sources for this page
        page_text = ' '.join([text for text in current_page_text_sources if text and isinstance(text, str)])
        source_info_str = ', '.join(text_source_log_parts) if text_source_log_parts else "No text found"
        log_text_snippet = page_text[:200] if len(page_text) > 0 else "..."
        logger.info(f"After combining sources for page {page_number}: final text length = {len(page_text)}")
        logger.info(f"CHUNK_LOOP Page {page_number} (Sources: {source_info_str}): Combined text length: {len(page_text)}. Snippet: '{log_text_snippet}...'" )

        # Final check: if page_text is empty after combining all sources, skip.
        if not page_text:
            logger.info(f"Page {page_number}: Combined text from all sources is empty. Skipping page.")
            continue

        page_chunk_idx = 0
        start_index = 0
        while start_index < len(page_text):
            end_index = start_index + chunk_size
            current_chunk_text = page_text[start_index:end_index]

            # Clean up chunk text and check if valid
            stripped_chunk_text = current_chunk_text.strip() if isinstance(current_chunk_text, str) else ""
            if not stripped_chunk_text: # Avoid empty or whitespace-only chunks
                # Advance to next potential chunk position
                start_index += (chunk_size - overlap_size) if chunk_size > overlap_size else chunk_size
                # Safety check to avoid infinite loops
                if start_index >= len(page_text) and end_index < len(page_text):
                    start_index = len(page_text)
                continue

            # Filter out chunks smaller than min_chunk_size_chars if specified
            if min_chunk_size_chars is not None and len(stripped_chunk_text) < min_chunk_size_chars:
                logger.info(f"DISCARDING CHUNK: Page {page_number}, Candidate (ID prefix {document_id}_page_{page_number}_chunk_{page_chunk_idx}) with length {len(stripped_chunk_text)} is smaller than min_chunk_size_chars ({min_chunk_size_chars}).")
                # Advance start_index similar to how it's done after appending a chunk
                if chunk_size <= overlap_size:
                    start_index += chunk_size if chunk_size > 0 else 1 
                else:
                    start_index += (chunk_size - overlap_size)
                if start_index >= len(page_text) and end_index >= len(page_text):
                    break
                continue # Skip this small chunk

            chunk_id = f"{document_id}_page_{page_number}_chunk_{page_chunk_idx}"
            
            try:
                logger.info(f"APPENDING CHUNK: ID: {chunk_id}, Stripped Length: {len(stripped_chunk_text)}, Original Length: {len(current_chunk_text)}, Page: {page_number}")
                all_chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    text_content=current_chunk_text,
                    document_name=document_id,
                    page_number=page_number,
                    chunk_type="text_snippet",
                    metadata={'source_char_start_index': start_index, 'source_char_end_index': min(end_index, len(page_text))}
                ))
                logger.info(f"Successfully added chunk {chunk_id}")
            except Exception as e:
                logger.error(f"Error adding chunk {chunk_id}: {str(e)}")
                continue
            
            page_chunk_idx += 1
            # Move start_index for the next chunk, considering overlap
            # Ensure that overlap doesn't cause an infinite loop if chunk_size <= chunk_overlap
            if chunk_size <= overlap_size:
                 # If overlap is too large, just move by chunk_size to avoid issues, or by a small step
                start_index += chunk_size if chunk_size > 0 else 1 
            else:
                start_index += (chunk_size - overlap_size)
            
            # If the remaining text is smaller than overlap, and we've already processed it, break
            if start_index >= len(page_text) and end_index >= len(page_text):
                 break

    logger.info(f"Finished chunk_parsed_document for doc: {document_id}. Generated {len(all_chunks)} chunks.")
    return all_chunks
