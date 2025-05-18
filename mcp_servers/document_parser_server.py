import asyncio
import io
import logging
import json
import sys
import traceback
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import pytesseract # Requires Tesseract OCR to be installed
from pydantic import BaseModel, Field

from mcp.server.fastmcp import FastMCP

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s][MCPDocumentParserServer] %(message)s')
logger = logging.getLogger(__name__)

mcp_app = FastMCP(
    name="MCPDocumentParserServer",
    version="0.1.0",
    description="MCP server for parsing documents (PDFs initially), extracting text, tables, and performing OCR."
)

# --- Helper Functions ---
def check_tesseract(lang_to_check: str = 'mon'):
    """Checks if Tesseract is installed and if the specified language pack is available."""
    try:
        langs = pytesseract.get_languages(config='')
        if lang_to_check not in langs:
            msg = f"Tesseract is installed, but '{lang_to_check}' language pack not found. Available: {langs}"
            logger.warning(msg)
            return False, msg
        logger.info(f"Tesseract OK. '{lang_to_check}' available. All languages: {langs}")
        return True, f"Tesseract OK, '{lang_to_check}' available."
    except pytesseract.TesseractNotFoundError:
        msg = "Tesseract is not installed or not in your PATH. OCR functionality will be unavailable."
        logger.error(msg)
        return False, msg
    except Exception as e:
        msg = f"Error checking Tesseract languages: {e}"
        logger.error(msg)
        return False, msg

TESSERACT_INITIALLY_OK, TESSERACT_INIT_MSG = check_tesseract()

# --- Pydantic Models ---
class TableData(BaseModel):
    page_number: int
    table_index_on_page: int
    rows: List[List[Optional[str]]] = Field(description="List of rows, where each row is a list of cell texts.")
    extraction_method: str = Field(default="pdfplumber", description="Method used for table extraction.")
    notes: Optional[str] = Field(None, description="Notes regarding table extraction, e.g., if any issues occurred.")

class ImageData(BaseModel):
    page_number: int
    image_index_on_page: int # Index of image on its page from fitz
    ocr_text: Optional[str] = None
    ocr_lang_used: Optional[str] = None
    ocr_error: Optional[str] = None

class PageContent(BaseModel):
    page_number: int
    text_pymupdf: Optional[str] = Field(None, description="Text extracted by PyMuPDF for the entire page.")
    page_full_ocr_text: Optional[str] = None
    page_full_ocr_error: Optional[str] = None

class DocumentParseInput(BaseModel):
    document_path: str = Field(description="Absolute path to the document file.")
    ocr_mode: Literal["auto", "force_images", "force_full_pages", "none"] = Field(default="auto", description="OCR mode: 'auto' (OCR images if low text), 'force_images' (OCR all extracted images), 'force_full_pages' (OCR each page as an image), 'none'.")
    ocr_lang: str = Field(default="mon", description="Tesseract language string for OCR (e.g., 'eng', 'mon').")
    min_chars_for_text_page_auto_ocr: int = Field(default=100, description="In 'auto' mode, min characters on a page (from PyMuPDF) to consider it a text page (and skip image OCR).")
    dpi_for_full_page_ocr: int = Field(default=300, description="DPI to use when rendering a page to an image for 'force_full_pages' OCR.")
    max_pages_to_process_for_testing: Optional[int] = Field(default=None, description="For testing: limits processing to this many pages from the start of the document.")

class DocumentParseOutput(BaseModel):
    status: Literal["success", "error"]
    message: Optional[str] = None
    file_name: Optional[str] = None
    total_pages: int = 0 # Actual total pages in the document
    processed_pages_count: int = 0 # Number of pages actually processed
    pages_content: List[PageContent] = []
    all_tables: List[TableData] = []
    all_images_ocr_results: List[ImageData] = []
    tesseract_available: bool = True
    tesseract_initial_check_message: Optional[str] = None

# --- MCP Tool Implementation ---
@mcp_app.tool(name="parse_document")
async def parse_document_tool(input_data: DocumentParseInput) -> DocumentParseOutput:
    doc_path = Path(input_data.document_path)
    if not doc_path.exists() or not doc_path.is_file():
        return DocumentParseOutput(
            status="error",
            message=f"Document not found or is not a file: {doc_path}",
            tesseract_available=TESSERACT_INITIALLY_OK,
            tesseract_initial_check_message=TESSERACT_INIT_MSG,
            processed_pages_count=0
        )

    output_pages_content: List[PageContent] = []
    output_tables: List[TableData] = []
    output_images_ocr: List[ImageData] = []
    
    current_run_tesseract_ok = TESSERACT_INITIALLY_OK

    fitz_doc = None
    pdfplumber_doc = None
    actual_total_pages = 0
    processed_pages_count = 0

    try:
        fitz_doc = fitz.open(doc_path)
        actual_total_pages = len(fitz_doc)
        try:
            pdfplumber_doc = pdfplumber.open(doc_path)
        except Exception as e_plumber_open:
            logger.warning(f"Could not open document with pdfplumber: {e_plumber_open}. Table extraction will be skipped.")
    except Exception as e:
        logger.error(f"Error opening document {doc_path}: {e}", exc_info=True)
        if fitz_doc: fitz_doc.close()
        if pdfplumber_doc: pdfplumber_doc.close()
        return DocumentParseOutput(
            status="error",
            message=f"Failed to open document: {str(e)}",
            tesseract_available=current_run_tesseract_ok,
            tesseract_initial_check_message=TESSERACT_INIT_MSG,
            total_pages=actual_total_pages,
            processed_pages_count=processed_pages_count
        )

    pages_to_iterate = actual_total_pages
    if input_data.max_pages_to_process_for_testing is not None and input_data.max_pages_to_process_for_testing > 0:
        pages_to_iterate = min(actual_total_pages, input_data.max_pages_to_process_for_testing)
        logger.info(f"Limiting processing to first {pages_to_iterate} pages of {actual_total_pages} total (due to max_pages_to_process_for_testing).")


    for i in range(pages_to_iterate):
        page_num = i + 1
        processed_pages_count += 1
        fitz_page = fitz_doc.load_page(i)
        
        page_text_pymupdf = fitz_page.get_text("text")
        current_page_obj = PageContent(page_number=page_num, text_pymupdf=page_text_pymupdf)

        # Table Extraction
        if pdfplumber_doc and i < len(pdfplumber_doc.pages):
            try:
                plumber_page = pdfplumber_doc.pages[i]
                extracted_tables_on_page = plumber_page.extract_tables()
                if extracted_tables_on_page:
                    for table_idx, table_data_rows in enumerate(extracted_tables_on_page):
                        cleaned_rows = []
                        if table_data_rows:
                            for row in table_data_rows:
                                if row: 
                                    cleaned_row = [str(cell) if cell is not None else None for cell in row]
                                    cleaned_rows.append(cleaned_row)
                                else: 
                                    cleaned_rows.append([]) 
                        output_tables.append(TableData(
                            page_number=page_num, table_index_on_page=table_idx,
                            rows=cleaned_rows, extraction_method="pdfplumber"
                        ))
            except Exception as e_table:
                logger.warning(f"Error extracting tables from page {page_num} with pdfplumber: {e_table}")
                output_tables.append(TableData(
                    page_number=page_num, table_index_on_page=-1, rows=[],
                    extraction_method="pdfplumber", notes=f"Extraction failed: {str(e_table)}"
                ))
        
        # OCR Logic
        if not current_run_tesseract_ok:
            pass 
        elif input_data.ocr_mode == "force_full_pages":
            try:
                pix = fitz_page.get_pixmap(dpi=input_data.dpi_for_full_page_ocr)
                img_bytes = pix.tobytes("png")
                pil_image = Image.open(io.BytesIO(img_bytes))
                current_page_obj.page_full_ocr_text = pytesseract.image_to_string(pil_image, lang=input_data.ocr_lang).strip()
            except pytesseract.TesseractNotFoundError:
                current_run_tesseract_ok = False; current_page_obj.page_full_ocr_error = TESSERACT_INIT_MSG
            except Exception as e_ocr_full:
                current_page_obj.page_full_ocr_error = f"Full page OCR error: {str(e_ocr_full)}"
                logger.warning(f"Error OCR'ing full page {page_num}: {e_ocr_full}")
        
        elif input_data.ocr_mode in ["auto", "force_images"]:
            perform_ocr_on_images = False
            if input_data.ocr_mode == "force_images":
                perform_ocr_on_images = True
            elif input_data.ocr_mode == "auto":
                if len(page_text_pymupdf or "") < input_data.min_chars_for_text_page_auto_ocr and fitz_page.get_images(full=True):
                    perform_ocr_on_images = True
            
            if perform_ocr_on_images:
                page_images_info = fitz_page.get_images(full=True)
                for img_idx, img_info in enumerate(page_images_info):
                    if not current_run_tesseract_ok: break
                    xref = img_info[0]
                    try:
                        base_image = fitz_doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        ocr_text_result = pytesseract.image_to_string(pil_image, lang=input_data.ocr_lang).strip()
                        output_images_ocr.append(ImageData(
                            page_number=page_num, image_index_on_page=img_idx,
                            ocr_text=ocr_text_result, ocr_lang_used=input_data.ocr_lang
                        ))
                    except pytesseract.TesseractNotFoundError:
                        current_run_tesseract_ok = False
                        output_images_ocr.append(ImageData(
                            page_number=page_num, image_index_on_page=img_idx,
                            ocr_error=TESSERACT_INIT_MSG, ocr_lang_used=input_data.ocr_lang
                        ))
                    except Exception as e_ocr_img:
                        ocr_error_msg = f"Image OCR error: {str(e_ocr_img)}"
                        logger.warning(f"Error OCR'ing image {img_idx} on page {page_num}: {e_ocr_img}")
                        output_images_ocr.append(ImageData(
                            page_number=page_num, image_index_on_page=img_idx,
                            ocr_error=ocr_error_msg, ocr_lang_used=input_data.ocr_lang
                        ))
        output_pages_content.append(current_page_obj)

    if fitz_doc: fitz_doc.close()
    if pdfplumber_doc: pdfplumber_doc.close()

    return DocumentParseOutput(
        status="success",
        message="Document parsed.",
        file_name=doc_path.name,
        total_pages=actual_total_pages,
        processed_pages_count=processed_pages_count,
        pages_content=output_pages_content,
        all_tables=output_tables,
        all_images_ocr_results=output_images_ocr,
        tesseract_available=current_run_tesseract_ok,
        tesseract_initial_check_message=TESSERACT_INIT_MSG
    )

async def main_test_interactive():
    print("--- Document Parser Server Standalone Interactive Test ---")
    print(f"Initial Tesseract Check: {TESSERACT_INIT_MSG} (Status: {'OK' if TESSERACT_INITIALLY_OK else 'Problem'})")

    try:
        file_path_input = input("Enter full PDF path: ").strip()
        if not file_path_input: print("Exiting."); return
        file_path = Path(file_path_input)
        if not file_path.exists(): print(f"File not found: {file_path}"); return

        ocr_modes = ["auto", "force_images", "force_full_pages", "none"]
        ocr_mode_input = input(f"OCR mode ({'/'.join(ocr_modes)}) [auto]: ").strip().lower() or "auto"
        if ocr_mode_input not in ocr_modes: ocr_mode_input = "auto"; print(f"Invalid OCR mode, using '{ocr_mode_input}'.")
        
        ocr_lang_input = input("OCR language (e.g., mon, eng) [mon]: ").strip().lower() or "mon"
        dpi_input = input("DPI for full page OCR (if used) [300]: ").strip()
        dpi = int(dpi_input) if dpi_input.isdigit() else 300
        
        max_pages_test_input = input("Max pages to process for this test [5, 0 for all]: ").strip()
        max_pages_for_this_test = 5
        if max_pages_test_input.isdigit():
            val = int(max_pages_test_input)
            if val == 0:
                max_pages_for_this_test = None # Process all
            elif val > 0:
                max_pages_for_this_test = val
        
        print(f"Using max_pages_to_process_for_testing = {max_pages_for_this_test or 'All'}")


        test_input = DocumentParseInput(
            document_path=str(file_path), ocr_mode=ocr_mode_input, #type: ignore
            ocr_lang=ocr_lang_input, dpi_for_full_page_ocr=dpi,
            max_pages_to_process_for_testing=max_pages_for_this_test 
        )
        result = await parse_document_tool(test_input)
        
        print("\n--- Result ---")
        print(f"Status: {result.status}, Message: {result.message}")
        print(f"File: {result.file_name}, Total Pages in Doc: {result.total_pages}, Pages Processed: {result.processed_pages_count}")
        print(f"Tesseract Status (this run): {'OK' if result.tesseract_available else 'Problem'}")

        print(f"\nText ({len(result.pages_content)} pages processed, showing first 3, 100 chars):")
        for i, p_content in enumerate(result.pages_content):
            if i >= 3: print(f"  ...and {len(result.pages_content) - 3} more processed pages content."); break
            print(f"  P{p_content.page_number}: PyMuPDF='{(p_content.text_pymupdf or '')[:100]}...' FullPageOCR='{(p_content.page_full_ocr_text or '')[:50]}...'")
        
        print(f"\nTables ({len(result.all_tables)} total from processed pages):")
        for i, tbl in enumerate(result.all_tables):
            if i >= 2: print(f"  ...and {len(result.all_tables) - 2} more tables."); break
            print(f"  P{tbl.page_number}, Idx{tbl.table_index_on_page}: {len(tbl.rows)} rows. Notes: {tbl.notes or 'OK'}")

        print(f"\nImage OCRs ({len(result.all_images_ocr_results)} total from processed pages):")
        for i, img_ocr in enumerate(result.all_images_ocr_results):
            if i >= 3: print(f"  ...and {len(result.all_images_ocr_results) - 3} more."); break
            err_info = f", Error: {img_ocr.ocr_error}" if img_ocr.ocr_error else ""
            print(f"  P{img_ocr.page_number}, ImgIdx{img_ocr.image_index_on_page}: '{(img_ocr.ocr_text or '')[:50]}...' (Lang: {img_ocr.ocr_lang_used}{err_info})")
        
        # print("\nFull JSON Output (for processed pages):")
        # print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"Error during test: {e}"); traceback.print_exc()

if __name__ == "__main__":
    if sys.platform == "win32" and sys.version_info >= (3,8) : # Proactor policy for Windows
        try:
            if not isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsProactorEventLoopPolicy):
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        except Exception as e_pol: logger.warning(f"Could not set WindowsProactorEventLoopPolicy: {e_pol}")
    
    if len(sys.argv) > 1 and sys.argv[1].lower() == "test":
        asyncio.run(main_test_interactive())
    else:
        logger.info(f"Starting MCPDocumentParserServer. Initial Tesseract Check: {TESSERACT_INIT_MSG} (Status: {'OK' if TESSERACT_INITIALLY_OK else 'Problem'})")
        mcp_app.run()