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
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s][%(levelname)s][MCPDocumentParserServer] %(message)s')
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
    logger.debug(f"Received parse_document request for {input_data.document_path}")
    file_path = Path(input_data.document_path)
    if not file_path.is_file():
        logger.error(f"Document not found: {input_data.document_path}")
        return DocumentParseOutput(status="error", message=f"Document not found: {input_data.document_path}")

    status = "error"
    message = "Unknown error during parsing."
    total_pages = 0
    processed_pages_count = 0
    pages_content: List[PageContent] = []
    all_tables: List[TableData] = []
    all_images_ocr_results: List[ImageData] = []

    tesseract_available_this_run = TESSERACT_INITIALLY_OK
    tesseract_initial_check_message_this_run = TESSERACT_INIT_MSG

    fitz_doc: Optional[fitz.Document] = None # Define fitz_doc here for broader scope

    try:
        logger.debug(f"Attempting to open PDF with pdfplumber and fitz: {file_path}")
        fitz_doc = fitz.open(file_path) # Open with fitz once
        
        with pdfplumber.open(file_path) as pdf_plumber_doc:
            total_pages = len(pdf_plumber_doc.pages)
            logger.debug(f"Total pages in document: {total_pages}")

            pages_to_process_plumber = pdf_plumber_doc.pages
            if input_data.max_pages_to_process_for_testing is not None:
                pages_to_process_plumber = pdf_plumber_doc.pages[:input_data.max_pages_to_process_for_testing]
                logger.debug(f"Processing limited to {len(pages_to_process_plumber)} pages for testing.")

            for page_index, plumber_page in enumerate(pages_to_process_plumber):
                processed_pages_count += 1
                logger.debug(f"Processing page {plumber_page.page_number}")
                page_content = PageContent(page_number=plumber_page.page_number)
                
                fitz_page = fitz_doc.load_page(plumber_page.page_number - 1) # fitz is 0-indexed

                # PyMuPDF text extraction
                try:
                    page_content.text_pymupdf = fitz_page.get_text()
                    logger.debug(f"PyMuPDF text extracted for page {plumber_page.page_number}")
                except Exception as e:
                    logger.warning(f"PyMuPDF text extraction failed for page {plumber_page.page_number}: {e}")
                
                # Table extraction with pdfplumber
                logger.debug(f"Extracting tables for page {plumber_page.page_number}")
                page_tables = plumber_page.extract_tables()
                for table_idx, table_data in enumerate(page_tables):
                    all_tables.append(TableData(page_number=plumber_page.page_number, table_index_on_page=table_idx, rows=table_data))
                logger.debug(f"Extracted {len(page_tables)} tables from page {plumber_page.page_number}")

                # OCR processing
                if input_data.ocr_mode != "none" and tesseract_available_this_run:
                    logger.debug(f"Starting OCR for page {plumber_page.page_number} with mode: {input_data.ocr_mode}")
                    if input_data.ocr_mode == "force_full_pages":
                        logger.debug(f"Performing full page OCR for page {plumber_page.page_number}")
                        try:
                            pix = fitz_page.get_pixmap(dpi=input_data.dpi_for_full_page_ocr)
                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            page_content.page_full_ocr_text = pytesseract.image_to_string(img, lang=input_data.ocr_lang)
                            logger.debug(f"Full page OCR completed for page {plumber_page.page_number}")
                        except Exception as e:
                            page_content.page_full_ocr_error = str(e)
                            logger.error(f"Full page OCR failed for page {plumber_page.page_number}: {e}")
                    elif input_data.ocr_mode == "force_images" or \
                        (input_data.ocr_mode == "auto" and (page_content.text_pymupdf is None or len(page_content.text_pymupdf) < input_data.min_chars_for_text_page_auto_ocr)):
                        logger.debug(f"Performing image OCR for page {plumber_page.page_number} using fitz image list")
                        
                        # Use fitz to get image list
                        image_list_fitz = fitz_page.get_images(full=True)
                        logger.debug(f"Found {len(image_list_fitz)} images on page {plumber_page.page_number} via fitz.")

                        for img_idx, img_info_fitz in enumerate(image_list_fitz):
                            img_xref = img_info_fitz[0] # xref is the first item
                            logger.debug(f"Processing image xref {img_xref} (index {img_idx}) on page {plumber_page.page_number}")
                            try:
                                base_image = fitz_doc.extract_image(img_xref)
                                img_bytes = base_image['image']
                                img_ext = base_image['ext']
                                img = Image.open(io.BytesIO(img_bytes))

                                ocr_text = pytesseract.image_to_string(img, lang=input_data.ocr_lang)
                                
                                # Get image coordinates (bbox) from fitz if possible, otherwise None
                                # Note: fitz_page.get_image_bbox(img_info_fitz) might be an option, or use rect from draw_rect for more precision if needed.
                                # For now, we'll leave coordinates as None as direct bbox from get_images is not straightforward like pdfplumber's page.images
                                
                                all_images_ocr_results.append(ImageData(
                                    page_number=plumber_page.page_number,
                                    image_index_on_page=img_idx, # This is now fitz's image index for the page
                                    ocr_text=ocr_text,
                                    ocr_lang_used=input_data.ocr_lang,
                                    image_extension=img_ext,
                                    # image_coordinates= # TODO: Explore getting bbox from fitz image info if necessary
                                ))
                                logger.debug(f"Image OCR completed for image xref {img_xref} on page {plumber_page.page_number}")
                            except Exception as e:
                                all_images_ocr_results.append(ImageData(page_number=plumber_page.page_number, image_index_on_page=img_idx, ocr_error=str(e)))
                                logger.warning(f"Image OCR failed for image xref {img_xref} on page {plumber_page.page_number}: {e}")
                
                pages_content.append(page_content)

        status = "success"
        message = "Document parsed successfully."
        logger.info(f"Document parsing completed successfully for {file_path}.")

    except fitz.fitz.FitzError as fe:
        status = "error"
        message = f"Fitz (PyMuPDF) error: {fe}"
        logger.error(f"Fitz (PyMuPDF) error processing {file_path}: {fe}", exc_info=True)
    except pdfplumber.exceptions.PDFSyntaxError as pe:
        status = "error"
        message = f"PDFPlumber syntax error: {pe}"
        logger.error(f"PDFPlumber syntax error processing {file_path}: {pe}", exc_info=True)
    except Exception as e:
        status = "error"
        message = f"An unexpected error occurred: {e}"
        logger.error(f"Unexpected error processing {file_path}: {e}", exc_info=True)
    finally:
        if fitz_doc:
            fitz_doc.close()
            logger.debug(f"Closed fitz document for {file_path}")
    
    return DocumentParseOutput(
        status=status,
        message=message,
        file_name=file_path.name,
        total_pages=total_pages,
        processed_pages_count=processed_pages_count,
        pages_content=pages_content,
        all_tables=all_tables,
        all_images_ocr_results=all_images_ocr_results,
        tesseract_available=tesseract_available_this_run,
        tesseract_initial_check_message=tesseract_initial_check_message_this_run
    )


@mcp_app.tool("main_test_interactive")
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
        logger.info("MCPDocumentParserServer script is starting up and configuring FastMCP app.")
mcp_app.run()