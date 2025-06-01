import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, Literal # Union, Literal added for new models
from uuid import uuid4 # Added for new models

import google.generativeai as genai
from langchain_core.tools import BaseTool, ToolException
from langchain_core.callbacks.manager import AsyncCallbackManagerForToolRun
from pydantic import BaseModel, Field, model_validator, PrivateAttr, RootModel, ValidationError # RootModel added for new models, ValidationError for _arun

# Original relative imports
from ..tools.mcp_document_parser_tool import MCPDocumentParserTool, MCPDocumentParserToolInput
from ..financial_modeling.phase_a_orchestrator import PhaseAOrchestrator
from ..financial_modeling.utils import FinancialModelingUtils
from ..financial_modeling.mock_quality_checker import QualityChecker, QualityReport
# The following two were in original but orchestrator handles them, so not strictly needed by refactored tool actions:
# from ..rag.embedding_service import GeminiEmbeddingService
# from ..rag.vector_store_manager import FaissVectorStoreManager

from .. import config as AlpyConfig # For proper relative import when run with -m
import inspect

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=AlpyConfig.LOG_LEVEL, format=AlpyConfig.LOG_FORMAT)
    logger.setLevel(AlpyConfig.LOG_LEVEL)

# --- Path Configuration (Derived Internally) --- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

FINANCIAL_MODEL_SCHEMA_PATH = PROJECT_ROOT / "src" / "financial_modeling" / "mock_financial_schema.json"
FINANCIAL_MODEL_PROMPTS_PATH = PROJECT_ROOT / "prompts" / "financial_modeling_prompts.yaml"
DEFAULT_SECURITIES_DOCS_BASE_PATH = PROJECT_ROOT / "otcmn_tool_test_output" / "current"

from enum import Enum

class FinancialPhaseAActionType(str, Enum):
    PARSE_DOCUMENTS = "parse_documents"
    EXTRACT_DATA = "extract_data"
    RUN_QUALITY_CHECK = "run_quality_check"

# --- Action-Specific Input Schemas --- #

class InnerFinancialParseDocsAction(BaseModel):
    """Input schema for the parse_documents action.
    
    This action converts PDF/image financial documents into structured text using OCR when needed.
    Documents should be in the security's documents folder: <securities_docs_base_path>/<security_id>/documents/
    """
    action: Literal["parse_documents"] = Field(
        default="parse_documents",
        description="Must be 'parse_documents' to parse financial documents for a security."
    )
    security_id: str = Field(
        description="Unique identifier for the target company/security. Used to locate documents in <base_path>/<security_id>/documents/"
    )
    primary_document_name: Optional[str] = Field(
        default=None,
        description="Optional: Name of the primary financial document (e.g., 'annual_report.pdf') within the security's document folder."
    )
    ocr_mode: str = Field(
        default="auto",
        description="OCR mode: 'auto' (detect if needed), 'force' (always use OCR), or 'off' (never use OCR)."
    )
    ocr_lang: str = Field(
        default="eng",
        description="OCR language(s). Use 'eng' for English, 'mon' for Mongolian, or 'eng+mon' for both."
    )
    max_pages_to_parse_per_doc: Optional[int] = Field(
        default=None,
        description="Optional: Maximum number of pages to parse per document. None means parse all pages."
    )
    securities_docs_base_path_override: Optional[str] = Field(
        default=None,
        description="Optional: Override the default base path (<project_root>/otcmn_tool_test_output/current/secondary_board_B) for locating security documents."
    )

class InnerFinancialExtractDataAction(BaseModel):
    """Input schema for the extract_data action.
    
    This action extracts financial data (line items, values, periods) from previously parsed documents.
    Requires output from the parse_documents action as input.
    """
    action: Literal["extract_data"] = Field(
        default="extract_data",
        description="Must be 'extract_data' to extract financial data from parsed documents."
    )
    security_id: str = Field(
        description="Unique identifier for the target company/security. Used to store results in <base_path>/<security_id>/extracted/"
    )
    parsed_documents_json: str = Field(
        description="JSON string containing the parsed document contents from 'parse_documents'. Must include text content and page metadata."
    )
    vector_store_root_path_override: Optional[str] = Field(
        default=None,
        description="Optional: Override the default root path for vector stores used in semantic search during extraction."
    )

class InnerFinancialQualityCheckAction(BaseModel):
    """Input schema for the run_quality_check action.
    
    This action validates extracted financial data against the schema and calculates completeness scores.
    Requires output from the extract_data action as input.
    """
    action: Literal["run_quality_check"] = Field(
        default="run_quality_check",
        description="Must be 'run_quality_check' to validate extracted data and generate quality report."
    )
    security_id: str = Field(
        description="Unique identifier for the target company/security. Used to store report in <base_path>/<security_id>/quality_check/"
    )
    extraction_result_json: str = Field(
        description="JSON string containing the extracted financial data from 'extract_data'. Must include line items with values and periods."
    )

class FinancialPhaseAActionInput(RootModel):
    root: Union[InnerFinancialParseDocsAction, InnerFinancialExtractDataAction, InnerFinancialQualityCheckAction] = Field(..., discriminator='action')

# --- Main Tool Definition --- #

class FinancialPhaseAToolError(ToolException):
    """Custom exception for the FinancialPhaseATool."""
    pass

class FinancialPhaseATool(BaseTool, BaseModel):
    # Custom input parsing to handle RootModel with discriminated union
    def _parse_input(
        self,
        tool_input: Union[str, Dict],
        tool_call_id: Optional[str] = None, # Langchain 0.1.x signature
    ) -> Dict: # Returns a dictionary representation of the validated RootModel
        if not isinstance(tool_input, dict):
            raise ToolException(
                f"Invalid input type {type(tool_input)}. Expected a dictionary for FinancialPhaseATool."
            )
        try:
            # Validate the input using the tool's args_schema (FinancialPhaseAActionInput).
            validated_model = self.args_schema.model_validate(tool_input)
            # Return the dictionary representation of the inner model from the RootModel.
            # This ensures _arun receives {'action': '...', 'security_id': '...'} as kwargs.
            if hasattr(validated_model, 'root') and isinstance(validated_model.root, BaseModel):
                return validated_model.root.model_dump()
            else:
                # This should not happen if FinancialPhaseAActionInput is correctly defined as a RootModel
                # containing one of the Inner...Action models in its 'root'.
                raise ToolException(
                    f"Validated model of type {type(validated_model)} does not have a 'root' attribute "
                    f"that is a Pydantic BaseModel, or the RootModel structure is unexpected."
                )
        except Exception as e: # Catches Pydantic ValidationError and other errors
            raise ToolException(f"Tool input validation error: {e}. Input was: {tool_input}")

    name: str = Field(
        default="financial_phase_a",
        description="Tool for financial document parsing, data extraction, and quality checking"
    )
    
    description: str = Field(
        default=(
            "Tool for processing financial documents through three sequential phases. Each action must be called with a root object containing an 'action' field that determines the action type and its specific input schema.\n\n"
            "Directory Structure:\n"
            "<base_path>/                # /otcmn_tool_test_output/current\n"
            "    <board_folder>/         # primary_board_[A,B,C] or secondary_board_[A,B,C]\n"
            "        <security_id>/      # e.g. MN0LNDB68390\n"
            "            documents/      # Input PDFs\n"
            "            parsed_data/    # Phase 1 output\n"
            "            extracted_data/ # Phase 2 output\n"
            "            quality_check/  # Phase 3 output\n\n"
            "Actions and Examples:\n\n"
            "1. parse_documents - Convert PDFs to structured text:\n"
            "   Input Example:\n"
            "   {\n"
            "     \"action\": \"parse_documents\",\n"
            "     \"security_id\": \"MN0LNDB68390\",\n"
            "     \"primary_document_name\": \"annual_report_2023.pdf\" // optional\n"
            "   }\n"
            "   Output: Saves parsed_financial_documents.json to <board>/<security_id>/parsed_data/\n\n"
            "2. extract_data - Extract financial line items:\n"
            "   Input Example:\n"
            "   {\n"
            "     \"action\": \"extract_data\",\n"
            "     \"security_id\": \"MN0LNDB68390\",\n"
            "     \"parsed_documents_json\": \"[{\\\"text\\\": \\\"...\\\"}]\"\n"
            "   }\n"
            "   Output: Saves extraction_result.json to <board>/<security_id>/extracted_data/\n\n"
            "3. run_quality_check - Validate extracted data:\n"
            "   Input Example:\n"
            "   {\n"
            "     \"action\": \"run_quality_check\",\n"
            "     \"security_id\": \"MN0LNDB68390\",\n"
            "     \"extraction_result_json\": \"{\\\"line_items\\\": [...]}\"\n"
            "   }\n"
            "   Output: Saves quality_report.json to <board>/<security_id>/quality_check/\n"
            "   - Report includes completeness score and validation errors\n\n"
            "Each action requires a security_id to locate its input/output files. "
            "Tool will search all board folders under <base_path> for the security_id."
        )
    )
    args_schema: Type[BaseModel] = FinancialPhaseAActionInput
    return_direct: bool = False

    # Core utilities and tools, with default factories or values
    financial_utils: FinancialModelingUtils = Field(default_factory=FinancialModelingUtils)
    mcp_document_parser_tool: MCPDocumentParserTool = Field(default_factory=MCPDocumentParserTool)

    # Configuration paths, with default values from global constants
    schema_definition_path: Path = Field(default=FINANCIAL_MODEL_SCHEMA_PATH)
    prompts_path: Path = Field(default=FINANCIAL_MODEL_PROMPTS_PATH)
    securities_data_base_path: Path = Field(default=DEFAULT_SECURITIES_DOCS_BASE_PATH)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data: Any):
        super().__init__(**data)

    async def close(self):
        """Closes any resources held by the tool, like the MCPDocumentParserTool."""
        logger.info("Closing FinancialPhaseATool and its resources...")
        if hasattr(self, 'mcp_document_parser_tool') and self.mcp_document_parser_tool:
            if hasattr(self.mcp_document_parser_tool, 'close') and callable(self.mcp_document_parser_tool.close):
                try:
                    if inspect.iscoroutinefunction(self.mcp_document_parser_tool.close):
                        logger.debug("Awaiting mcp_document_parser_tool.close()...")
                        await self.mcp_document_parser_tool.close()
                    else:
                        logger.debug("Calling mcp_document_parser_tool.close()...")
                        self.mcp_document_parser_tool.close() # type: ignore
                    logger.info("MCPDocumentParserTool closed successfully.")
                except Exception as e:
                    logger.error(f"Error closing MCPDocumentParserTool: {e}", exc_info=True)
            else:
                logger.debug("mcp_document_parser_tool does not have a callable 'close' method.")
        else:
            logger.debug("mcp_document_parser_tool not found or not initialized.")

    async def _handle_parse_documents(self, action_input: InnerFinancialParseDocsAction) -> str:
        logger.info(f"Starting document parsing for security: {action_input.security_id}")
        current_docs_base_str = action_input.securities_docs_base_path_override if action_input.securities_docs_base_path_override else str(getattr(AlpyConfig, "DEFAULT_SECURITIES_DOCS_BASE_PATH", DEFAULT_SECURITIES_DOCS_BASE_PATH))
        current_docs_base = Path(current_docs_base_str)

        # Search all board folders for the security_id
        board_folders = [f for f in current_docs_base.iterdir() if f.is_dir() and (f.name.startswith("primary_board_") or f.name.startswith("secondary_board_"))]
        
        docs_root_path = None
        for board in board_folders:
            potential_path = board / action_input.security_id / "documents"
            if potential_path.exists() and potential_path.is_dir():
                docs_root_path = potential_path
                logger.info(f"Found security {action_input.security_id} in board folder {board.name}")
                break

        if not docs_root_path:
            msg = f"Documents directory not found for security {action_input.security_id} in any board folder under {current_docs_base}"
            logger.error(msg)
            return json.dumps({"error": msg, "parsed_documents": []})

        parsed_document_objects: List[Dict] = []
        doc_files = [f for f in docs_root_path.iterdir() if f.is_file() and f.suffix.lower() == '.pdf']

        if not doc_files:
            msg = f"No PDF documents found in {docs_root_path}"
            logger.warning(msg)
            return json.dumps({"status": "warning", "message": msg, "parsed_documents": []})

        for doc_path in doc_files:
            if action_input.primary_document_name and doc_path.name != action_input.primary_document_name:
                logger.debug(f"Skipping non-primary document: {doc_path.name}")
                continue
            
            logger.info(f"Parsing document: {doc_path}")
            parser_input_data = MCPDocumentParserToolInput(
                document_path=str(doc_path),
                ocr_mode=action_input.ocr_mode,
                ocr_lang=action_input.ocr_lang,
                max_pages_to_parse=action_input.max_pages_to_parse_per_doc,
            )
            try:
                parser_result_json = await self.mcp_document_parser_tool._arun(**parser_input_data.model_dump())
                parser_result = json.loads(parser_result_json)

                if parser_result.get("error"):
                    logger.error(f"Error parsing {doc_path.name}: {parser_result['error']}")
                    continue
                
                full_text_content_parts = []
                page_contents_for_output = [] # Store structured page content if needed later

                for i, page_data in enumerate(parser_result.get("pages_content", [])):
                    current_page_number = page_data.get("page_number")
                    page_text = page_data.get("page_full_ocr_text") or page_data.get("text_pymupdf", "")
                    
                    # Ensure page_text is a string, even if it's None initially from .get()
                    page_text = page_text if page_text is not None else ""

                    if not page_text.strip(): # If primary methods yield no text
                        logger.debug(f"Page {current_page_number}: Primary text extraction (full OCR/PyMuPDF) empty. Checking image OCR results.")
                        image_texts_for_page = []
                        for img_ocr_res in parser_result.get("all_images_ocr_results", []):
                            if img_ocr_res.get("page_number") == current_page_number and img_ocr_res.get("ocr_text"):
                                image_texts_for_page.append(img_ocr_res.get("ocr_text"))
                        if image_texts_for_page:
                            page_text = "\n".join(image_texts_for_page)
                            logger.info(f"Page {current_page_number}: Used concatenated text from {len(image_texts_for_page)} images.")
                        else:
                            logger.warning(f"Page {current_page_number}: No text from primary extraction or image OCR.")

                    full_text_content_parts.append(page_text)
                    
                    # Store individual page content for potential future use or detailed output
                    page_contents_for_output.append({
                        "page_number": page_data.get("page_number"),
                        "text": page_text, # Text used for aggregation
                        "text_pymupdf": page_data.get("text_pymupdf"),
                        "page_full_ocr_text": page_data.get("page_full_ocr_text"),
                        "page_full_ocr_error": page_data.get("page_full_ocr_error")
                    })

                full_text_content = "\n\n".join(filter(None, full_text_content_parts)) # Join pages with double newline

                parsed_doc = {
                    "document_name": doc_path.name,
                    "full_text_content": full_text_content,
                    "page_contents": page_contents_for_output, # Now contains more structured page data
                    "metadata": {
                        "total_pages": parser_result.get("total_pages", 0),
                        "processed_pages_count": parser_result.get("processed_pages_count", 0),
                        "file_name_from_parser": parser_result.get("file_name"), # Original filename from parser
                        "parser_status": parser_result.get("status"),
                        "parser_message": parser_result.get("message"),
                        "tesseract_available": parser_result.get("tesseract_available"),
                        "tesseract_initial_check_message": parser_result.get("tesseract_initial_check_message")
                    }
                }
                parsed_document_objects.append(parsed_doc)
                logger.info(f"Successfully parsed: {doc_path.name}")
            except Exception as e:
                logger.error(f"Failed to parse document {doc_path.name}: {e}", exc_info=True)
        
        output_list = [doc for doc in parsed_document_objects]

        # Use the same board folder we found earlier
        board_root = docs_root_path.parent.parent  # Go up from /documents to get board folder
        output_data_dir = board_root / action_input.security_id / "parsed_data"
        try:
            output_data_dir.mkdir(parents=True, exist_ok=True)
            output_file_path = output_data_dir / "parsed_financial_documents.json"

            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(output_list, f, ensure_ascii=False, indent=4)
            logger.info(f"Successfully saved parsed documents to: {output_file_path}")
            
            return json.dumps({
                "status": "success",
                "message": f"Documents parsed and results saved to {output_file_path}",
                "output_file_path": str(output_file_path),
                "num_documents_parsed": len(output_list),
                "parsed_documents_summary": [
                    {
                        "document_name": doc.get("document_name"), 
                        "pages_processed": doc.get("metadata", {}).get("processed_pages_count", 0)
                    }
                    for doc in output_list
                ]
            })
        except Exception as e:
            logger.error(f"Failed to save parsed documents for security {action_input.security_id}: {e}", exc_info=True)
            # Still return the parsed data in memory if saving failed, along with an error status
            return json.dumps({
                "status": "error",
                "message": f"Documents parsed but failed to save to file: {e}",
                "parsed_data_in_memory": output_list # Allow caller to potentially use in-memory data
            })

    async def _handle_extract_data(self, action_input: InnerFinancialExtractDataAction) -> str:
        logger.info(f"Starting financial data extraction for security: {action_input.security_id}")
        try:
            list_of_parsed_document_dicts = json.loads(action_input.parsed_documents_json)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in parsed_documents_json: {e}")
            return json.dumps({"error": "Invalid JSON format for parsed documents."})
        except Exception as e:
            logger.error(f"Error converting parsed document JSON to objects: {e}", exc_info=True)
            return json.dumps({"error": f"Error processing parsed_documents_json: {e}"})

        if not list_of_parsed_document_dicts:
            logger.warning(f"No parsed documents for extraction: {action_input.security_id}")
            return json.dumps({"error": "No parsed documents provided.", "extracted_data": None, "quality_report": None})

        orchestrator = PhaseAOrchestrator(
            security_id=action_input.security_id,
            prompts_yaml_path=self.prompts_path, 
            financial_schema_path=self.schema_definition_path 
        )

        try:
            # The orchestrator's run_phase_a_extraction method will internally handle
            # adding documents to its RAG manager using the provided parsed_documents.
            extraction_result = await orchestrator.run_phase_a_extraction(
                parsed_documents=list_of_parsed_document_dicts, # Pass the direct output from MCPDocumentParserTool
                document_names=[doc_content.get('metadata', {}).get('document_name', f'doc_{i}') for i, doc_content in enumerate(list_of_parsed_document_dicts)],
                primary_document_name=None # Ensure this is correctly passed if available
            )
            
            # Find the correct board folder for this security
            current_docs_base = Path(str(getattr(AlpyConfig, "DEFAULT_SECURITIES_DOCS_BASE_PATH", DEFAULT_SECURITIES_DOCS_BASE_PATH)))
            board_folders = [f for f in current_docs_base.iterdir() if f.is_dir() and (f.name.startswith("primary_board_") or f.name.startswith("secondary_board_"))]
            
            board_root = None
            for board in board_folders:
                if (board / action_input.security_id).exists():
                    board_root = board
                    logger.info(f"Found security {action_input.security_id} in board folder {board.name}")
                    break

            if not board_root:
                msg = f"Security folder not found for {action_input.security_id} in any board folder under {current_docs_base}"
                logger.error(msg)
                return json.dumps({"error": msg})

            # Save extraction result to security folder
            output_data_dir = board_root / action_input.security_id / "extracted_data"
            try:
                output_data_dir.mkdir(parents=True, exist_ok=True)
                output_file_path = output_data_dir / "extraction_result.json"
                
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(extraction_result, f, ensure_ascii=False, indent=4)
                logger.info(f"Successfully saved extraction result to: {output_file_path}")
                
                return json.dumps({
                    "status": "success",
                    "message": f"Data extracted and results saved to {output_file_path}",
                    "output_file_path": str(output_file_path),
                    "extraction_result": extraction_result
                })
            except Exception as save_error:
                logger.error(f"Failed to save extraction result for security {action_input.security_id}: {save_error}", exc_info=True)
                return json.dumps({
                    "status": "error",
                    "message": f"Data extracted but failed to save to file: {save_error}",
                    "extraction_result": extraction_result  # Return the data even if save failed
                })
        except Exception as e:
            logger.error(f"Error during Phase A extraction for {action_input.security_id}: {e}", exc_info=True)
            return json.dumps({"error": f"Extraction process failed: {e}"})



    # --- Placeholder methods to satisfy BaseTool requirements ---
    
    def _run(self, *args: Any, **kwargs: Any) -> str:
        # This tool is designed to be used via its specific actions (parse_documents, extract_data, run_quality_check).
        # Direct _run is not the primary interface for this multi-action tool.
        # Consider calling a default action or raising NotImplementedError if direct _run is not supported.
        # For now, returning a message indicating how to use it.
        logger.warning(
        "FinancialPhaseATool._run() called directly. This tool is intended to be used via specific actions. "
        "See call_action() or the individual _handle_... methods."
        )
        # You could choose to raise NotImplementedError or provide a default behavior.
        # Example: return self.call_action_sync("parse_documents", {"security_id": "DEFAULT_ID_IF_NEEDED"})
        raise NotImplementedError(
        "FinancialPhaseATool is a multi-action tool. Use call_action() or its specific async handlers."
        )

    async def _handle_run_quality_check(self, action_input: InnerFinancialQualityCheckAction) -> str:
        logger.info(f"Running quality check for security: {action_input.security_id}")
        try:
            extraction_result_dict = json.loads(action_input.extraction_result_json)
            
            # Create quality checker and generate report
            quality_checker = QualityChecker(schema_definition_path=self.schema_definition_path)
            quality_report = quality_checker.run_all_checks(extraction_result_dict)
            
            # Find the correct board folder for this security
            current_docs_base = Path(str(getattr(AlpyConfig, "DEFAULT_SECURITIES_DOCS_BASE_PATH", DEFAULT_SECURITIES_DOCS_BASE_PATH)))
            board_folders = [f for f in current_docs_base.iterdir() if f.is_dir() and (f.name.startswith("primary_board_") or f.name.startswith("secondary_board_"))]
            
            board_root = None
            for board in board_folders:
                if (board / action_input.security_id).exists():
                    board_root = board
                    logger.info(f"Found security {action_input.security_id} in board folder {board.name}")
                    break

            if not board_root:
                msg = f"Security folder not found for {action_input.security_id} in any board folder under {current_docs_base}"
                logger.error(msg)
                return json.dumps({"error": msg})

            # Save quality report to security folder
            output_data_dir = board_root / action_input.security_id / "quality_check"
            logger.info(f"Determined output data directory for quality check: {output_data_dir}")
            try:
                output_data_dir.mkdir(parents=True, exist_ok=True)
                output_file_path = output_data_dir / "quality_report.json"
                logger.info(f"Attempting to save quality report to: {output_file_path}")
                
                quality_report_dict = quality_report.model_dump()
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(quality_report_dict, f, ensure_ascii=False, indent=4)
                logger.info(f"Successfully saved quality report to: {output_file_path}")
                
                return json.dumps({
                    "status": "success",
                    "message": f"Quality report generated and saved to {output_file_path}",
                    "output_file_path": str(output_file_path),
                    "quality_report": quality_report.model_dump()
                })
            except Exception as save_error:
                logger.error(f"Failed to save quality report for security {action_input.security_id}: {save_error}", exc_info=True)
                return json.dumps({
                    "status": "error",
                    "message": f"Quality report generated but failed to save to file: {save_error}",
                    "quality_report": quality_report.model_dump()  # Return the report even if save failed
                })
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in extraction_result_json: {e}")
            return json.dumps({"error": "Invalid JSON format for extraction result."})
        except Exception as e:
            logger.error(f"Error during quality check: {e}", exc_info=True)
            return json.dumps({"error": f"Failed to run quality check: {e}"})

    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any # These kwargs are the fields of one of the Inner...Action models
    ) -> str:
        action_type_str = kwargs.get("action")
        logger.debug(f"FinancialPhaseATool._arun called with action_type '{action_type_str}' and kwargs: {kwargs}")

        try:
            if action_type_str == FinancialPhaseAActionType.PARSE_DOCUMENTS.value:
                specific_action_input = InnerFinancialParseDocsAction.model_validate(kwargs)
                return await self._handle_parse_documents(specific_action_input)
            elif action_type_str == FinancialPhaseAActionType.EXTRACT_DATA.value:
                specific_action_input = InnerFinancialExtractDataAction.model_validate(kwargs)
                return await self._handle_extract_data(specific_action_input)
            elif action_type_str == FinancialPhaseAActionType.RUN_QUALITY_CHECK.value:
                specific_action_input = InnerFinancialQualityCheckAction.model_validate(kwargs)
                return await self._handle_run_quality_check(specific_action_input)
            else:
                valid_actions = [e.value for e in FinancialPhaseAActionType]
                error_msg = f"Invalid or missing 'action' field in input. Must be one of {valid_actions}. Received: '{action_type_str}'"
                logger.error(error_msg + f" with kwargs: {kwargs}")
                return json.dumps({
                    "error": error_msg,
                    "details": f"The 'action' parameter must be one of {valid_actions}."
                })
        except ValidationError as ve:
            error_msg = f"Input validation error for action '{action_type_str}': {ve}"
            logger.error(error_msg + f" for kwargs: {kwargs}", exc_info=True)
            return json.dumps({
                "error": "Input validation error",
                "action_type": action_type_str,
                "details": str(ve)
            })
        except Exception as e:
            error_msg = f"Unexpected error during execution of action '{action_type_str}': {e}"
            logger.error(error_msg + f" for kwargs: {kwargs}", exc_info=True)
            return json.dumps({
                "error": "Unexpected tool error",
                "action_type": action_type_str,
                "details": str(e)
            })
# --- Standalone Test Functions --- #

async def test_parse_documents_action(tool: FinancialPhaseATool):
    print("--- Testing: Parse Documents Action ---", flush=True)
    try:
        # tool instance is now passed as an argument
        test_security_id = "MN0LNDB68390" 
        parse_action_input = {
            "action": "parse_documents",
            "security_id": test_security_id,
            "primary_document_name": None, 
            "ocr_mode": "auto",
            "ocr_lang": "mon",
            "max_pages_to_parse_per_doc": None
        }
        print(f"Invoking tool._arun with input: {json.dumps(parse_action_input, indent=2)}", flush=True)
        # Call _arun by unpacking the action input dictionary as keyword arguments
        parsed_docs_json_str = await tool._arun(**parse_action_input)
        print("tool._arun call completed.", flush=True)
        
        try:
            parsed_data_result = json.loads(parsed_docs_json_str)
            print(f"Parse Documents Output (Full JSON if possible, or sample):\n{json.dumps(parsed_data_result, indent=2)}", flush=True)
            
            if isinstance(parsed_data_result, dict) and parsed_data_result.get("error"):
                print(f"Parse action failed with error in result: {parsed_data_result['error']}", flush=True)
            elif isinstance(parsed_data_result, dict) and parsed_data_result.get("parsed_documents_summary"):
                summary = parsed_data_result.get("parsed_documents_summary", [])
                if not summary and parsed_data_result.get("num_documents_parsed", 0) == 0 :
                     print("Parse action returned no documents or an empty result based on summary.", flush=True)
                else:
                    print(f"Parsed {len(summary)} documents successfully according to summary.", flush=True)

            print("Test 'parse_documents' completed its main logic.", flush=True)

        except json.JSONDecodeError as je:
            print(f"Error decoding JSON from parse_documents: {je}", flush=True)
            print(f"Raw output was: {parsed_docs_json_str}", flush=True)
            logger.error("Test 'parse_documents' failed due to JSON decode error", exc_info=True)

    except Exception as e:
        print(f"Error during 'parse_documents' action test: {e}", flush=True)
        logger.error("Test 'parse_documents' failed with an exception", exc_info=True)
    # Note: Closing the tool, especially mcp_document_parser_tool, should be handled by the main test runner
    # to avoid closing it prematurely if other tests need it.

async def test_extract_data_action(tool: FinancialPhaseATool):
    print("\n--- Testing: Extract Data Action ---")
    test_security_id = "MN0LNDB68390"

    # Path to the expected output from the parse_documents action
    # This assumes parse_documents was run and successfully created this file in one of the board folders.
    # We need to find which board folder it used.
    current_docs_base_dir = tool.securities_data_base_path # Use tool's configured path
    
    parsed_documents_path = None
    possible_board_folders = [d for d in current_docs_base_dir.iterdir() if d.is_dir() and (d.name.startswith("primary_board_") or d.name.startswith("secondary_board_"))]
    
    for board_folder in possible_board_folders:
        potential_path = board_folder / test_security_id / "parsed_data" / "parsed_financial_documents.json"
        if potential_path.exists():
            parsed_documents_path = potential_path
            logger.info(f"Found parsed documents for extraction at: {parsed_documents_path}")
            break
            
    if not parsed_documents_path:
        error_message = f"Required parsed documents file not found for test_extract_data_action in any board folder for {test_security_id} under {current_docs_base_dir}. Run 'test_parse_documents_action' first or ensure the file exists."
        logger.error(error_message)
        print(f"Error: {error_message}")
        return

    parsed_documents_json_str = ""
    try:
        with open(parsed_documents_path, 'r', encoding='utf-8') as f:
            parsed_documents_json_str = f.read()
        # Validate JSON structure after reading
        parsed_docs_content_for_extraction = json.loads(parsed_documents_json_str)
        # The actual content for "parsed_documents_json" in the action input should be a JSON string
        # representing the list of parsed document objects.
        # So, we re-dump the loaded list back into a string.
        parsed_documents_input_str = json.dumps(parsed_docs_content_for_extraction)

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {parsed_documents_path}: {e}")
        print(f"Error: Invalid JSON in {parsed_documents_path}: {e}")
        return
    except Exception as e:
        logger.error(f"Error reading or parsing {parsed_documents_path}: {e}", exc_info=True)
        print(f"Error reading or parsing {parsed_documents_path}: {e}")
        return

    extract_action_input = {
        "action": "extract_data",
        "security_id": test_security_id, 
        "parsed_documents_json": parsed_documents_input_str # This must be a string
    }
    try:
        extraction_result_json_str = await tool._arun(**extract_action_input)
        extraction_result = json.loads(extraction_result_json_str)
        print(f"Extract Data Output (JSON sample):\n{json.dumps(extraction_result, indent=2)[:500]}...")
        if isinstance(extraction_result, dict) and extraction_result.get("error"):
            print(f"Extract action failed: {extraction_result['error']}")
    except Exception as e:
        print(f"Error during 'extract_data' action test: {e}")
        logger.error("Test 'extract_data' failed", exc_info=True)

async def test_run_quality_check_action(tool: FinancialPhaseATool):
    print("\n--- Testing: Run Quality Check Action ---")
    test_security_id = "MN0LNDB68390"

    # Path to the expected output from the extract_data action
    current_docs_base_dir = tool.securities_data_base_path

    extraction_result_path = None
    possible_board_folders = [d for d in current_docs_base_dir.iterdir() if d.is_dir() and (d.name.startswith("primary_board_") or d.name.startswith("secondary_board_"))]

    for board_folder in possible_board_folders:
        potential_path = board_folder / test_security_id / "extracted_data" / "extraction_result.json"
        if potential_path.exists():
            extraction_result_path = potential_path
            logger.info(f"Found extraction results for QC at: {extraction_result_path}")
            break
    
    if not extraction_result_path:
        error_message = f"Required extraction result file not found for test_run_quality_check_action in any board folder for {test_security_id} under {current_docs_base_dir}. Run 'test_extract_data_action' first or ensure the file exists."
        logger.error(error_message)
        print(f"Error: {error_message}")
        return

    extraction_result_json_str_for_qc = ""
    try:
        with open(extraction_result_path, 'r', encoding='utf-8') as f:
            extraction_result_json_str_for_qc = f.read()
        # Validate JSON
        json.loads(extraction_result_json_str_for_qc) # Ensures it's valid JSON string
        logger.info(f"Successfully loaded extraction result from {extraction_result_path}")
    except Exception as e:
        logger.error(f"Error reading extraction result file {extraction_result_path}: {e}")
        print(f"Error reading extraction result file {extraction_result_path}: {e}")
        return

    quality_check_action_input = {
        "action": "run_quality_check",
        "security_id": test_security_id,
        "extraction_result_json": extraction_result_json_str_for_qc # This must be a string
    }
    try:
        quality_report_json_str = await tool._arun(**quality_check_action_input)
        quality_report = json.loads(quality_report_json_str)
        print(f"Quality Check Output (JSON sample):\n{json.dumps(quality_report, indent=2)[:500]}...")
        if isinstance(quality_report, dict) and quality_report.get("error"):
            print(f"QC action failed: {quality_report['error']}")
    except Exception as e:
        print(f"Error during 'run_quality_check' action test: {e}")
        logger.error("Test 'run_quality_check' failed", exc_info=True)

if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Run specific tests for FinancialPhaseATool actions.")
    parser.add_argument("--test-parse", action="store_true", help="Run the 'parse_documents' action test.")
    parser.add_argument("--test-extract", action="store_true", help="Run the 'extract_data' action test.")
    parser.add_argument("--test-qc", action="store_true", help="Run the 'run_quality_check' action test.")
    parser.add_argument("--test-all", action="store_true", help="Run all action tests sequentially.")

    args = parser.parse_args()

    # Configure logging for test runs
    logging.basicConfig(level=AlpyConfig.LOG_LEVEL, format=AlpyConfig.LOG_FORMAT) # Use main app config for direct runs
    logger.setLevel(logging.INFO) 
    logging.getLogger("mcp.client.session").setLevel(logging.WARNING) 
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


    async def main_test_runner():
        run_any_test = False
        # Create a single tool instance to be used by all tests
        # This allows resources like MCPDocumentParserTool to be initialized once.
        tool_instance = FinancialPhaseATool()
        try:
            if args.test_all or args.test_parse:
                await test_parse_documents_action(tool_instance)
                run_any_test = True
            if args.test_all or args.test_extract:
                await test_extract_data_action(tool_instance)
                run_any_test = True
            if args.test_all or args.test_qc:
                await test_run_quality_check_action(tool_instance)
                run_any_test = True
            
            if not run_any_test:
                print("No test specified. Use --test-parse, --test-extract, --test-qc, or --test-all.")
        finally:
            # Ensure the tool (and its sub-tools like MCPDocumentParserTool) are closed after all tests
            if hasattr(tool_instance, 'close') and callable(tool_instance.close):
                print("Closing FinancialPhaseATool instance...")
                await tool_instance.close()
                print("FinancialPhaseATool instance closed.")

    asyncio.run(main_test_runner())

async def test_extract_data_action(tool: FinancialPhaseATool):
    print("\n--- Testing: Extract Data Action ---")
    # tool = FinancialPhaseATool() # Tool instance is now passed as an argument
    # Use the same security_id as test_parse_documents_action to load its output
    test_security_id = "MN0LNDB68390"

    # Construct the path to the parsed_financial_documents.json file
    # Use the same base path as the parse_documents action uses for its output.
    # DEFAULT_SECURITIES_DOCS_BASE_PATH is a global Path object defined in this file.
    current_docs_base_dir = DEFAULT_SECURITIES_DOCS_BASE_PATH
    parsed_documents_path = DEFAULT_SECURITIES_DOCS_BASE_PATH / "secondary_board_B" / test_security_id / "parsed_data" / "parsed_financial_documents.json"

    parsed_documents_json_str = ""
    if not parsed_documents_path.exists():
        error_message = f"Required parsed documents file not found for test_extract_data_action: {parsed_documents_path}. Run 'test_parse_documents_action' first or ensure the file exists."
        logger.error(error_message)
        raise FileNotFoundError(error_message)
    else:
        logger.info(f"Loading parsed documents from: {parsed_documents_path}")
        try:
            with open(parsed_documents_path, 'r', encoding='utf-8') as f:
                parsed_documents_json_str = f.read()
            # Validate JSON structure after reading, before using it in extract_action_input
            json.loads(parsed_documents_json_str) 
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {parsed_documents_path}: {e}")
            print(f"Error: Invalid JSON in {parsed_documents_path}: {e}")
            return # Or raise, to make the test fail explicitly
        except Exception as e: # Catch other potential file reading errors
            logger.error(f"Error reading or parsing {parsed_documents_path}: {e}", exc_info=True)
            print(f"Error reading or parsing {parsed_documents_path}: {e}")
            return # Or raise

    extract_action_input = {
        "action": "extract_data",
        "security_id": test_security_id, 
        "parsed_documents_json": parsed_documents_json_str
    }
    try:
        extraction_result_json = await tool._arun(**extract_action_input)
        print(f"Extract Data Output (JSON sample):\n{json.dumps(json.loads(extraction_result_json), indent=2)[:500]}...")
        extracted_data = json.loads(extraction_result_json)
        if isinstance(extracted_data, dict) and extracted_data.get("error"):
            print(f"Extract action failed: {extracted_data['error']}")
    except Exception as e:
        print(f"Error during 'extract_data' action test: {e}")
        logger.error("Test 'extract_data' failed", exc_info=True)

async def test_run_quality_check_action(tool: FinancialPhaseATool):
    print("\n--- Testing: Run Quality Check Action ---")
    # tool = FinancialPhaseATool() # Tool instance is now passed as an argument
    test_security_id = "MN0LNDB68390"

    # Read the actual extraction result file
    extraction_result_path = DEFAULT_SECURITIES_DOCS_BASE_PATH / "secondary_board_B" / test_security_id / "extracted_data" / "extraction_result.json"
    
    try:
        with open(extraction_result_path, 'r', encoding='utf-8') as f:
            extraction_result_json = f.read()
        # Validate JSON
        json.loads(extraction_result_json)
        logger.info(f"Successfully loaded extraction result from {extraction_result_path}")
    except Exception as e:
        logger.error(f"Error reading extraction result file: {e}")
        raise

    quality_check_action_input = {
        "action": "run_quality_check",
        "security_id": test_security_id,
        "extraction_result_json": extraction_result_json
    }
    try:
        quality_report_json = await tool._arun(**quality_check_action_input)
        print(f"Quality Check Output (JSON sample):\n{json.dumps(json.loads(quality_report_json), indent=2)[:500]}...")
        qc_data = json.loads(quality_report_json)
        if isinstance(qc_data, dict) and qc_data.get("error"):
            print(f"QC action failed: {qc_data['error']}")
    except Exception as e:
        print(f"Error during 'run_quality_check' action test: {e}")
        logger.error("Test 'run_quality_check' failed", exc_info=True)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run specific tests for FinancialPhaseATool actions.")
    parser.add_argument("--test-parse", action="store_true", help="Run the 'parse_documents' action test.")
    parser.add_argument("--test-extract", action="store_true", help="Run the 'extract_data' action test.")
    parser.add_argument("--test-qc", action="store_true", help="Run the 'run_quality_check' action test.")
    parser.add_argument("--test-all", action="store_true", help="Run all action tests sequentially.")

    args = parser.parse_args()

    # Configure logging for test runs
    logging.basicConfig(level=AlpyConfig.LOG_LEVEL, format=AlpyConfig.LOG_FORMAT)
    logger.setLevel(logging.INFO) 
    logging.getLogger("mcp.client.session").setLevel(logging.WARNING) 

    async def main():
        run_any_test = False
        tool = FinancialPhaseATool()
        try:
            if args.test_parse:
                await test_parse_documents_action(tool)
                run_any_test = True
            if args.test_extract:
                await test_extract_data_action(tool)
                run_any_test = True
            if args.test_qc:
                await test_run_quality_check_action(tool)
                run_any_test = True
            if args.test_all:
                await test_parse_documents_action(tool)
                await test_extract_data_action(tool)
                await test_run_quality_check_action(tool)
                run_any_test = True
            
            if not run_any_test:
                print("No test specified. Use --test-parse, --test-extract, --test-qc, or --test-all.")
        finally:
            logger.info("Closing FinancialPhaseATool after tests.")
            await tool.close()
            parser.print_help()

    asyncio.run(main())
