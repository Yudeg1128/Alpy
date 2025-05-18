import asyncio
import json
import logging
import os
import shutil # For shutil.which for server paths if needed, not directly used by tool logic here
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type

from langchain_core.tools import BaseTool, ToolException
from langchain_core.callbacks.manager import AsyncCallbackManagerForToolRun # Optional for LangChain integration
from pydantic import BaseModel, Field, model_validator, PrivateAttr

# --- Local Module Imports ---
# Assuming these are in src/financial_modeling/ and src/tools/ relative to Alpy root
# and Alpy's src directory is in PYTHONPATH

# Placeholder for MCPDocumentParserTool - in a real setup, this would be the actual tool
# For this file, we'll define a mock version for standalone testing.
from ..tools.mcp_document_parser_tool import MCPDocumentParserTool # Example of relative import if structured

class MockMCPDocumentParserTool(BaseTool, BaseModel): # For testing this tool standalone
    name: str = "MockMCPDocumentParserTool"
    description: str = "Mock document parser"
    args_schema: Type[BaseModel] = None # type: ignore

    async def _arun(self, document_path: str, **kwargs: Any) -> str:
        logger.info(f"[MockMCPDocumentParserTool] Parsing: {document_path}")
        # Simulate finding some text and a table
        return json.dumps({
            "status": "success",
            "file_name": Path(document_path).name,
            "total_pages": 5,
            "processed_pages_count": 5,
            "pages_content": [
                {"page_number": 1, "text_pymupdf": f"Cover page for {Path(document_path).name}. Company: TestCo {kwargs.get('security_id', '')}", "page_full_ocr_text": None},
                {"page_number": 2, "text_pymupdf": "Financial Statements Section. Revenue FY2023: 1000 MNT. COGS FY2023: 600 MNT."},
                {"page_number": 3, "text_pymupdf": "Balance Sheet data... Cash: 50, AR: 150"},
                {"page_number": 4, "text_pymupdf": "More data... FY2022 Revenue: 800 MNT"},
                {"page_number": 5, "text_pymupdf": "Notes section."}
            ],
            "all_tables": [{
                "page_number": 2, "table_index_on_page": 0, 
                "rows": [["Header1", "Header2"], ["Data1A", "Data1B"], ["Data2A", "Data2B"]],
                "extraction_method": "mock_pdfplumber"
            }],
            "all_images_ocr_results": [],
            "tesseract_available": True,
            "tesseract_initial_check_message": "Mock Tesseract OK"
        })

from ..financial_modeling.phase_a_orchestrator import PhaseAOrchestrator, BaseLlmService, FinancialModelingUtils, GeminiLlmService
from ..financial_modeling.quality_checker import QualityChecker, QualityReport

# Module logger
logger = logging.getLogger(__name__)
# BasicConfig if no handlers on root logger (typical for library code)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s')


# --- Pydantic Schema for LangChain Tool Input ---
class FinancialPhaseAToolInput(BaseModel):
    security_id: str = Field(description="A unique identifier for the target company/security (e.g., ticker, CUSIP, internal ID). Used for context and output naming.")
    documents_root_path: str = Field(description="Absolute path to the directory containing financial documents specifically for the given security_id.")
    primary_document_name: Optional[str] = Field(default=None, description="Optional: Name of the primary document (e.g., 'prospectus.pdf', 'latest_10K.pdf') within documents_root_path to prioritize or focus on.")
    document_type_hint: Optional[str] = Field(default="Financial Report/Prospectus", description="A hint about the type of documents being processed (e.g., '10-K', 'Bond Prospectus').")
    
    # Parameters for MCPDocumentParserTool
    ocr_mode: Literal["auto", "force_images", "force_full_pages", "none"] = Field(default="auto", description="OCR mode for document parsing.")
    ocr_lang: str = Field(default="mon", description="Language for OCR (e.g., 'mon', 'eng').")
    max_pages_to_parse_per_doc: Optional[int] = Field(default=None, description="Optional: Limit the number of pages parsed per document (for testing/efficiency).")

# --- Main Tool Class ---
class FinancialPhaseATool(BaseTool, BaseModel):
    name: str = "GenerateFinancialModelPhaseA"
    description: str = (
        "Initiates Phase A (Historical Data Extraction) for financial model generation. "
        "It parses specified documents for a given security, uses an LLM to extract historical financial data "
        "based on a predefined schema, and performs quality checks on the extracted data. "
        "The tool requires a 'security_id' and a 'documents_root_path' containing relevant financial PDFs for that security. "
        "Returns a JSON string containing the populated Phase A model data and a quality report."
    )
    args_schema: Type[BaseModel] = FinancialPhaseAToolInput
    return_direct: bool = False # Output goes back to the LLM/agent for further processing

    # Dependencies to be injected or configured
    # For real use, these would be actual tool/service instances
    mcp_document_parser_tool: Any # Should be an instance of MCPDocumentParserTool
    llm_service: BaseLlmService
    financial_utils: FinancialModelingUtils

    # Configuration paths
    schema_definition_path: Path
    prompts_path: Path
    
    # Internal logger
    _logger_instance: logging.Logger = PrivateAttr(default=None)

    class Config:
        arbitrary_types_allowed = True # For Path and tool instances

    @model_validator(mode='after')
    def _tool_post_init_validator(self) -> 'FinancialPhaseATool':
        if self._logger_instance is None:
            self._logger_instance = logger.getChild(f"{self.name}.{id(self)}")
            # Avoid adding handlers if root logger already has them or if this logger already does
            if not self._logger_instance.handlers and not logging.getLogger().hasHandlers() and not logger.handlers:
                _handler = logging.StreamHandler(sys.stdout)
                _formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s')
                _handler.setFormatter(_formatter)
                self._logger_instance.addHandler(_handler)
                self._logger_instance.propagate = False # Don't propagate to root if we add our own handler
                self._logger_instance.setLevel(logging.INFO) # Default for this tool's logger

        # Resolve paths
        self.schema_definition_path = self.schema_definition_path.resolve()
        self.prompts_path = self.prompts_path.resolve()

        if not self.schema_definition_path.exists():
            self._logger_instance.error(f"CRITICAL: Schema definition file not found at {self.schema_definition_path}")
            # Consider raising an error here or handling it in _arun
        if not self.prompts_path.exists():
            self._logger_instance.error(f"CRITICAL: Prompts file not found at {self.prompts_path}")

        self._logger_instance.info(f"FinancialPhaseATool initialized. Schema: {self.schema_definition_path}, Prompts: {self.prompts_path}")
        return self

    def _run(
        self,
        security_id: str, # Add all expected args from args_schema
        documents_root_path: str,
        primary_document_name: Optional[str] = None,
        document_type_hint: Optional[str] = "Financial Report/Prospectus",
        ocr_mode: Literal["auto", "force_images", "force_full_pages", "none"] = "auto",
        ocr_lang: str = "mon",
        max_pages_to_parse_per_doc: Optional[int] = None,
        run_manager: Optional[Any] = None, # SyncCallbackManagerForToolRun
        **kwargs: Any
    ) -> str:
        raise NotImplementedError(
            f"{self.name} is an async-native tool. Use its `_arun` or `ainvoke` method."
        )

    async def _arun(
        self,
        security_id: str,
        documents_root_path: str,
        primary_document_name: Optional[str] = None,
        document_type_hint: Optional[str] = "Financial Report/Prospectus",
        ocr_mode: Literal["auto", "force_images", "force_full_pages", "none"] = "auto",
        ocr_lang: str = "mon",
        max_pages_to_parse_per_doc: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None, # For LangChain integration
        **kwargs: Any # Catches any other args passed by LangChain
    ) -> str:
        self._logger_instance.info(f"Starting Phase A for security_id='{security_id}', docs_path='{documents_root_path}'")

        output_payload = {
            "status": "PENDING",
            "message": "Processing started.",
            "security_id": security_id,
            "financial_model_phase_a": None,
            "quality_report": None,
            "parsed_documents_summary": []
        }

        try:
            # --- Step 1: Document Discovery and Parsing ---
            docs_path = Path(documents_root_path)
            if not docs_path.is_dir():
                msg = f"Documents root path is not a valid directory: {docs_path}"
                self._logger_instance.error(msg)
                output_payload["status"] = "ERROR"; output_payload["message"] = msg
                return json.dumps(output_payload, ensure_ascii=False, indent=2)

            pdf_files_to_process: List[Path] = []
            if primary_document_name:
                primary_doc_path = docs_path / primary_document_name
                if primary_doc_path.exists() and primary_doc_path.is_file() and primary_doc_path.suffix.lower() == ".pdf":
                    pdf_files_to_process.append(primary_doc_path)
                else:
                    self._logger_instance.warning(f"Specified primary_document_name '{primary_document_name}' not found or not a PDF. Scanning directory.")
            
            if not pdf_files_to_process: # If no primary or primary not found, scan
                pdf_files_to_process = sorted(list(docs_path.glob("*.pdf"))) # Simple glob, non-recursive

            if not pdf_files_to_process:
                msg = f"No PDF documents found in directory: {docs_path}"
                self._logger_instance.error(msg)
                output_payload["status"] = "ERROR"; output_payload["message"] = msg
                return json.dumps(output_payload, ensure_ascii=False, indent=2)

            self._logger_instance.info(f"Found {len(pdf_files_to_process)} PDF(s) to process: {[p.name for p in pdf_files_to_process]}")

            aggregated_parsed_content: Dict[str, Any] = { # Aggregate content from all docs
                "pages_content": [], "all_tables": [], "all_images_ocr_results": [],
                "processed_files_count": 0, "total_pages_processed_across_files": 0
            }
            
            # For simplicity, this version focuses on parsing one document (the first one found or the primary)
            # A multi-document strategy would require more complex aggregation and context management.
            # Let's process the first document in the list.
            
            doc_to_parse = pdf_files_to_process[0] # Or loop if handling multiple docs' content for one model
            self._logger_instance.info(f"Processing document: {doc_to_parse.name}")

            # Invoke MCPDocumentParserTool
            # Ensure the tool instance is correctly passed or accessed
            if not hasattr(self.mcp_document_parser_tool, "_arun"): # Basic check
                 msg = "MCPDocumentParserTool is not correctly configured or provided."
                 self._logger_instance.error(msg); output_payload["status"] = "ERROR"; output_payload["message"] = msg
                 return json.dumps(output_payload, ensure_ascii=False, indent=2)

            parser_input_args = {
                "document_path": str(doc_to_parse.resolve()),
                "ocr_mode": ocr_mode,
                "ocr_lang": ocr_lang,
                "max_pages_to_process_for_testing": max_pages_to_parse_per_doc, # Server tool uses this name
                "security_id": security_id # Pass for context if parser tool uses it
            }
            parsed_doc_json_str = await self.mcp_document_parser_tool._arun(**parser_input_args)
            parsed_doc_result = json.loads(parsed_doc_json_str)

            output_payload["parsed_documents_summary"].append({
                "file_name": doc_to_parse.name,
                "status": parsed_doc_result.get("status"),
                "message": parsed_doc_result.get("message"),
                "total_pages_in_doc": parsed_doc_result.get("total_pages", 0),
                "processed_pages_in_doc": parsed_doc_result.get("processed_pages_count", 0)
            })

            if parsed_doc_result.get("status") != "success":
                msg = f"Failed to parse document '{doc_to_parse.name}': {parsed_doc_result.get('message')}"
                self._logger_instance.error(msg)
                # Continue if other docs, but for single doc focus, this is an error for Phase A
                output_payload["status"] = "ERROR"; output_payload["message"] = msg
                return json.dumps(output_payload, ensure_ascii=False, indent=2)

            # Use content from this one parsed document
            # In a multi-doc strategy, you'd merge these lists carefully
            aggregated_parsed_content["pages_content"].extend(parsed_doc_result.get("pages_content", []))
            aggregated_parsed_content["all_tables"].extend(parsed_doc_result.get("all_tables", []))
            aggregated_parsed_content["all_images_ocr_results"].extend(parsed_doc_result.get("all_images_ocr_results", []))
            aggregated_parsed_content["processed_files_count"] = 1
            aggregated_parsed_content["total_pages_processed_across_files"] = parsed_doc_result.get("processed_pages_count",0)


            # --- Step 2: Orchestrate Phase A Extraction ---
            self._logger_instance.info("Instantiating PhaseAOrchestrator...")
            orchestrator = PhaseAOrchestrator(
                schema_definition_path=self.schema_definition_path,
                prompts_path=self.prompts_path,
                llm_service=self.llm_service,
                financial_utils=self.financial_utils,
                output_log_path=Path("logs") / f"orchestrator_{security_id}.log" # Security specific log
            )
            
            populated_model, extraction_log = await orchestrator.run_phase_a_extraction(
                parsed_doc_content=aggregated_parsed_content, # Pass the actual parsed data
                document_type_hint=document_type_hint or "Financial Report"
            )
            output_payload["financial_model_phase_a"] = populated_model
            # extraction_log could be part of quality_report details if needed

            # --- Step 3: Quality Check ---
            self._logger_instance.info("Instantiating QualityChecker...")
            quality_checker = QualityChecker(schema_definition_path=self.schema_definition_path)
            
            qc_report_obj: QualityReport = quality_checker.run_all_checks(populated_model)
            output_payload["quality_report"] = qc_report_obj.model_dump(exclude_none=True)

            # --- Step 4: Finalize ---
            if qc_report_obj.overall_status == "FAILED":
                output_payload["status"] = "COMPLETED_WITH_CRITICAL_QC_FAILURES"
                output_payload["message"] = "Phase A processing completed, but critical quality check failures were found."
            elif qc_report_obj.overall_status == "PASSED_WITH_WARNINGS":
                output_payload["status"] = "COMPLETED_WITH_QC_WARNINGS"
                output_payload["message"] = "Phase A processing completed with quality check warnings."
            else: # PASSED
                output_payload["status"] = "SUCCESS"
                output_payload["message"] = "Phase A processing completed successfully."
            
            self._logger_instance.info(f"Phase A completed for '{security_id}'. Final status: {output_payload['status']}")

        except Exception as e:
            self._logger_instance.error(f"Critical error in FinancialPhaseATool for '{security_id}': {e}", exc_info=True)
            output_payload["status"] = "ERROR"
            output_payload["message"] = f"An unexpected error occurred: {str(e)}"
        
        return json.dumps(output_payload, ensure_ascii=False, indent=2)


# --- Example Standalone Test for FinancialPhaseATool ---
# Mock LLM Service for testing
class MockPhaseALLMService(BaseLlmService):
    async def invoke(self, prompt: str, **kwargs) -> str:
        # This mock needs to be more sophisticated to handle orchestrator's specific prompts
        logger.info(f"[MockPhaseALLMService] Received prompt starting with: {prompt[:150]}...")
        if "extract the following metadata" in prompt.lower():
            return json.dumps({"target_company_name": "Mock Security Inc.", "ticker_symbol": "MSI", "currency": "USD", "fiscal_year_end": "12-31"})
        elif "identify the distinct historical financial periods" in prompt.lower():
            return json.dumps({"historical_period_labels": ["FY2023", "FY2022"]})
        elif "extract a specific financial data point" in prompt.lower():
            # Simplified: return a generic success for any line item
            return json.dumps({"value": 123.45, "currency": "USD", "unit": "actuals", "source_reference": "Mocked Page X", "status": "EXTRACTED_SUCCESSFULLY"})
        return json.dumps({"error": "MockPhaseALLMService - Unknown prompt type for this mock."})

async def main_test_real_parser():
    if MCPDocumentParserTool is None:
        logger.error("MCPDocumentParserTool not imported, cannot run test.")
        return

    logger.info("Starting MCPDocumentParserTool standalone test...")

    # --- IMPORTANT: CONFIGURE THIS PATH ---
    # Replace with the actual path to your document_parser_server.py
    # This assumes your Alpy project root is the parent of 'mcp_servers'
    project_root = Path(__file__).resolve().parent.parent.parent
    default_server_script = project_root / "mcp_servers" / "document_parser_server.py"
    
    # Path to a real PDF document you want to test with
    # --- !!! REPLACE THIS WITH YOUR PDF PATH !!! ---
    pdf_to_test = Path("/home/me/Documents/FundDocs/test_portfolio/_Matsuya_Үнэт_цаасны_танилцуулга_05_07.pdf")
    # --- !!! REPLACE THIS WITH YOUR PDF PATH !!! ---

    if not default_server_script.exists():
        logger.error(f"Server script not found: {default_server_script}")
        logger.error("Please ensure MCPDocumentParserTool's 'server_script' default points to the correct location,")
        logger.error("or pass the correct path during instantiation if it's configurable.")
        return
        
    if not pdf_to_test.exists() or not pdf_to_test.is_file():
        logger.error(f"Test PDF not found or is not a file: {pdf_to_test}")
        logger.error("Please provide a valid path to a PDF document for testing.")
        return

    # Instantiate the real MCPDocumentParserTool
    # It should use its default server_executable, server_script, and server_cwd_path
    # or you can override them here if needed.
    try:
        # If your MCPDocumentParserTool takes specific paths in constructor, provide them:
        # tool_instance = MCPDocumentParserTool(
        #     server_script=Path("/path/to/your/document_parser_server.py"),
        #     # server_executable=Path("/path/to/your/python"), # if not default python3
        #     # server_cwd_path=Path("/path/to/server_script_parent/"), # if needed
        # )
        # Otherwise, defaults from the tool's Pydantic model will be used.
        tool_instance = MCPDocumentParserTool()
        if hasattr(tool_instance, '_logger_instance') and tool_instance._logger_instance:
            tool_instance._logger_instance.setLevel(logging.DEBUG) # For verbose output from the tool
        
    except Exception as e_init:
        logger.error(f"Failed to instantiate MCPDocumentParserTool: {e_init}", exc_info=True)
        return

    test_input_args = {
        "document_path": str(pdf_to_test.resolve()),
        "ocr_mode": "auto", # Or "force_images", "force_full_pages", "none"
        "ocr_lang": "mon", # Or your desired language e.g., "eng"
        # "max_pages_to_process_for_testing": 5, # Optional: limit pages for faster testing
    }

    logger.info(f"Attempting to parse PDF: {pdf_to_test}")
    logger.info(f"Using server script (default or from tool): {tool_instance.server_script if hasattr(tool_instance, 'server_script') else 'N/A'}")
    logger.info(f"Tool input args: {test_input_args}")

    result_json_str = None
    try:
        result_json_str = await tool_instance._arun(**test_input_args)
        logger.info("MCPDocumentParserTool._arun call finished.")
        
        print("\n--- MCPDocumentParserTool Result ---")
        try:
            result_data = json.loads(result_json_str)
            # print(json.dumps(result_data, indent=2, ensure_ascii=False))
            
            # Basic assertions on successful output structure
            if result_data.get("status") == "success":
                assert "pages_content" in result_data
                assert "all_tables" in result_data
                logger.info("Basic assertions on output structure PASSED for successful parse.")
            else:
                logger.warning(f"Parsing was not fully successful. Status: {result_data.get('status')}, Message: {result_data.get('message')}")

        except json.JSONDecodeError:
            logger.error("Failed to parse tool output as JSON.")
            # print("Raw output:", result_json_str)
        except AssertionError as e_assert:
            logger.error(f"Assertion failed on tool output: {e_assert}")

    except ToolException as e_tool: # Specific LangChain tool exception
        logger.error(f"ToolException during MCPDocumentParserTool test run: {e_tool}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error during MCPDocumentParserTool test run: {e}", exc_info=True)
    finally:
        if 'tool_instance' in locals() and hasattr(tool_instance, 'close'):
            logger.info("Closing MCPDocumentParserTool session...")
            await tool_instance.close()
            logger.info("MCPDocumentParserTool session closed.")
        logger.info("MCPDocumentParserTool standalone test finished.")

# This is the existing main_test_financial_phase_a_tool function from your src/tools/financial_phase_a_tool.py
# Ensure the imports at the top of the file correctly bring in:
# - FinancialPhaseATool, MockPhaseALLMService (or a more advanced LLM mock/service)
# - FinancialModelingUtils
# - ImportedRealMCPDocumentParserTool (from previous fix, to get the real parser)
# - Path, logging, json, asyncio

async def main_test_financial_phase_a_tool():
    # Setup basic logging for the test
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s')
    test_logger = logging.getLogger(__name__ + ".TestFinancialPhaseATool") # Changed logger name for clarity
    test_logger.info("Starting FinancialPhaseATool (FULL WORKFLOW) standalone test...")

    project_root = Path(__file__).resolve().parent.parent.parent
    schema_file = project_root / "fund" / "financial_model_schema.json"
    prompts_file = project_root / "prompts" / "financial_modeling_prompts.yaml"
    
    # Ensure logs directory exists for orchestrator and parser client logs
    (project_root / "logs").mkdir(parents=True, exist_ok=True)


    if not schema_file.exists() or not prompts_file.exists():
        test_logger.error(f"Schema or Prompts file not found. Schema: {schema_file}, Prompts: {prompts_file}. Aborting test.")
        return

    dummy_doc_dir = project_root / "test_docs_phase_a_full_tool" # New dir for this test
    dummy_doc_dir.mkdir(exist_ok=True)
    
    pdf_to_test_with_full_tool_name = "_Matsuya_Үнэт_цаасны_танилцуулга_05_07.pdf" # Example name
    # --- !!! IMPORTANT: REPLACE WITH YOUR ACTUAL PDF PATH OR COPY IT HERE !!! ---
    # Option 1: Use a specific PDF path directly
    # pdf_to_test_path = Path("/home/me/Documents/FundDocs/test_portfolio/_Matsuya_Үнэт_цаасны_танилцуулга_05_07.pdf")
    # Option 2: Copy your test PDF into dummy_doc_dir for this test
    # shutil.copy("/home/me/Documents/FundDocs/test_portfolio/_Matsuya_Үнэт_цаасны_танилцуулга_05_07.pdf", dummy_doc_dir / pdf_to_test_with_full_tool_name)
    # pdf_to_test_path = dummy_doc_dir / pdf_to_test_with_full_tool_name
    
    # For this example, let's assume you will MANUALLY place your PDF in dummy_doc_dir
    # Or provide the full path directly:
    pdf_to_test_path = Path("/home/me/Documents/FundDocs/test_portfolio/_Matsuya_Үнэт_цаасны_танилцуулга_05_07.pdf") # REPLACE THIS
    # --- !!! IMPORTANT: ENSURE THIS PDF EXISTS !!! ---

    if not pdf_to_test_path.exists():
        test_logger.error(f"Test PDF for full tool test not found: {pdf_to_test_path}. Please configure the path.")
        # Attempt to create a very simple dummy if the target is not found, just to run the flow
        pdf_to_test_path = dummy_doc_dir / "fallback_dummy_report.pdf"
        pdf_to_test_with_full_tool_name = "fallback_dummy_report.pdf" # Update name for args
        try:
            from reportlab.pdfgen import canvas
            c = canvas.Canvas(str(pdf_to_test_path))
            c.drawString(100, 750, f"Fallback Dummy PDF for FinancialPhaseATool Test. Target: {pdf_to_test_with_full_tool_name}")
            c.showPage(); c.save()
            test_logger.info(f"Created fallback dummy PDF: {pdf_to_test_path}")
        except ImportError:
            test_logger.error("reportlab not installed and target PDF not found. Cannot proceed.")
            return
        except Exception as e_pdf:
            test_logger.error(f"Error creating fallback dummy PDF: {e_pdf}")
            return


    # --- Instantiate with REAL MCPDocumentParserTool ---
    if MCPDocumentParserTool is None: # Check if import was successful
        test_logger.error("Real MCPDocumentParserTool (ImportedRealMCPDocumentParserTool) not available. Aborting.")
        return
    
    real_parser_tool_instance = MCPDocumentParserTool()
    # Set log level for the parser instance if needed
    if hasattr(real_parser_tool_instance, '_logger_instance') and real_parser_tool_instance._logger_instance:
        real_parser_tool_instance._logger_instance.setLevel(logging.DEBUG) # More logs from parser

    # Use a simple mock LLM for now for PhaseAOrchestrator.
    # For real extraction, this needs to be a capable LLM service.
    llm_service_instance = GeminiLlmService()

    utils = FinancialModelingUtils() # This is also defined/imported in your file

    tool_instance = FinancialPhaseATool(
        mcp_document_parser_tool=real_parser_tool_instance, # USE THE REAL PARSER
        llm_service=llm_service_instance,
        financial_utils=utils,
        schema_definition_path=schema_file,
        prompts_path=prompts_file
    )
    if hasattr(tool_instance, '_logger_instance') and tool_instance._logger_instance:
         tool_instance._logger_instance.setLevel(logging.DEBUG)


    test_input_args = {
        "security_id": "MATSUYA_TEST_001",
        "documents_root_path": str(pdf_to_test_path.parent.resolve()), # Directory containing the PDF
        "primary_document_name": pdf_to_test_path.name, # Name of the PDF file
        "document_type_hint": "Bond Prospectus", # Or whatever is appropriate
        "ocr_mode": "auto",
        "ocr_lang": "mon", # Mongolian
        "max_pages_to_parse_per_doc": None # Process all pages from the primary doc
    }

    test_logger.info(f"Invoking FinancialPhaseATool (FULL WORKFLOW) with args: {test_input_args}")
    result_json_str = ""
    try:
        result_json_str = await tool_instance._arun(**test_input_args)
        test_logger.info("FinancialPhaseATool (FULL WORKFLOW) execution finished.")
        
        print("\n--- FinancialPhaseATool (FULL WORKFLOW) Result ---")
        try:
            result_data = json.loads(result_json_str)
            # Save the full output to a file for inspection
            output_file_path = project_root / "logs" / f"financial_phase_a_tool_output_{test_input_args['security_id']}.json"
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            test_logger.info(f"Full output saved to: {output_file_path}")

            # Print a summary
            print(f"Status: {result_data.get('status')}")
            print(f"Message: {result_data.get('message')}")
            if result_data.get("parsed_documents_summary"):
                print(f"Parsed Docs Summary: {result_data['parsed_documents_summary']}")
            if result_data.get("quality_report"):
                qr = result_data['quality_report']
                print(f"Quality Report Overall Status: {qr.get('overall_status')}")
                print(f"  Extraction Success Rate: {qr.get('data_extraction_summary', {}).get('overall_extraction_success_rate_percent', 'N/A')}%")
                print(f"  Schema Valid: {qr.get('schema_validation_results', {}).get('is_valid', 'N/A')}")
                print(f"  Sanity Checks Passed: {len([s for s in qr.get('sanity_check_results', []) if s.get('status') == 'PASSED'])} / {len(qr.get('sanity_check_results', []))}")
            
            # Basic assertions
            assert result_data.get("status") in ["SUCCESS", "COMPLETED_WITH_QC_WARNINGS", "COMPLETED_WITH_CRITICAL_QC_FAILURES"], f"Unexpected status: {result_data.get('status')}"
            assert result_data.get("financial_model_phase_a") is not None, "financial_model_phase_a is missing"
            assert result_data.get("quality_report") is not None, "quality_report is missing"
            test_logger.info("Basic assertions on FinancialPhaseATool output structure passed.")

        except json.JSONDecodeError:
            test_logger.error("Failed to parse FinancialPhaseATool output as JSON.")
            # print("Raw output:", result_json_str)
        except AssertionError as e_assert:
            test_logger.error(f"Assertion failed on FinancialPhaseATool output: {e_assert}")

    except ToolException as e_tool:
        test_logger.error(f"ToolException during FinancialPhaseATool (FULL WORKFLOW) test: {e_tool}", exc_info=True)
        # if result_json_str: print("Partial/Error output:", result_json_str)
    except Exception as e:
        test_logger.error(f"Error during FinancialPhaseATool (FULL WORKFLOW) test: {e}", exc_info=True)
        # if result_json_str: print("Partial/Error output:", result_json_str)
    finally:
        # Close the FinancialPhaseATool itself (which should close its underlying tools if they have close methods, like the parser)
        if hasattr(tool_instance, 'close') and callable(tool_instance.close): # Check if tool_instance itself has close, though BaseTool does not
             pass # FinancialPhaseATool itself does not have a close method, relies on constituent tools
        
        # Explicitly close the real_parser_tool_instance
        if 'real_parser_tool_instance' in locals() and hasattr(real_parser_tool_instance, 'close'):
            test_logger.info("Closing Real MCPDocumentParserTool instance used by FinancialPhaseATool...")
            await real_parser_tool_instance.close()
            test_logger.info("Real MCPDocumentParserTool instance closed.")

        if dummy_doc_dir.exists(): # Clean up the directory if it was specifically for this test
             try:
                 # Only remove if it was the specific one created, be careful here
                 if "test_docs_phase_a_full_tool" in str(dummy_doc_dir) or "fallback_dummy_report.pdf" in str(pdf_to_test_path):
                     import shutil
                     shutil.rmtree(dummy_doc_dir)
                     test_logger.info(f"Cleaned up test directory: {dummy_doc_dir}")
             except OSError as e_del:
                 test_logger.warning(f"Could not delete test directory {dummy_doc_dir}: {e_del}")
        test_logger.info("FinancialPhaseATool (FULL WORKFLOW) standalone test finished.")
    
if __name__ == "__main__":
    # Ensure logs directory exists if any component tries to write there by default
    Path("logs").mkdir(parents=True, exist_ok=True)
    
    # To run this specific test:
    # 1. Replace the main_test_financial_phase_a_tool() call with main_test_real_parser()
    #    at the bottom of financial_phase_a_tool.py.
    # 2. OR, if you want to keep both, modify the if __name__ == "__main__": block
    #    to choose which test to run, e.g., based on a command-line argument.
    # For now, assuming you replace it:
    
    # Comment out or remove the FinancialPhaseATool test call:
    # asyncio.run(main_test_financial_phase_a_tool())
    
    # Add the call for the real parser test:
    # asyncio.run(main_test_real_parser())

    asyncio.run(main_test_financial_phase_a_tool())
