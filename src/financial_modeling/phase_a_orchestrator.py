from typing import List, Dict, Optional, Any, Union, Tuple, Literal, Type
import logging
from pathlib import Path
from pydantic import BaseModel, Field
import asyncio
import yaml
import google.api_core.exceptions
import random # For jitter in backoff
import json
import re
import datetime
import os
import sys
import shutil
import random
import time
import os
import threading

from .. import config as alpy_config
from ..rag.embedding_service import GeminiEmbeddingService
from ..rag.vector_store_manager import FaissVectorStoreManager
from ..rag.rag_manager import RagManager
from ..rag.retriever import RagRetriever
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

# Placeholder Pydantic Models (as per finmodel_rag_guide.md context)
# These would ideally be in a shared models.py file


class _InternalGeminiLLMService:
    def __init__(self, api_key: str, logger: logging.Logger, model_name: str = "gemini-2.0-flash"):
        self.logger = logger
        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            self.logger.error(f"Failed to configure Gemini API: {e}", exc_info=True)
            raise
        
        self.model_name = model_name
        self.generation_config = GenerationConfig()
        try:
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini Model ({self.model_name}): {e}", exc_info=True)
            raise

        # --- Concurrency Control ---
        self.request_semaphore = asyncio.Semaphore(alpy_config.GEMINI_CONCURRENT_REQUEST_LIMIT)

        # --- RPM Limiter ---
        self.rpm_limit = alpy_config.GEMINI_RPM_LIMIT
        self.rpm_window_seconds = 60
        self.rpm_request_timestamps = []
        self.rpm_lock = asyncio.Lock()

        # --- TPM Limiter ---
        self.tpm_limit = alpy_config.GEMINI_TPM_LIMIT
        self.tpm_window_seconds = 60
        self.tpm_records = []  # Stores (timestamp, token_count)
        self.tpm_lock = asyncio.Lock()

        # --- RPD Limiter ---
        self.rpd_limit = alpy_config.GEMINI_RPD_LIMIT
        self.rpd_current_day_count = 0
        self.rpd_last_reset_date = datetime.date.min 
        self.rpd_lock = asyncio.Lock()
        self.daily_quota_exceeded_flag = False 

        # --- Retry Settings ---
        self.max_retries = alpy_config.GEMINI_MAX_RETRIES
        self.initial_retry_delay_seconds = alpy_config.GEMINI_INITIAL_RETRY_DELAY_SECONDS

    async def _count_tokens(self, text_content: str) -> int:
        if not text_content:
            return 0
        try:
            count_response = await self.model.count_tokens_async(text_content)
            return count_response.total_tokens
        except Exception as e:
            self.logger.warning(f"Token counting failed for model {self.model_name}: {e}. Estimating 0 tokens.")
            return 0

    async def _enforce_rpm_limit(self):
        async with self.rpm_lock:
            while True:
                current_time = time.time()
                self.rpm_request_timestamps = [ts for ts in self.rpm_request_timestamps if current_time - ts < self.rpm_window_seconds]
                if len(self.rpm_request_timestamps) < self.rpm_limit:
                    self.rpm_request_timestamps.append(current_time)
                    return
                
                oldest_ts = min(self.rpm_request_timestamps)
                wait_time = (oldest_ts + self.rpm_window_seconds) - current_time
                wait_time = max(0.1, wait_time) 
                self.logger.warning(f"RPM limit ({self.rpm_limit}/{self.rpm_window_seconds}s) hit. Waiting {wait_time:.2f}s. Requests in window: {len(self.rpm_request_timestamps)}")
                self.rpm_lock.release()
                try:
                    await asyncio.sleep(wait_time + 0.05) 
                finally:
                    await self.rpm_lock.acquire()
    
    async def _enforce_tpm_limit(self, prompt_tokens: int):
        async with self.tpm_lock:
            while True:
                current_time = time.time()
                self.tpm_records = [(ts, tc) for ts, tc in self.tpm_records if current_time - ts < self.tpm_window_seconds]
                current_total_tokens_in_window = sum(tc for _, tc in self.tpm_records)

                if current_total_tokens_in_window + prompt_tokens <= self.tpm_limit:
                    return

                if not self.tpm_records: 
                    self.logger.warning("TPM limit: tpm_records empty during wait calculation but limit exceeded. This is unexpected. Proceeding cautiously after short delay.")
                    wait_time = 0.5
                else:
                    oldest_ts, _ = min(self.tpm_records, key=lambda x: x[0])
                    wait_time = (oldest_ts + self.tpm_window_seconds) - current_time
                    wait_time = max(0.1, wait_time)
                
                self.logger.warning(f"TPM limit ({self.tpm_limit}/{self.tpm_window_seconds}s) potentially hit. Waiting {wait_time:.2f}s. Tokens in window: {current_total_tokens_in_window}, prompt: {prompt_tokens}")
                self.tpm_lock.release()
                try:
                    await asyncio.sleep(wait_time + 0.05)
                finally:
                    await self.tpm_lock.acquire()

    def _add_tpm_record_unsafe(self, total_tokens_for_call: int):
        current_time = time.time()
        self.tpm_records.append((current_time, total_tokens_for_call))
        self.tpm_records = [(ts, tc) for ts, tc in self.tpm_records if current_time - ts < self.tpm_window_seconds]

    async def _enforce_rpd_limit(self):
        async with self.rpd_lock:
            today = datetime.date.today()
            if today != self.rpd_last_reset_date:
                self.logger.info(f"RPD counter resetting for new day: {today}. Previous count: {self.rpd_current_day_count} for {self.rpd_last_reset_date}")
                self.rpd_current_day_count = 0
                self.rpd_last_reset_date = today
                self.daily_quota_exceeded_flag = False

            if self.daily_quota_exceeded_flag:
                 self.logger.error(f"Daily RPD quota ({self.rpd_limit}) previously hit for {self.rpd_last_reset_date}. Request blocked.")
                 raise google.api_core.exceptions.ResourceExhausted(f"Daily RPD Quota Exceeded on {self.rpd_last_reset_date} (internal flag)")

            if self.rpd_current_day_count >= self.rpd_limit:
                self.logger.error(f"Daily RPD quota ({self.rpd_limit}) reached for {self.rpd_last_reset_date}. Count: {self.rpd_current_day_count}. Blocking further requests today.")
                self.daily_quota_exceeded_flag = True
                raise google.api_core.exceptions.ResourceExhausted(f"Daily RPD Quota ({self.rpd_limit}) Reached on {self.rpd_last_reset_date}")

    def _increment_rpd_count_unsafe(self):
        today = datetime.date.today()
        if today != self.rpd_last_reset_date: 
            self.logger.info(f"RPD counter resetting (during increment) for new day: {today}. Previous count: {self.rpd_current_day_count}")
            self.rpd_current_day_count = 1
            self.rpd_last_reset_date = today
            self.daily_quota_exceeded_flag = False
        else:
            self.rpd_current_day_count += 1
        self.logger.debug(f"RPD count for {self.rpd_last_reset_date}: {self.rpd_current_day_count}/{self.rpd_limit}")

    async def generate_text_response(self, prompt: str) -> Optional[str]:
        async with self.rpd_lock:
            if self.daily_quota_exceeded_flag and datetime.date.today() == self.rpd_last_reset_date:
                self.logger.warning(f"Skipping request - daily RPD quota previously hit for {self.model_name} on {self.rpd_last_reset_date}.")
                return ""
            elif self.daily_quota_exceeded_flag and datetime.date.today() != self.rpd_last_reset_date:
                self.logger.info(f"RPD flag auto-reset for new day: {datetime.date.today()}.")
                self.rpd_current_day_count = 0
                self.rpd_last_reset_date = datetime.date.today()
                self.daily_quota_exceeded_flag = False
        
        prompt_tokens = await self._count_tokens(prompt)
        if prompt_tokens == 0 and prompt: 
            self.logger.warning("Prompt token count failed or zero for non-empty prompt. Using estimate of 250 for TPM pre-check.")
            prompt_tokens = 250 

        current_retry_delay_seconds = self.initial_retry_delay_seconds

        for attempt in range(self.max_retries + 1):
            try:
                await self._enforce_rpd_limit()
                await self._enforce_rpm_limit()
                await self._enforce_tpm_limit(prompt_tokens) 

                async with self.request_semaphore:
                    self.logger.debug(f"Attempt {attempt + 1}/{self.max_retries + 1} for {self.model_name}. Prompt tokens: {prompt_tokens}")
                    api_response = await self.model.generate_content_async(prompt)

                    if api_response.prompt_feedback and api_response.prompt_feedback.block_reason:
                        self.logger.error(f"Prompt blocked for {self.model_name}. Reason: {api_response.prompt_feedback.block_reason}. Ratings: {api_response.prompt_feedback.safety_ratings}")
                        return "" 

                    response_text = "".join(part.text for part in api_response.candidates[0].content.parts)
                    response_tokens = await self._count_tokens(response_text)
                    if response_tokens == 0 and response_text:
                        self.logger.warning("Response token count failed or zero. Estimating 100 for TPM record.")
                        response_tokens = 100
                    
                    total_tokens_this_call = prompt_tokens + response_tokens

                    async with self.rpd_lock:
                        self._increment_rpd_count_unsafe()
                    async with self.tpm_lock:
                        self._add_tpm_record_unsafe(total_tokens_this_call)
                    
                    if not response_text.strip():
                        self.logger.warning(f"{self.model_name} returned an empty response.")
                    return response_text

            except google.api_core.exceptions.ResourceExhausted as e:
                self.logger.warning(f"ResourceExhausted (attempt {attempt + 1}/{self.max_retries + 1}) for {self.model_name}: {e}")
                
                error_str_lower = str(e).lower()
                is_api_daily_quota = (
                    "perday" in error_str_lower or 
                    "daily quota" in error_str_lower or 
                    "generaterequestsperdayperprojectpermodel" in error_str_lower
                )

                if is_api_daily_quota:
                    self.logger.error(f"API indicates daily quota exceeded for {self.model_name}. Setting flag and stopping retries. Error details: {e}")
                    async with self.rpd_lock:
                        self.daily_quota_exceeded_flag = True
                        self.rpd_last_reset_date = datetime.date.today()
                    return ""

                if "Daily RPD Quota Exceeded" in str(e) or "Daily RPD Quota Reached" in str(e): # Our internal flag
                    async with self.rpd_lock: 
                        self.daily_quota_exceeded_flag = True
                        self.rpd_last_reset_date = datetime.date.today()
                    self.logger.error(f"Internal daily RPD quota hit confirmed by error message. No more requests today for {self.model_name}.")
                    return "" 

                if attempt >= self.max_retries:
                    self.logger.error(f"Max retries ({self.max_retries}) reached for {self.model_name} due to ResourceExhausted. Error: {e}", exc_info=True)
                    if "quota" in error_str_lower or "limit" in error_str_lower: # General check after max retries
                         async with self.rpd_lock:
                            self.logger.info(f"Assuming daily quota hit after max retries with error containing 'quota' or 'limit'.")
                            self.daily_quota_exceeded_flag = True 
                            self.rpd_last_reset_date = datetime.date.today()
                    return ""

                wait_time = current_retry_delay_seconds 
                if hasattr(e, 'metadata') and e.metadata:
                    self.logger.debug(f"Raw API error metadata for ResourceExhausted: {e.metadata}")
                    for item_tuple in e.metadata:
                        if item_tuple and len(item_tuple) == 2 and item_tuple[0] == 'retry_delay':
                            delay_value_str = str(item_tuple[1]).lower().replace('s', '')
                            if delay_value_str.isdigit():
                                api_suggested_delay = int(delay_value_str)
                                wait_time = max(api_suggested_delay, 1) 
                                self.logger.info(f"Using API suggested retry delay: {wait_time}s. Original calculated base delay was: {current_retry_delay_seconds:.2f}s")
                                break
                
                self.logger.info(f"Retrying in {wait_time:.2f}s (jitter to be added). Current internal backoff progression: {current_retry_delay_seconds:.2f}s")
                await asyncio.sleep(wait_time + random.uniform(0, min(wait_time * 0.2, 5.0))) 
                current_retry_delay_seconds = min(current_retry_delay_seconds * 1.5, 90) 

            except google.generativeai.types.BlockedPromptException as e:
                self.logger.error(f"Prompt blocked by Gemini API (attempt {attempt + 1}): {e}", exc_info=True)
                return "" 

            except Exception as e:
                self.logger.error(f"Gemini API call failed (attempt {attempt + 1}/{self.max_retries + 1}) with unexpected error: {e}", exc_info=True)
                if attempt >= self.max_retries:
                    return ""
                await asyncio.sleep(current_retry_delay_seconds + random.uniform(0, 1))
                current_retry_delay_seconds = min(current_retry_delay_seconds * 1.5, 90)

        self.logger.error(f"All {self.max_retries + 1} attempts failed for {self.model_name}.")
        return ""

class ExtractedMetadata(BaseModel):
    company_name: Optional[str] = None
    ticker_symbol: Optional[str] = None
    primary_exchange: Optional[str] = None
    reporting_currency: Optional[str] = None
    fiscal_year_end: Optional[str] = None # MM-DD
    latest_report_date: Optional[str] = None # YYYY-MM-DD
    latest_period_end_date: Optional[str] = None # YYYY-MM-DD
    document_type: Optional[str] = None
    document_source_name: Optional[str] = None

class HistoricalPeriod(BaseModel):
    period_id: str # e.g., "FY2023", "Q1-2024"
    period_label: str # As seen in document, e.g., "2023", "3 сар 2024"
    start_date: Optional[str] = None # YYYY-MM-DD
    end_date: Optional[str] = None # YYYY-MM-DD
    period_type: Literal["fiscal_year", "quarter", "half_year", "trailing_twelve_months", "custom"] = "custom"
    order_in_document: Optional[int] = None # For sorting as they appear

class LineItemData(BaseModel):
    item_name_english: str
    item_name_mongolian: Optional[str] = None
    period_id: str # Links to HistoricalPeriod.period_id
    value: Optional[Any] = None # Could be float, int, str (e.g. "N/A")
    unit: Optional[str] = None # e.g., "MNT", "USD", "%"
    page_reference: Optional[int] = None
    notes: Optional[str] = None
    extraction_status: Optional[str] = None # Status from LLM extraction (e.g., SUCCESS, NOT_FOUND_PER_LLM)

class HistoricalPeriodsExtractionResponse(BaseModel):
    historical_periods: List[HistoricalPeriod] = Field(default_factory=list)

class SingleLineItemLLMOutput(BaseModel):
    value: Optional[Union[float, int]] = None
    currency: Optional[str] = None
    unit: Optional[str] = None
    source_reference: Optional[str] = None
    status: Optional[str] = None # EXTRACTED_SUCCESSFULLY_OR_CONFIRMED_NOT_FOUND


class LineItemsExtractionResponse(BaseModel):
    line_items: List[LineItemData] = Field(default_factory=list)


class LineItemTarget(BaseModel):
    """Represents a target line item derived from the schema for extraction."""
    line_item_name: str
    line_item_name_mongolian: Optional[str] = None
    statement_key: str  # e.g., "income_statement", "balance_sheet"
    section_path: str   # e.g., "line_items", "assets.current_assets"
    schema_path: str    # Full path in the schema, e.g., "financial_statements_core.income_statement.line_items[0]"
    is_total_or_subtotal: bool = False # Indicates if it's a total/subtotal that might not need direct LLM extraction
    raw_schema_definition: Dict[str, Any] # The actual schema definition for this item


class PhaseAOrchestrator:

    def __init__(self, security_id: str, prompts_yaml_path: Path, financial_schema_path: Path, output_log_path: Optional[Path] = None):
        """Initializes the Phase A orchestrator.

        Args:
            security_id: The security ID for which the model is being generated.
            prompts_yaml_path: Path to the YAML file containing prompts.
            llm_service: An instance of an LLM service interface.
            output_log_path: Optional path to save the output log.
        """
        self.security_id = security_id
        self.output_log_path = output_log_path
        self.logger = self._setup_logger() # Logger needs to be setup before LLM service if passed

        # Load prompts configuration from YAML
        self.prompts_config = {}
        if prompts_yaml_path and prompts_yaml_path.exists():
            try:
                with open(prompts_yaml_path, 'r', encoding='utf-8') as f:
                    self.prompts_config = yaml.safe_load(f) or {}
                if not self.prompts_config:
                    self.logger.warning(f"Prompts file {prompts_yaml_path} is empty or invalid. Using minimal fallback prompts.")
                    # Provide minimal structure if file is empty/invalid to avoid None errors later
                    self.prompts_config = {
                        "metadata_extraction": {"prompt_template": "Extract metadata. Context: {context}"},
                        "historical_period_extraction": {"prompt_template": "Extract periods. Context: {context}"},
                        "single_line_item_extraction": {"prompt_template": "Extract item: {line_item_english_name} for {period_label}. Context: {context}"}
                    }
                else:
                    self.logger.info(f"Loaded {len(self.prompts_config)} prompt configurations from {prompts_yaml_path}")
            except Exception as e:
                self.logger.error(f"Error loading prompts from {prompts_yaml_path}: {e}. Orchestrator might not function correctly.", exc_info=True)
                # Critical error, consider raising or setting a more robust default/faulty state
                raise RuntimeError(f"Failed to load prompts configuration from {prompts_yaml_path}") from e
        else:
            self.logger.error(f"Prompts YAML file not found at {prompts_yaml_path}. Orchestrator cannot function without prompts.")
            raise FileNotFoundError(f"Prompts YAML file not found at {prompts_yaml_path}")

        # Initialize internal LLM Service
        try:
            self.llm_service = _InternalGeminiLLMService(
                api_key=alpy_config.GOOGLE_API_KEY,
                logger=self.logger
                # model_name will use the default from _InternalGeminiLLMService class definition
            )
        except Exception as e:
            self.logger.critical(f"Failed to initialize _InternalGeminiLLMService: {e}. Orchestrator cannot function.", exc_info=True)
            raise RuntimeError(f"Critical: _InternalGeminiLLMService initialization failed.") from e

        self.embedding_service = GeminiEmbeddingService(api_key=alpy_config.GOOGLE_API_KEY)
        self.vector_store_manager = FaissVectorStoreManager(
            vector_stores_base_path=Path(alpy_config.RAG_VECTOR_STORE_BASE_PATH)
        )
        self.rag_retriever = RagRetriever(
            vector_store_manager=self.vector_store_manager,
            embedding_service=self.embedding_service
        )

        # Initialize RagManager
        self.rag_manager = RagManager(
            embedding_service=self.embedding_service,
            vector_store_manager=self.vector_store_manager,
            logger=self.logger # Pass the orchestrator's logger
        )

        self.parsed_financial_schema: Optional[Dict[str, Any]] = None
        self._load_schema_and_derive_targets(financial_schema_path)
        self.model_data = self._create_initial_model(security_id)

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"PhaseAOrchestrator_{self.security_id}")
        logger.setLevel(alpy_config.LOG_LEVEL)
        # Basic console handler
        if not logger.handlers: # This check ensures we don't add handlers if they already exist (e.g. from parent loggers)
            ch = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        # Add file handler if output_log_path is provided
        if self.output_log_path:
            try:
                # Ensure the directory for the log file exists
                log_file_path = Path(self.output_log_path)
                log_file_path.parent.mkdir(parents=True, exist_ok=True)

                # Assuming 'formatter' is defined from the console handler setup earlier in this method
                # If not, a default one should be created here.
                # For robustness, let's ensure 'formatter' is available or define a fallback.
                current_formatter = None
                if logger.handlers and isinstance(logger.handlers[0].formatter, logging.Formatter):
                    current_formatter = logger.handlers[0].formatter
                else:
                    # Fallback formatter if console handler/formatter isn't set up as expected or not first
                    current_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

                fh = logging.FileHandler(self.output_log_path, mode='w', encoding='utf-8') # 'w' to overwrite for each test run
                fh.setFormatter(current_formatter) 
                logger.addHandler(fh)
                logger.info(f"Logging to file: {self.output_log_path}")
            except Exception as e:
                # Use a basic print here if logger itself is failing during setup for file handler
                print(f"ERROR [PhaseAOrchestrator._setup_logger]: Failed to set up file handler for {self.output_log_path}: {e}")
                # Optionally, re-raise or handle more gracefully depending on desired behavior

        return logger

    def _create_initial_model(self, security_id: str) -> Dict[str, Any]:
        """Create initial model structure purely from schema.
        
        Args:
            security_id: The security ID for the model.
            
        Returns:
            Dict with initial model structure following schema format.
        """
        if not self.parsed_financial_schema:
            self.logger.error("Schema not loaded. Cannot create initial model.")
            return {}
        
        def _resolve_ref(ref: str, schema: Dict[str, Any]) -> Dict[str, Any]:
            """Resolve a JSON schema $ref."""
            if not ref.startswith("#/$defs/"):
                return {}
            
            path = ref.replace("#/", "").split("/")
            current = schema
            for part in path:
                if part not in current:
                    return {}
                current = current[part]
            return current
        
        def _merge_schemas(schemas: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Merge multiple schemas into one."""
            result = {}
            for schema in schemas:
                if "properties" in schema:
                    if "properties" not in result:
                        result["properties"] = {}
                    result["properties"].update(schema["properties"])
                if "type" in schema and "type" not in result:
                    result["type"] = schema["type"]
                if "default" in schema:
                    result["default"] = schema["default"]
            return result
        
        def _init_from_schema(schema_node: Dict[str, Any]) -> Any:
            """Recursively initialize structure from schema node."""
            # Handle $ref
            if "$ref" in schema_node:
                schema_node = _resolve_ref(schema_node["$ref"], self.parsed_financial_schema)
            
            # Handle allOf
            if "allOf" in schema_node:
                schemas = [_init_from_schema(s) if "$ref" in s else s for s in schema_node["allOf"]]
                schema_node = _merge_schemas(schemas)
                if "properties" in schema_node:
                    schema_node["properties"].update(schema_node.get("properties", {}))
            
            if "type" not in schema_node:
                return None
                
            if schema_node["type"] == "object":
                obj = {}
                if "properties" in schema_node:
                    for prop_name, prop_schema in schema_node["properties"].items():
                        obj[prop_name] = _init_from_schema(prop_schema)
                if "default" in schema_node:
                    if isinstance(schema_node["default"], dict):
                        # Deep merge for objects
                        deep_update = lambda d1, d2: {k: deep_update(d1.get(k, {}), v) 
                                                    if isinstance(v, dict) and k in d1 
                                                    else v for k, v in d2.items()}
                        obj = deep_update(obj, schema_node["default"])
                    else:
                        obj = schema_node["default"]
                return obj
                
            elif schema_node["type"] == "array":
                if "items" in schema_node:
                    # Initialize array with empty item structure
                    item_schema = schema_node["items"]
                    if "default" in schema_node:
                        return schema_node["default"]
                    return []
                return schema_node.get("default", [])
                
            else: # string, number, etc
                return schema_node.get("default")
        
        # Create structure based on schema
        model = _init_from_schema(self.parsed_financial_schema)
        
        # Only add runtime values that must be present
        if isinstance(model.get("model_metadata"), dict):
            model["model_metadata"].update({
                "security_id": security_id,
                "status_overall_model_generation": "PENDING",
                "date_created": datetime.datetime.now(datetime.timezone.utc).isoformat()
            })
        
        # Ensure financial statements core sections are initialized
        if "financial_statements_core" not in model:
            model["financial_statements_core"] = {}
        
        # Initialize statement sections with empty line_items arrays
        for statement in ["income_statement", "balance_sheet", "cash_flow_statement"]:
            if statement not in model["financial_statements_core"]:
                model["financial_statements_core"][statement] = {}
            if "line_items" not in model["financial_statements_core"][statement]:
                model["financial_statements_core"][statement]["line_items"] = []
        
        return model

    def _update_model_metadata(self, metadata: ExtractedMetadata) -> None:
        """Update model_metadata section with extracted data.
        
        Args:
            metadata: The extracted metadata to update with.
        """
        if not self.model_data:
            self.logger.error("Model data not initialized. Cannot update metadata.")
            return
            
        self.model_data["model_metadata"].update({
            "company_name": metadata.company_name,
            "reporting_currency": metadata.reporting_currency,
            "document_type": metadata.document_type
        })

    def _add_line_item(self, item: LineItemData) -> None:
        """Add a line item to the correct statement section.
        
        Args:
            item: The line item data to add.
        """
        if not self.model_data:
            self.logger.error("Model data not initialized. Cannot add line item.")
            return
            
        # Determine which statement this belongs to
        statement = "income_statement" if "income" in item.statement_key.lower() else "balance_sheet"
        
        # Create line item in schema format
        line_item = {
            "name": item.line_item_name,
            "name_mn": item.line_item_name_mn,
            "periods": [{
                "period_label": period.period_label,
                "value": period.value,
                "source_reference": period.source_reference
            } for period in item.periods]
        }
        
        self.model_data["financial_statements_core"][statement]["line_items"].append(line_item)




    async def process_documents_for_rag(self, parsed_documents: List[Dict[str, Any]], document_names: List[str]) -> bool:
        """Processes a list of parsed documents to build/update the RAG vector store."""
        if len(parsed_documents) != len(document_names):
            self.logger.error("Mismatch between number of parsed documents and document names.")
            raise ValueError("Parsed documents and document names lists must have the same length.")

        total_chunks_processed = 0
        for i, parsed_doc in enumerate(parsed_documents):
            doc_name = document_names[i]
            # Assuming self.rag_manager.process_document_for_rag is the correct method call
            chunks_count = await self.rag_manager.process_document_for_rag(
                security_id=self.security_id,
                parsed_doc_content=parsed_doc,
                document_name=doc_name
            )
            total_chunks_processed += chunks_count

        if total_chunks_processed > 0:
            self.logger.info(f"Successfully processed {total_chunks_processed} chunks in total for RAG for security ID {self.security_id}.")
            return True
        else:
            self.logger.warning(f"No chunks were processed for RAG for security ID {self.security_id}. Vector store might be empty or not updated.")
            return False

    async def _get_rag_context_for_llm(self, query_elements: List[str], k_retrieved_chunks: Optional[int] = None, max_context_length: int = 10000) -> str:
        """DEPRECATED: This method's logic is now integrated into _run_llm_extraction_task using RagManager.
        Retrieves relevant RAG context for a given set of query elements."""
        self.logger.warning("_get_rag_context_for_llm is deprecated and should not be called directly. Logic moved to _run_llm_extraction_task.")
        query_str = " ".join(query_elements)
        context_chunks = await self.rag_manager.retrieve_context_chunks(
            security_id=self.security_id,
            query=query_str,
            k_retrieved_chunks=k_retrieved_chunks
        )
        rag_context = "\n\n---\n\n".join([chunk.text_content for chunk in context_chunks])
        if len(rag_context) > max_context_length:
            rag_context = rag_context[:max_context_length]
        return rag_context

    async def _extract_financial_data_from_chunk(self, chunk, item_name_english, item_name_mongolian, period_label):
        """Extract financial data directly from a document chunk by looking for key patterns."""
        # Check if this is likely a financial table
        has_financial_indicators = False
        lower_content = chunk.text_content.lower()
        
        # Look for common financial statement headers/sections
        financial_indicators = [
            'balance sheet', 'баланс', 'income statement', 'орлогын тайлан', 
            'cash flow', 'мөнгөн гүйлгээ', 'financial statement', 'санхүүгийн тайлан',
            'profit', 'loss', 'revenue', 'ашиг', 'алдагдал', 'орлого', 'зардал',
            'total assets', 'нийт хөрөнгө', 'liabilities', 'өр төлбөр'
        ]
        
        for indicator in financial_indicators:
            if indicator in lower_content:
                has_financial_indicators = True
                break
        
        if not has_financial_indicators:
            return None
        
        # Look for the item name (both English and Mongolian versions)
        item_patterns = [re.escape(item_name_english.lower())]
        if item_name_mongolian:
            item_patterns.append(re.escape(item_name_mongolian.lower()))
        
        # Add common variations for key financial terms
        if 'profit' in item_name_english.lower():
            item_patterns.extend(['ашиг', 'орлого'])
        if 'revenue' in item_name_english.lower():
            item_patterns.extend(['борлуулалт', 'орлого'])
        if 'ebit' in item_name_english.lower():
            item_patterns.extend(['хүүгийн өмнөх ашиг', 'татварын өмнөх ашиг'])
            
        # Find lines containing the item name
        item_matches = []
        for pattern in item_patterns:
            # Look for lines with the item name followed by numbers
            matches = re.finditer(f'.*{pattern}.*?(\d[\d\s,.]*\d).*', lower_content, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                line = match.group(0)
                value = match.group(1).strip()
                item_matches.append((line, value))
        
        # If direct matches found, try to find the one matching the period
        if item_matches:
            # First look for exact period match with the value
            period_pattern = re.escape(period_label.lower())
            period_matches = []
            for line, value in item_matches:
                if re.search(period_pattern, line.lower()):
                    period_matches.append((line, value, "direct_period_match"))
            
            # Take the first match or return None if no matches
            if period_matches:
                best_match = period_matches[0]
                return {
                    "found": True,
                    "line": best_match[0],
                    "value": best_match[1],
                    "match_type": best_match[2]
                }
        
        # No direct match found
        return None

    async def _run_llm_extraction_task(self, task_name: str, query_elements: List[str], pydantic_model: Type[BaseModel], llm_prompt_template_key: str, additional_prompt_format_vars: Optional[Dict[str, Any]] = None) -> Optional[BaseModel]:
        """Runs a specific LLM extraction task, including RAG context retrieval and response validation.

        Args:
            task_name: A descriptive name for the task (for logging and raw responses).
            query_elements: List of strings to form the RAG query.
            pydantic_model: The Pydantic model to validate and parse the LLM's JSON response into.
            llm_prompt_template_key: The key to retrieve the prompt template from self.prompts_config.
            additional_prompt_format_vars: Optional dictionary of additional variables for prompt formatting.

        Returns:
            An instance of the pydantic_model if successful, None otherwise.
        """
        self.logger.info(f"Starting LLM task: {task_name}")
        prompt_template_data = self.prompts_config.get(llm_prompt_template_key)
        if not prompt_template_data:
            self.logger.error(f"Prompt template key '{llm_prompt_template_key}' not found in prompts configuration.")
            if not hasattr(self, "raw_llm_responses"):
                self.raw_llm_responses = {}
            self.raw_llm_responses[task_name] = {"error": f"Prompt template key '{llm_prompt_template_key}' not found."}
            return None

        prompt_template_str = prompt_template_data.get('prompt_template')
        if not prompt_template_str:
            self.logger.error(f"'prompt_template' field missing in prompt configuration for key '{llm_prompt_template_key}'.")
            if not hasattr(self, "raw_llm_responses"):
                self.raw_llm_responses = {}
            self.raw_llm_responses[task_name] = {"error": f"'prompt_template' field missing for prompt key '{llm_prompt_template_key}'."}
            return None

        # 1. Retrieve RAG context
        rag_query = " ".join(query_elements)
        k_chunks = prompt_template_data.get("rag_config", {}).get("k_retrieved_chunks", alpy_config.RAG_NUM_RETRIEVED_CHUNKS)
        max_len = prompt_template_data.get("rag_config", {}).get("max_context_length", alpy_config.RAG_MAX_CONTEXT_LENGTH)
        
        context_chunks = await self.rag_manager.retrieve_context_chunks(
            security_id=self.security_id,
            query=rag_query,
            k_retrieved_chunks=k_chunks
        )
        
        # Enhanced debug logging to inspect chunk content and OCR data
        if task_name.startswith('line_item_extraction_'):
            self.logger.info(f"Retrieved {len(context_chunks)} chunks for '{task_name}'")
            for i, chunk in enumerate(context_chunks):
                doc_name = chunk.metadata.get('document_name', 'unknown') if chunk.metadata else 'unknown'
                page = chunk.metadata.get('page_number', 'unknown') if chunk.metadata else 'unknown'
                source = chunk.metadata.get('source', 'unknown') if chunk.metadata else 'unknown'
                self.logger.info(f"  Chunk {i+1}/{len(context_chunks)} from '{doc_name}' page {page}, source: {source}")
                
                # Check if this chunk contains any numbers that might be financial data
                # This regex finds sequences that look like numbers (with commas, periods, spaces)
                financial_data = re.findall(r'\b\d[\d\s,.]*\d\b', chunk.text_content)
                if financial_data:
                    self.logger.info(f"    Found potential financial values: {financial_data[:10]}")
                
                # Show a preview with at most 250 chars
                chunk_preview = chunk.text_content.strip()[:250]
                self.logger.info(f"    Preview: {chunk_preview}...")
                
                # If this is OCR content, show more of it as it might contain tables
                if source == 'ocr_results':
                    ocr_content = chunk.text_content.strip()
                    if len(ocr_content) > 250:
                        self.logger.info(f"    Extended OCR content: {ocr_content[250:500]}...")
        
        rag_context = "\n\n---\n\n".join([chunk.text_content for chunk in context_chunks])
        if len(rag_context) > max_len:
            rag_context = rag_context[:max_len]
            self.logger.debug(f"Truncated RAG context for task {task_name} to {max_len} characters.")

        # 2. Format prompt
        prompt_format_vars = {"context": rag_context}
        if additional_prompt_format_vars:
            prompt_format_vars.update(additional_prompt_format_vars)
        
        try:
            final_prompt = prompt_template_str.format(**prompt_format_vars)
        except KeyError as e:
            self.logger.error(f"Missing key {e} in prompt_format_vars for task {task_name}. Template: '{prompt_template_str}', Vars: {prompt_format_vars}")
            if not hasattr(self, "raw_llm_responses"):
                self.raw_llm_responses = {}
            self.raw_llm_responses[task_name] = {"error": f"Prompt formatting error: Missing key {e}"}
            return None

        # Conditionally log the full prompt for Depreciation & Amortization to help debug hangs
        if "Depreciation & Amortization" in task_name:
            self.logger.warning(f"FULL PROMPT for {task_name}:\n{final_prompt}")

        # Skip logging the full prompt to reduce log size for other items
        self.logger.debug(f"Prepared prompt for task {task_name} (length: {len(final_prompt)} chars)")

        # 3. Call LLM
        llm_response_str = await self.llm_service.generate_text_response(final_prompt)
        # Store raw LLM response
        if not hasattr(self, "raw_llm_responses"):
            self.raw_llm_responses = {}
        self.raw_llm_responses[task_name] = {"response": llm_response_str}

        if not llm_response_str:
            self.logger.warning(f"LLM returned no response for task: {task_name}")
            return None

        # 4. Parse and validate response
        try:
            json_str_to_parse = llm_response_str
            if llm_response_str.strip().startswith("```json"):
                match = re.search(r"```json\s*([\s\S]*?)\s*```", llm_response_str, re.MULTILINE)
                if match:
                    json_str_to_parse = match.group(1).strip()
            elif llm_response_str.strip().startswith("```") and llm_response_str.strip().endswith("```"):
                 json_str_to_parse = llm_response_str.strip()[3:-3].strip()

            parsed_response = pydantic_model.model_validate_json(json_str_to_parse)
            self.logger.info(f"Successfully parsed LLM response for task: {task_name}")
            return parsed_response
        except json.JSONDecodeError as e:
            self.logger.error(f"JSONDecodeError parsing LLM response for {task_name}: {e}. Response: {json_str_to_parse[:500]}...", exc_info=True)
        except Exception as e:
            self.logger.error(f"Error parsing/validating LLM response for {task_name} into {pydantic_model.__name__}: {e}. Response: {json_str_to_parse[:500]}...", exc_info=True)
        
        return None

    async def extract_metadata(self, document_name: str, document_type_hint: Optional[str] = None) -> Optional[ExtractedMetadata]:
        """Extracts metadata from the document(s) associated with the security_id."""
        task_name = "metadata_extraction"
        query_elements = [
            "company name", "ticker symbol", "exchange", "reporting currency", 
            "fiscal year end date", "latest report date", "document type",
            f"document name: {document_name}"
        ]
        if document_type_hint:
            query_elements.append(f"document type hint: {document_type_hint}")

        class LlmOutputMetadata(BaseModel):
            target_company_name: Optional[str] = None
            ticker_symbol: Optional[str] = None
            currency: Optional[str] = None
            fiscal_year_end: Optional[str] = None

        llm_output: Optional[LlmOutputMetadata] = await self._run_llm_extraction_task(
            task_name=task_name,
            query_elements=query_elements,
            pydantic_model=LlmOutputMetadata,  # Use the temporary LLM output model
            llm_prompt_template_key="metadata_extraction",
            additional_prompt_format_vars={
                "document_name": document_name,
                "document_type": document_type_hint or "financial report"
            }
        )

        if llm_output:
            # Map from LlmOutputMetadata to the canonical ExtractedMetadata model
            mapped_metadata = ExtractedMetadata(
                company_name=llm_output.target_company_name,
                ticker_symbol=llm_output.ticker_symbol,
                reporting_currency=llm_output.currency,
                fiscal_year_end=llm_output.fiscal_year_end,
                document_type=document_type_hint or "financial report", # Preserve hint or default
                document_source_name=document_name # Add document name as source
            )
            self.logger.info(f"Successfully extracted and mapped metadata for {document_name}: {mapped_metadata.model_dump_json(indent=2)}")
            return mapped_metadata
        
        self.logger.warning(f"Failed to extract metadata for {document_name}.")
        return None

    # Helper methods for schema parsing
    def _get_schema_value(self, schema_node: Dict[str, Any], key_path: List[str], default: Any = None) -> Any:
        """Safely retrieves a nested value from a schema dictionary."""
        current = schema_node
        for key in key_path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            elif isinstance(current, list) and key.isdigit() and int(key) < len(current): # Check for list index access
                current = current[int(key)]
            else:
                return default
        return current

    def _is_line_item_schema(self, schema_node: Dict) -> bool:
        """Determines if a schema node represents a line item based on its properties."""
        self.logger.debug(f"_is_line_item_schema: Evaluating node: {str(schema_node)[:300]}...") # Log part of the node
        
        # Check for schema inheritance via allOf with reference to financial_line_item_base_schema
        if isinstance(schema_node, dict) and "allOf" in schema_node and isinstance(schema_node["allOf"], list):
            for item in schema_node["allOf"]:
                if isinstance(item, dict) and "$ref" in item:
                    ref_path = item["$ref"]
                    if "financial_line_item_base_schema" in ref_path:
                        self.logger.debug(f"  _is_line_item_schema: Found financial_line_item_base_schema reference: {ref_path}")
                        # If it has properties with name.const, it's definitely a line item
                        if "properties" in schema_node and "name" in schema_node["properties"]:
                            name_prop = schema_node["properties"]["name"]
                            if isinstance(name_prop, dict) and "const" in name_prop:
                                self.logger.debug(f"  _is_line_item_schema: Found line item via allOf with name: {name_prop['const']}")
                                return True
                        # Even without properties, it inherits from the base schema so it's a line item
                        self.logger.debug(f"  _is_line_item_schema: Recognized as line item via base schema inheritance")
                        return True
                        
        if not isinstance(schema_node, dict) or "properties" not in schema_node:
            self.logger.debug(f"  _is_line_item_schema: Node is not a dict or no 'properties' key. Returning False.")
            return False
        
        props = schema_node["properties"]
        self.logger.debug(f"  _is_line_item_schema: Node properties: {list(props.keys())}")

        # Check for 'name' with 'const'
        name_prop = props.get("name")
        has_name_const = isinstance(name_prop, dict) and "const" in name_prop and isinstance(name_prop["const"], str)
        self.logger.debug(f"  _is_line_item_schema: Check 'name.const' (string): {has_name_const}. Name prop: {str(name_prop)[:100]}")

        # Check for 'periods' array of objects
        periods_prop = props.get("periods")
        has_periods_array_of_objects = (
            isinstance(periods_prop, dict) and
            periods_prop.get("type") == "array" and
            isinstance(periods_prop.get("items"), dict) and
            periods_prop.get("items", {}).get("type") == "object"
        )
        self.logger.debug(f"  _is_line_item_schema: Check 'periods' (array of objects): {has_periods_array_of_objects}. Periods prop: {str(periods_prop)[:100]}")

        if not (has_name_const and has_periods_array_of_objects):
            self.logger.debug(f"  _is_line_item_schema: Basic checks (name/periods) failed. Returning False.")
            return False

        # Check if it's a container for other line items (more complex logic)
        # This checks if the 'items' of 'periods' itself has 'properties' (which it shouldn't for a simple line item's period structure)
        # OR if any *other* property (not the standard line item fields) is a complex object with its own 'properties'.
        
        # Check 1: Does the 'items' schema of 'periods' have its own 'properties' key?
        # This would mean each period object is itself a complex nested structure, not typical for a line item's period value.
        periods_items_schema = periods_prop.get("items", {})
        periods_items_has_properties = "properties" in periods_items_schema
        self.logger.debug(f"  _is_line_item_schema: Check 'periods.items' has 'properties': {periods_items_has_properties}. periods.items: {str(periods_items_schema)[:100]}")

        # Check 2: Do any *other* properties (besides the standard line item attributes) define complex objects?
        # These are the expected simple properties of a line item itself.
        standard_line_item_prop_keys = [
            "name", "name_mn", "data_type", "periods", "unit", "notes", 
            "is_calculated", "calculation_logic_description", 
            "source_guidance_historical", "ai_instructions", "ai_instructions_projected",
            "is_total_or_subtotal", "item_type" # Added item_type and is_total_or_subtotal as they are simple consts
        ]
        has_other_complex_properties = False
        for prop_key, prop_value in props.items():
            if prop_key not in standard_line_item_prop_keys:
                if isinstance(prop_value, dict) and "properties" in prop_value:
                    self.logger.debug(f"  _is_line_item_schema: Found other complex property '{prop_key}' with its own 'properties'. Marking as potential container.")
                    has_other_complex_properties = True
                    break
        self.logger.debug(f"  _is_line_item_schema: Check for other complex properties (not in standard list): {has_other_complex_properties}")

        is_container_type = periods_items_has_properties or has_other_complex_properties
        self.logger.debug(f"  _is_line_item_schema: Is container type (based on periods.items.properties or other complex props): {is_container_type}")

        final_decision = has_name_const and has_periods_array_of_objects and not is_container_type
        self.logger.debug(f"  _is_line_item_schema: Final decision: {final_decision}")
        return final_decision

    def _find_line_items_recursive(self, current_schema_node: Any, current_json_path_parts: List[str], statement_key: str, current_semantic_path_parts: List[str]) -> List[LineItemTarget]:
        """
        Recursively traverses the schema to find all line item definitions.
        current_json_path_parts: The parts of the JSON path from the statement_key downwards relative to the node being processed.
        current_semantic_path_parts: The parts of the user-friendly semantic path from the statement_key downwards.
        """
        targets = []
        current_path_str_for_logging = ".".join(current_json_path_parts)
        self.logger.debug(f"_find_line_items_recursive: Path='{current_path_str_for_logging}', Node Type='{type(current_schema_node)}', Statement='{statement_key}'")

        # Base for constructing the full schema path to the item's definition
        full_schema_path_prefix = f"financial_statements_core.{statement_key}."

        if isinstance(current_schema_node, dict):
            self.logger.debug(f"  Path='{current_path_str_for_logging}': Node is a dict. Keys: {list(current_schema_node.keys())}")
            is_item = self._is_line_item_schema(current_schema_node)
            self.logger.debug(f"  Path='{current_path_str_for_logging}': _is_line_item_schema returned: {is_item}")

            if is_item:
                props = current_schema_node.get("properties", {})
                line_item_name = self._get_schema_value(props, ["name", "const"])
                self.logger.debug(f"  Path='{current_path_str_for_logging}': Extracted line_item_name: '{line_item_name}'")
                if line_item_name:
                    # Ensure current_json_path_parts accurately reflects the path to *this specific item's schema definition*
                    # If current_json_path_parts was ['properties', 'line_items'] and we are now processing the item *within* that array,
                    # the path should reflect that. The logic in _load_schema_and_derive_targets passes ['items'] when recursing for array items.
                    schema_path_str = full_schema_path_prefix + ".".join(current_json_path_parts)
                    semantic_section_str = ".".join(current_semantic_path_parts[:-1]) if len(current_semantic_path_parts) > 1 else current_semantic_path_parts[0] if current_semantic_path_parts else ""
                    self.logger.info(f"    FOUND Line Item: '{line_item_name}' at schema_path: '{schema_path_str.rstrip('.')}', semantic_path: '{semantic_section_str}'")
                    targets.append(
                        LineItemTarget(
                            line_item_name=line_item_name,
                            line_item_name_mongolian=self._get_schema_value(props, ["name_mn", "const"]),
                            statement_key=statement_key,
                            section_path=semantic_section_str,
                            schema_path=schema_path_str.rstrip('.'), 
                            is_total_or_subtotal=self._get_schema_value(props, ["is_total_or_subtotal", "const"], False),
                            raw_schema_definition=current_schema_node
                        )
                    )
                else:
                    self.logger.debug(f"  Path='{current_path_str_for_logging}': Node was considered a line item by _is_line_item_schema, but 'name.const' was not found.")

            if "properties" in current_schema_node: # Object with properties
                self.logger.debug(f"  Path='{current_path_str_for_logging}': Recursing into 'properties': {list(current_schema_node['properties'].keys())}")
                for key, sub_node in current_schema_node["properties"].items():
                    targets.extend(self._find_line_items_recursive(sub_node, current_json_path_parts + ["properties", key], statement_key, current_semantic_path_parts + [key]))
            elif "items" in current_schema_node: # Array
                items_schema = current_schema_node["items"]
                self.logger.debug(f"  Path='{current_path_str_for_logging}': Node has 'items'. Type of items_schema: {type(items_schema)}")
                if isinstance(items_schema, dict): # Array of similar objects (e.g. line_items array where 'items' is the schema for one such object)
                    self.logger.debug(f"    Path='{current_path_str_for_logging}': Recursing into 'items' (which is a dict - schema for array elements)")
                    # Pass current_json_path_parts + ['items'] to indicate we are now defining the path to the item type within the array
                    targets.extend(self._find_line_items_recursive(items_schema, current_json_path_parts + ["items"], statement_key, current_semantic_path_parts))
                elif isinstance(items_schema, list): # Tuple-like array (array of different types of objects)
                    self.logger.debug(f"    Path='{current_path_str_for_logging}': Recursing into 'items' (which is a list - tuple-like array)")
                    for i, item_sub_schema in enumerate(items_schema):
                        targets.extend(self._find_line_items_recursive(item_sub_schema, current_json_path_parts + ["items", str(i)], statement_key, current_semantic_path_parts + [str(i)]))
                else:
                    self.logger.debug(f"    Path='{current_path_str_for_logging}': 'items' is neither dict nor list. Type: {type(items_schema)}. Skipping recursion for it.")
            else:
                # This case means it's a dict, but not a line item itself, and has no 'properties' or 'items'.
                # Could be a simple type definition like {"type": "string", "const": "foo"} or an empty object.
                if not is_item: # Avoid double logging if it was already processed as a line item.
                     self.logger.debug(f"  Path='{current_path_str_for_logging}': Node is a dict but not a line item, and has no 'properties' or 'items' to recurse. Keys: {list(current_schema_node.keys())}")
    
        elif isinstance(current_schema_node, list): # Direct list of schemas (e.g. oneOf, anyOf, or a direct list of items not under an 'items' key)
            self.logger.debug(f"  Path='{current_path_str_for_logging}': Node is a list. Recursing into its elements. Length: {len(current_schema_node)}")
            for i, item_sub_schema in enumerate(current_schema_node):
                targets.extend(self._find_line_items_recursive(item_sub_schema, current_json_path_parts + [str(i)], statement_key, current_semantic_path_parts))
        else:
            self.logger.debug(f"  Path='{current_path_str_for_logging}': Node is not dict or list (Type: {type(current_schema_node)}). Skipping.")
    
        if targets:
            self.logger.debug(f"_find_line_items_recursive: Path='{current_path_str_for_logging}', Statement='{statement_key}' - Returning {len(targets)} targets found at or below this level.")
        return targets

    def _load_schema_and_derive_targets(self, schema_path: Path) -> None:
        """Loads the financial model schema and dynamically derives all LineItemTarget objects."""
        self.line_item_targets: List[LineItemTarget] = []
        self.logger.debug(f"Attempting to load schema from: {schema_path}")
        if not schema_path.exists():
            self.logger.error(f"Financial model schema file not found at {schema_path}. Cannot load targets.")
            return

        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                self.parsed_financial_schema = json.load(f)
            self.logger.info(f"Successfully read and parsed JSON from {schema_path}")
            
            if not self.parsed_financial_schema or "properties" not in self.parsed_financial_schema or \
               "financial_statements_core" not in self.parsed_financial_schema["properties"]:
                self.logger.error("Schema is missing 'properties.financial_statements_core'. Cannot derive targets. Parsed schema keys: "
                                  f"{list(self.parsed_financial_schema.keys()) if self.parsed_financial_schema else 'None'}")
                return

            fs_core_schema = self.parsed_financial_schema["properties"]["financial_statements_core"]
            if "properties" not in fs_core_schema:
                self.logger.error("'financial_statements_core' is missing 'properties'. Cannot derive targets. "
                                  f"financial_statements_core keys: {list(fs_core_schema.keys())}")
                return

            all_found_targets: List[LineItemTarget] = []
            self.logger.debug(f"Iterating financial_statements_core properties: {list(fs_core_schema['properties'].keys())}")

            for statement_key, statement_schema_definition in fs_core_schema["properties"].items():
                self.logger.debug(f"Processing statement_key: '{statement_key}'")
                if not isinstance(statement_schema_definition, dict):
                    self.logger.debug(f"Skipping non-dict statement schema for '{statement_key}' (type: {type(statement_schema_definition)}).Value: {str(statement_schema_definition)[:200]}")
                    continue
                
                # Log what's inside statement_schema_definition (e.g. for 'income_statement')
                self.logger.debug(f"  '{statement_key}' schema definition keys: {list(statement_schema_definition.keys())}")

                if "properties" in statement_schema_definition: 
                    self.logger.debug(f"  '{statement_key}' has 'properties'. Iterating sections: {list(statement_schema_definition['properties'].keys())}")
                    for section_key, section_node in statement_schema_definition["properties"].items():
                        self.logger.debug(f"    Calling _find_line_items_recursive for section '{section_key}' in statement '{statement_key}'. Node type: {type(section_node)}")
                        found = self._find_line_items_recursive(section_node, ["properties", section_key], statement_key, [section_key])
                        if found: self.logger.debug(f"      Found {len(found)} targets in '{section_key}'.")
                        all_found_targets.extend(found)
                elif "items" in statement_schema_definition: # e.g. a direct array of line items under a statement (less common for top-level statements but possible)
                    self.logger.debug(f"  '{statement_key}' has 'items' but no 'properties'. Calling _find_line_items_recursive for its items. Node type: {type(statement_schema_definition['items'])}")
                    found = self._find_line_items_recursive(statement_schema_definition["items"], ["items"], statement_key, ["line_items"])
                    if found: self.logger.debug(f"      Found {len(found)} targets directly under items of '{statement_key}'.")
                    all_found_targets.extend(found)
                else:
                    self.logger.debug(f"  '{statement_key}' has neither 'properties' nor 'items'. Skipping recursive search for it. Keys: {list(statement_schema_definition.keys())}")

            unique_targets_map: Dict[str, LineItemTarget] = {}
            for target in all_found_targets:
                unique_key = target.schema_path 
                if unique_key not in unique_targets_map:
                    unique_targets_map[unique_key] = target
                else:
                    self.logger.debug(f"Duplicate target found for schema_path '{unique_key}'. Keeping first one found: {target.line_item_name}")

            self.line_item_targets = list(unique_targets_map.values())
            
            # Log the final list of line item targets
            if self.line_item_targets:
                self.logger.info(f"Derived {len(self.line_item_targets)} line item targets from schema.")
                # Add a compact summary of found line items by statement
                statement_counts = {}
                for target in self.line_item_targets:
                    statement_counts[target.statement_key] = statement_counts.get(target.statement_key, 0) + 1
                
                for statement, count in statement_counts.items():
                    self.logger.info(f"  • {statement}: {count} line items")
                    
                # Log first 10 items as a sample
                sample_size = min(10, len(self.line_item_targets))
                self.logger.info(f"Sample of line items that will be extracted:")
                for i in range(sample_size):
                    self.logger.info(f"  {i+1}. {self.line_item_targets[i].line_item_name} ({self.line_item_targets[i].statement_key})")
            else:
                self.logger.warning("No line item targets derived from schema. Extraction will yield no results.")

        except json.JSONDecodeError:
            self.logger.error(f"Failed to decode JSON from financial model schema at {schema_path}.", exc_info=True)
            self.parsed_financial_schema = None 
        except Exception as e:
            self.logger.error(f"Unexpected error loading or parsing financial model schema from {schema_path}: {e}", exc_info=True)
            self.parsed_financial_schema = None

    async def extract_line_items(self, document_name: str) -> None:
        """Extracts line item data (defined in self.line_item_targets) for all historical periods."""
        if not self.model_data["financial_statements_core"]["historical_period_labels"]:
            self.logger.warning(f"Skipping line item extraction for {document_name} as no historical periods are available.")
            return

        if not self.line_item_targets:
            self.logger.info(f"No target line items derived from schema for {document_name}. Skipping extraction.")
            return
            
        # To avoid excessive LLM calls, sample a subset of high-value line items for testing
        # In production, this can be set to use all items by setting a higher sample_size
        sample_size = len(self.line_item_targets)  # Process all available line items
        
        # Process targets up to the sample_size
        # (sample_size is currently set to len(self.line_item_targets) to process all items)
        sampled_targets = self.line_item_targets[:sample_size]
        
        self.logger.info(f"Starting line item extraction for {document_name} for {len(sampled_targets)} high-priority items across {len(self.model_data['financial_statements_core']['historical_period_labels'])} periods.")
        
        extracted_count = 0
        for target in sampled_targets: 
            item_name_english = target.line_item_name
            line_item_mongolian_name = target.line_item_name_mongolian if target.line_item_name_mongolian else f"{item_name_english} (MN)"
            line_item_definition = target.raw_schema_definition.get("description", f"Standard definition for {item_name_english}")

            for period in self.model_data["financial_statements_core"]["historical_period_labels"]:
                task_name = f"line_item_extraction_{item_name_english}_{period}"
                # Use all available schema information without hardcoding
                statement_type = target.statement_key.replace('_', ' ')
                
                # Extract schema definitions to inform the query
                section_path = target.section_path
                definition = target.raw_schema_definition.get("description", "")
                
                # Build a focused query using schema data only
                query_elements = [
                    # Include both languages - critical for matching OCR content
                    item_name_english, 
                    line_item_mongolian_name,
                    
                    # Add period and statement information
                    period,
                    statement_type,
                    
                    # Add section path for context (more specific than statement type)
                    section_path,
                    
                    # Include document name
                    document_name
                ]
                llm_output: Optional[SingleLineItemLLMOutput] = await self._run_llm_extraction_task(
                    task_name=task_name,
                    query_elements=query_elements,
                    pydantic_model=SingleLineItemLLMOutput,
                    llm_prompt_template_key="single_line_item_extraction",
                    additional_prompt_format_vars={
                        "document_name": document_name,
                        "line_item_english_name": item_name_english,
                        "period_label": period,
                        "line_item_mongolian_name": line_item_mongolian_name,
                        "line_item_definition": line_item_definition,
                    }
                )

                if llm_output and llm_output.status == "EXTRACTED_SUCCESSFULLY":
                    data_unit = llm_output.currency if llm_output.currency else llm_output.unit
                
                    line_item_data = LineItemData(
                        item_name_english=item_name_english,
                        item_name_mongolian=line_item_mongolian_name,
                        period_id=period,
                        value=llm_output.value,
                        unit=data_unit,
                        notes=llm_output.source_reference,
                        extraction_status=llm_output.status
                    )
                    # Add line item to appropriate statement
                    statement = "income_statement" if "income" in target.statement_key.lower() else "balance_sheet"
                    if "cash_flow" in target.statement_key.lower():
                        statement = "cash_flow_statement"
                    self.model_data["financial_statements_core"][statement]["line_items"].append(line_item_data.model_dump())
                    extracted_count += 1
                    self.logger.info(f"✓ DATA POINT: '{item_name_english}' [{period}] = {llm_output.value} {data_unit or ''})")
                elif llm_output:
                    self.logger.warning(f"✗ FAILED: '{item_name_english}' [{period}] - Status: {llm_output.status}")
                else:
                    self.logger.error(f"✗ ERROR: '{item_name_english}' [{period}] - LLM returned no response")

        self.logger.info(f"✅ EXTRACTION SUMMARY: {extracted_count}/{len(self.line_item_targets) * len(self.model_data['financial_statements_core']['historical_period_labels'])} data points successfully extracted from {document_name}")

    def _create_line_item_output(self, schema_item_properties: Dict,
                                 explicit_english_name: Optional[str], # ADDED
                                 historical_period_labels: List[str],
                                 all_extracted_data: List[LineItemData]) -> Dict[str, Any]:
        """Helper to create a single line item's output structure based on schema and extracted data."""
        english_name = schema_item_properties.get("name", {}).get("const")
        if not english_name: # If name.const is not found in schema_item_properties
            english_name = explicit_english_name # Use the explicit name passed from parent
            if english_name:
                self.logger.debug(f"TRANSFORM_TRACE: Used explicit_english_name '{english_name}' for item as 'name.const' was missing. Schema props: {str(schema_item_properties)[:200]}")
            # else: english_name remains None here if explicit_english_name was also None
        
        if not english_name: # If still no name (neither const nor explicit found)
            self.logger.warning(f"Schema item definition missing 'name.const' and no explicit_english_name provided. Schema props: {str(schema_item_properties)[:200]}, Explicit: {explicit_english_name}")
            return {}

        line_item_output = {
            "name": english_name,
            "name_mn": schema_item_properties.get("name_mn", {}).get("const"),
            "data_type": schema_item_properties.get("data_type", {}).get("const", "currency_value"),
            "periods": [],
            "is_calculated": schema_item_properties.get("is_calculated", {}).get("const", False),
            "calculation_logic_description": schema_item_properties.get("calculation_logic_description", {}).get("const"),
            "source_guidance_historical": schema_item_properties.get("source_guidance_historical", {}).get("const"),
            "ai_instructions": schema_item_properties.get("ai_instructions", {}).get("const"),
            "ai_instructions_projected": schema_item_properties.get("ai_instructions_projected", {}).get("const"),
            "notes": None, 
            "unit": None   
        }

        item_units = set()
        first_note_for_item = None

        for hp_label in historical_period_labels:
            extracted_data_for_period = next((
                lid for lid in all_extracted_data
                if lid.item_name_english == english_name and lid.period_id == hp_label
            ), None)

            period_data_obj = {
                "period_label": hp_label,
                "value": None,
                "source_reference": None, 
                "extraction_status": "NOT_FOUND_IN_EXTRACTED_DATA"
            }

            if extracted_data_for_period:
                period_data_obj["value"] = extracted_data_for_period.value
                period_data_obj["source_reference"] = extracted_data_for_period.notes
                period_data_obj["extraction_status"] = extracted_data_for_period.extraction_status or "UNKNOWN_STATUS_FROM_EXTRACTION"
                if extracted_data_for_period.unit:
                    item_units.add(extracted_data_for_period.unit)
                if extracted_data_for_period.notes and not first_note_for_item:
                    first_note_for_item = extracted_data_for_period.notes
            
            line_item_output["periods"].append(period_data_obj)

        if item_units:
            if len(item_units) == 1:
                line_item_output["unit"] = item_units.pop()
            else:
                self.logger.warning(f"Inconsistent units for line item '{english_name}': {item_units}. Setting unit to None.")
        
        if first_note_for_item:
            line_item_output["notes"] = first_note_for_item
            
        return line_item_output

    def _recursively_build_output_node(self, current_schema_node: Dict[str, Any],
                                       historical_period_labels: List[str],
                                       model_line_items_data: List[LineItemData],
                                       current_item_key: Optional[str] = None) -> Any: # ADDED current_item_key
        """
        Recursively traverses a part of the parsed financial schema and builds the
        corresponding output structure, populating line items with data.
        """
        if not isinstance(current_schema_node, dict):
            if isinstance(current_schema_node, list):
                output_list = []
                for item_schema in current_schema_node:
                    built_item = self._recursively_build_output_node(item_schema, historical_period_labels, model_line_items_data, current_item_key=None)
                    if built_item is not None:
                        output_list.append(built_item)
                return output_list if output_list else None # Return None if list is empty to prune empty structures
            self.logger.debug(f"Expected dict for current_schema_node, got {type(current_schema_node)}. Value: {str(current_schema_node)[:100]}")
            return None

        node_type = current_schema_node.get("type")
        node_title = current_schema_node.get("title", "Unknown Node") # For logging

        # Case 1: It's a line item schema
        if self._is_line_item_schema(current_schema_node):
            line_item_schema_props = current_schema_node.get("properties")
            if line_item_schema_props:
                # This is a line item, create its structure
                current_line_item_name_en = current_schema_node.get("properties", {}).get("name", {}).get("const")
                self.logger.debug(f"TRANSFORM_TRACE: Processing line item: {current_line_item_name_en}")

                # Temporarily log the data we expect to find for this line item
                relevant_model_data_for_item = []
                if current_line_item_name_en:
                    self.logger.debug(f"TRANSFORM_TRACE_DEBUG: Comparing with schema name: {repr(current_line_item_name_en)} (len: {len(current_line_item_name_en) if current_line_item_name_en else 'N/A'})")
                    for i, lid in enumerate(model_line_items_data):
                        self.logger.debug(f"TRANSFORM_TRACE_DEBUG: Checking model_line_items_data[{i}].item_name_english: {repr(lid.item_name_english)} (len: {len(lid.item_name_english) if lid.item_name_english else 'N/A'})")
                        comparison_result = lid.item_name_english == current_line_item_name_en
                        self.logger.debug(f"TRANSFORM_TRACE_DEBUG: Comparison result: {comparison_result}")
                        if comparison_result:
                            relevant_model_data_for_item.append(lid.model_dump_json(indent=2))
                self.logger.debug(f"TRANSFORM_TRACE: Raw model_line_items_data for '{current_line_item_name_en}':\n{json.dumps(relevant_model_data_for_item, indent=2) if relevant_model_data_for_item else '[]'}")

                output_node = self._create_line_item_output(
                    schema_item_properties=current_schema_node.get("properties", {}),
                    explicit_english_name=current_item_key, # PASS current_item_key
                    historical_period_labels=historical_period_labels,
                    all_extracted_data=model_line_items_data
                )
                # Log what _create_line_item_output produced for the periods
                if output_node and 'periods' in output_node and isinstance(output_node['periods'], list):
                    for period_data_out in output_node['periods']:
                        self.logger.debug(f"TRANSFORM_TRACE: Output for '{current_line_item_name_en}' Period '{period_data_out.get('period_label')}': Value='{period_data_out.get('value')}', Status='{period_data_out.get('extraction_status')}'")
            else:
                self.logger.warning(f"Line item schema '{node_title}' missing 'properties'.")
                return None

        # Case 2: It's an object with further properties (a section or sub-section)
        elif node_type == "object" and "properties" in current_schema_node:
            # self.logger.debug(f"Processing object node: {node_title}")
            output_object = {}
            # Copy descriptive/meta properties from schema to output
            for key, value in current_schema_node.items():
                if key not in ["properties", "items", "type", "$schema", "title", "required", "allOf", "anyOf", "oneOf", "not", "$id", "$ref", "definitions"]:
                    output_object[key] = value
            
            has_content = False
            for prop_key, prop_schema_node in current_schema_node["properties"].items():
                # self.logger.debug(f"Recursively building property: {prop_key} within {node_title}")
                built_node = self._recursively_build_output_node(
                    prop_schema_node, historical_period_labels, model_line_items_data, current_item_key=prop_key
                )
                if built_node is not None: 
                    output_object[prop_key] = built_node
                    has_content = True
            
            # Return object only if it has actual data content or is a pre-defined non-data carrying section (e.g. empty obj with description)
            # For now, if it has content from recursion or was meant to have descriptive props, keep it.
            if has_content or any(k not in ["properties", "items", "type"] for k in output_object.keys()):
                return output_object
            else:
                # self.logger.debug(f"Pruning empty object node: {node_title}")
                return None

        # Case 3: It's an array (typically of line items or other objects)
        elif node_type == "array" and "items" in current_schema_node:
            # self.logger.debug(f"Processing array node: {node_title}")
            output_array = []
            items_schema_definition = current_schema_node["items"]

            if isinstance(items_schema_definition, list): # Tuple-like array
                for single_item_schema in items_schema_definition:
                    item_output = self._recursively_build_output_node(
                        single_item_schema, historical_period_labels, model_line_items_data
                    )
                    if item_output is not None:
                         output_array.append(item_output)
            elif isinstance(items_schema_definition, dict): # Array of similar items
                # This case is complex. If 'items' is a single schema, it means an array of items of THAT type.
                # If that type is a line item, it's ambiguous how to populate multiple distinct instances
                # unless the data source itself provides an array for that specific line item name (which is not current model).
                # For now, we'll assume if it's an array of line items, they are explicitly listed or it's a placeholder.
                # self.logger.debug(f"Array node '{node_title}' has a single dict for 'items'. This usually means a list of similar items.")
                # If the items_schema_definition is a line item itself, it's likely a placeholder in the schema for where line items go.
                # We don't try to build it directly here as _create_line_item_output expects one specific item.
                # This structure is typically handled if 'properties' points to an array of specific line item schemas.
                pass # output_array remains empty, expecting specific line items to be defined if data is to be populated.
            
            return output_array if output_array else None # Prune empty arrays

        # Case 4: Direct value from schema (e.g., 'const', 'default' for simple types)
        # This is for schema-defined fixed values, not typically for financial data points.
        if "const" in current_schema_node:
            return current_schema_node["const"]
        if "default" in current_schema_node and isinstance(current_schema_node["default"], (str, int, float, bool, type(None))):
             return current_schema_node["default"]

        # self.logger.debug(f"Schema node (Title: '{node_title}', Type: '{node_type}') did not match specific processing rules or resulted in no data. Returning None.")
        return None

    async def run_phase_a_extraction(
        self,
        parsed_documents: List[Dict[str, Any]],
        document_names: List[str],
        primary_document_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Runs the full Phase A extraction pipeline and returns a JSON-compliant dictionary."""
        self.logger.info(f"Starting Phase A extraction for security ID: {self.security_id}")
        self.model_data = self._create_initial_model(self.security_id)

        if not document_names or not parsed_documents:
            self.logger.error("Error: No document names or parsed_documents provided for extraction.")
            self.model_data["model_metadata"].update({
                "status_overall_model_generation": "FAILED_PREPARATION",
                "message_overall_model_generation": "Error: No document names or parsed_documents provided."
            })
            return self.model_data

        effective_primary_doc_name = primary_document_name or document_names[0]
        self.model_data["model_metadata"]["primary_document_name"] = effective_primary_doc_name

        try:
            rag_success = await self.process_documents_for_rag(parsed_documents, document_names)
            message = ""
            if not rag_success:
                self.logger.warning("RAG processing did not complete successfully or yielded no chunks. Extraction quality may be affected.")
                message = "Warning: RAG processing issues. Extraction may be incomplete."
            else:
                message = "RAG processing completed."

            metadata = await self.extract_metadata(document_name=effective_primary_doc_name)
            if metadata:
                self._update_model_metadata(metadata)
            else:
                message += " Metadata extraction failed."

            # Generate historical periods
            current_real_year = datetime.datetime.now(datetime.timezone.utc).year
            historical_period_labels = []
            for i in range(5):
                year = current_real_year - 1 - i  # For 2025: 2024, 2023, 2022, 2021, 2020
                historical_period_labels.append(f"FY{year}")
            
            # Sort periods from oldest to newest
            historical_period_labels.sort()
            self.model_data["financial_statements_core"]["historical_period_labels"] = historical_period_labels
            
            # Extract line items
            line_items = await self.extract_line_items(document_name=effective_primary_doc_name)
            if not line_items:
                message += " Line item extraction completed but yielded no data points."
            else:
                for item in line_items:
                    self._add_line_item(item)
                message += " Line item extraction completed."

            # Update final status
            self.model_data["model_metadata"].update({
                "status_overall_model_generation": "COMPLETED_EXTRACTION",
                "message_overall_model_generation": message.strip()
            })
            
            self.logger.info(f"Phase A extraction completed for {self.security_id}. Status: COMPLETED_EXTRACTION")
            return self.model_data

        except Exception as e:
            self.logger.error(f"Critical error during Phase A extraction for {self.security_id}: {e}", exc_info=True)
            self.model_data["model_metadata"].update({
                "status_overall_model_generation": "FAILED_EXTRACTION",
                "message_overall_model_generation": f"Critical error during extraction: {str(e)}"
            })
            return self.model_data



async def _test_orchestrator_pipeline():
    """Stand-alone test function for PhaseAOrchestrator."""
    print("Starting PhaseAOrchestrator test pipeline...")
    print("IMPORTANT: Ensure GOOGLE_API_KEY environment variable is set.")
    print(f"IMPORTANT: Ensure RAG_VECTOR_STORE_BASE_PATH ('{alpy_config.RAG_VECTOR_STORE_BASE_PATH}') is writable and exists.")
    print(f"IMPORTANT: Ensure prompts YAML file exists at specified path.")

    # --- Configuration for the test ---
    test_security_id = "TEST_REAL_DOC_001"
    # Path to your actual prompts YAML file. This should match the one used in __init__.
    prompts_file = Path(__file__).resolve().parent.parent.parent / "prompts" / "financial_modeling_prompts.yaml"
    # schema_file = Path(__file__).resolve().parent.parent.parent / "fund" / "financial_model_schema.json"
    # Use the new mock schema for simplified testing:
    schema_file = Path(__file__).resolve().parent / "mock_financial_schema.json" # Points to src/financial_modeling/mock_financial_schema.json
    output_log_file = Path(f"./logs/test_orchestrator_output_{test_security_id}.log")

    # Path to the mock parsed document JSON file
    # The mock file is in the project root (Alpy/) relative to this script's parent's parent.
    # current_script_dir is defined at the start of this function
    # Define the directory containing the real parsed documents
    parsed_doc_dir = Path("/home/me/CascadeProjects/Alpy/test_outputs/parsed_doc_cache") # Your target directory

    if not parsed_doc_dir.exists() or not parsed_doc_dir.is_dir():
        print(f"CRITICAL ERROR: Parsed documents directory {parsed_doc_dir} not found or is not a directory. Test cannot run.", file=sys.stderr)
        return

    # Get all .json files from the specified directory
    json_files = list(parsed_doc_dir.glob("*.json"))

    if not json_files:
        print(f"CRITICAL ERROR: No JSON files found in {parsed_doc_dir}", file=sys.stderr)
        return

    print(f"INFO: Found {len(json_files)} JSON files in {parsed_doc_dir}")
    # Sort files to ensure consistent primary_doc_name selection if needed, e.g., alphabetically
    json_files.sort()
    for json_file in json_files:
        print(f"  - {json_file.name}")
    
    # Target line items will now be derived from the schema by the orchestrator.
    # --- End Configuration ---

    if not prompts_file.exists():
        print(f"CRITICAL ERROR: Prompts file not found at {prompts_file}. Test cannot run.", file=sys.stderr)
        return

    if not schema_file.exists():
        print(f"CRITICAL ERROR: Financial model schema file not found at {schema_file}. Test cannot run.", file=sys.stderr)
        return

    # Ensure RAG base path exists and is writable
    rag_base_path = Path(alpy_config.RAG_VECTOR_STORE_BASE_PATH)
    try:
        if not rag_base_path.exists():
            print(f"INFO: RAG vector store base path {rag_base_path} does not exist. Creating it.")
            rag_base_path.mkdir(parents=True, exist_ok=True)
        if not os.access(rag_base_path, os.W_OK):
            print(f"CRITICAL ERROR: RAG vector store base path {rag_base_path} is not writable. Test cannot run.", file=sys.stderr)
            return
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to prepare RAG base path {rag_base_path}: {e}. Test cannot run.", file=sys.stderr)
        return

    # Delete existing vector store for the test_security_id to ensure a clean run
    vector_store_to_delete = rag_base_path / test_security_id
    if vector_store_to_delete.exists():
        print(f"INFO: Deleting existing vector store for {test_security_id} at {vector_store_to_delete}")
        try:
            shutil.rmtree(vector_store_to_delete)
            print(f"INFO: Successfully deleted {vector_store_to_delete}")
        except Exception as e:
            print(f"WARNING: Failed to delete {vector_store_to_delete}: {e}", file=sys.stderr)
            # Decide if this should be a critical error or just a warning
    else:
        print(f"INFO: No existing vector store found for {test_security_id} at {vector_store_to_delete}. No deletion needed.")

    # Load all parsed documents data
    try:
        test_parsed_docs = []
        test_doc_names = []
        primary_doc_name = None
        
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                loaded_json_content = json.load(f)
                test_parsed_docs.append(loaded_json_content)
                
                doc_name_from_metadata = loaded_json_content.get("metadata", {}).get("source_filename")
                if not doc_name_from_metadata:
                    doc_name_from_metadata = loaded_json_content.get("document_name")
                doc_name = doc_name_from_metadata if doc_name_from_metadata else json_file.name.replace(".pdf.json",".pdf")
                test_doc_names.append(doc_name)
                
                # Use the first document as the primary document
                if primary_doc_name is None:
                    primary_doc_name = doc_name
                    
                print(f"INFO: Successfully loaded parsed document: {doc_name} from {json_file}")
        
        print(f"INFO: Total of {len(test_parsed_docs)} documents loaded for extraction.")
        print(f"INFO: Primary document will be: {primary_doc_name}")

    except json.JSONDecodeError as e:
        print(f"CRITICAL ERROR: Failed to parse JSON from one of the files: {e}. Test cannot run.", file=sys.stderr)
        return
    except KeyError as e:
        print(f"CRITICAL ERROR: Missing expected key {e} in parsed document JSON. Test cannot run.", file=sys.stderr)
        return
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load or process JSON files: {e}. Test cannot run.", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return

    orchestrator = None
    try:
        orchestrator = PhaseAOrchestrator(
            security_id=test_security_id,
            prompts_yaml_path=prompts_file,
            financial_schema_path=schema_file,
            output_log_path=output_log_file
        )
        print(f"INFO: PhaseAOrchestrator initialized successfully. Logging to: {output_log_file}")
    except Exception as e:
        print(f"CRITICAL ERROR: Error initializing PhaseAOrchestrator: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return

    if orchestrator:
        print(f"INFO: Running Phase A extraction for security ID '{test_security_id}' with document '{primary_doc_name}'...")
        try:
            result_dict = await orchestrator.run_phase_a_extraction(
                parsed_documents=test_parsed_docs,
                document_names=test_doc_names,
                primary_document_name=primary_doc_name
            )
            print("\n--- Extraction Result (JSON Output) ---")
            print(json.dumps(result_dict, indent=2, ensure_ascii=False, default=str))
            print("\n--- End Extraction Result ---")

            result_output_file = Path(f"./test_extraction_result_{test_security_id}.json")
            with open(result_output_file, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False, default=str)
            print(f"INFO: Extraction result saved to {result_output_file}")

        except Exception as e:
            print(f"CRITICAL ERROR: Error during Phase A extraction: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            if orchestrator and orchestrator.model_data:
                print("\nExtracted Model:")
                if orchestrator.model_data:
                    print(json.dumps(orchestrator.model_data, indent=2, ensure_ascii=False, default=str))
                print("\n--- End Partial State ---")

    print("\nPhaseAOrchestrator test pipeline finished.")


if __name__ == "__main__":
    # This allows running the test directly using: python -m src.financial_modeling.phase_a_orchestrator
    # Make sure PYTHONPATH includes your project root (e.g., Alpy/)
    # or run from the Alpy directory: python src/financial_modeling/phase_a_orchestrator.py
    
    # Setup basic logging for the test run if not configured elsewhere
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("Executing PhaseAOrchestrator test from __main__...")
    asyncio.run(_test_orchestrator_pipeline())
