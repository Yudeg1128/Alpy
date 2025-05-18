import asyncio
import json
import logging
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Literal, Union

from pydantic import BaseModel, ValidationError

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold # For safety settings
from .. import config
from langchain_core.tools import ToolException

# --- LLM Service Placeholder ---
class BaseLlmService:
    async def invoke(
        self, prompt: str, model_name: Optional[str] = "gemini-2.0-flash",
        temperature: float = 0.1, max_output_tokens: int = 2048,
        json_mode: bool = True, **kwargs
    ) -> str:
        raise NotImplementedError("LLM Service invoke method not implemented.")

# --- Financial Utilities Placeholder ---
class FinancialModelingUtils:
    @staticmethod
    def normalize_value(value_str: Optional[Any], unit_str: Optional[str], currency_str: Optional[str]) -> Tuple[Optional[float], Optional[str], Optional[str]]:
        if value_str is None or str(value_str).strip().lower() in ['null', 'n/a', '']:
            return None, currency_str, unit_str
        try:
            cleaned_value_str = str(value_str).replace(',', '').strip()
            if not cleaned_value_str: return None, currency_str, unit_str
            val = float(cleaned_value_str)
            multiplier = 1.0
            if unit_str:
                unit_lower = unit_str.lower()
                if 'сая' in unit_lower or 'million' in unit_lower: multiplier = 1_000_000
                elif 'мян' in unit_lower or 'thousand' in unit_lower: multiplier = 1_000
            return val * multiplier, currency_str, unit_str
        except (ValueError, TypeError) as e:
            logging.warning(f"Could not normalize value: '{value_str}' (type: {type(value_str)}) unit: '{unit_str}'. Error: {e}")
            return None, currency_str, unit_str

# --- Pydantic Models for LLM Response Validation ---
class ExtractedMetadata(BaseModel):
    target_company_name: Optional[str] = None
    ticker_symbol: Optional[str] = None
    currency: Optional[str] = None
    fiscal_year_end: Optional[str] = None

class IdentifiedPeriods(BaseModel):
    historical_period_labels: List[str] = []

class ExtractedLineItemDetail(BaseModel):
    value: Optional[Any] = None
    currency: Optional[str] = None
    unit: Optional[str] = None
    source_reference: Optional[str] = None
    status: Literal["EXTRACTED_SUCCESSFULLY", "CONFIRMED_NOT_FOUND"]

class PhaseAOrchestrator:
    def __init__(
        self, schema_definition_path: Path, prompts_path: Path,
        llm_service: BaseLlmService, financial_utils: FinancialModelingUtils,
        output_log_path: Path = Path("logs/financial_modeling_orchestrator.log")
    ):
        self.schema_definition_path = schema_definition_path
        self.prompts_path = prompts_path
        self.llm_service = llm_service
        self.utils = financial_utils
        self.financial_model: Dict[str, Any] = {}
        self.extraction_log: List[Dict[str, Any]] = []
        self.logger = self._setup_logger(output_log_path)

        try:
            with open(self.schema_definition_path, 'r', encoding='utf-8') as f:
                self.model_schema_definition = json.load(f)
            self.logger.info(f"Loaded schema definition: {self.schema_definition_path}")
        except Exception as e: self.logger.error(f"Failed to load schema definition: {e}", exc_info=True); raise
        
        try:
            with open(self.prompts_path, 'r', encoding='utf-8') as f:
                self.prompts = yaml.safe_load(f)
            self.logger.info(f"Loaded prompts: {self.prompts_path}")
        except Exception as e: self.logger.error(f"Failed to load prompts: {e}", exc_info=True); raise
        
        self._initialize_model_from_schema_recursive(self.model_schema_definition, self.financial_model)
        self.logger.info("Financial model structure initialized from schema.")

    def _setup_logger(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger_instance = logging.getLogger(f"{__name__}.PhaseAOrchestrator.{id(self)}")
        log_level_str = config.LOG_LEVEL
        numeric_level = getattr(logging, log_level_str.upper(), logging.INFO)
        logger_instance.setLevel(numeric_level)
        
        if not logger_instance.handlers:
            fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(name)s - %(module)s:%(lineno)d - %(message)s')
            fh.setFormatter(formatter)
            logger_instance.addHandler(fh)
        return logger_instance

    def _resolve_ref(self, ref_path: str) -> Dict[str, Any]:
        if not ref_path.startswith("#/"): raise ValueError(f"Unsupported $ref format: {ref_path}")
        parts = ref_path[2:].split('/')
        current = self.model_schema_definition
        for part in parts:
            if isinstance(current, dict) and part in current: current = current[part]
            else: raise ValueError(f"$ref path not found: {ref_path} (part: {part})")
        return current

    def _get_effective_item_schema(self, item_schema_part: Dict[str, Any]) -> Dict[str, Any]:
        if "$ref" in item_schema_part:
            base_schema = self._resolve_ref(item_schema_part["$ref"])
            effective_schema = {**base_schema, **{k:v for k,v in item_schema_part.items() if k != "$ref"}}
            return self._get_effective_item_schema(effective_schema) 

        if "allOf" in item_schema_part:
            merged_properties = {}
            for sub_schema_wrapper in item_schema_part["allOf"]:
                resolved_sub_schema = self._get_effective_item_schema(sub_schema_wrapper)
                if "properties" in resolved_sub_schema:
                    merged_properties.update(resolved_sub_schema["properties"])
            
            if "properties" in item_schema_part:
                merged_properties.update(item_schema_part["properties"])
            
            final_schema = {k: v for k, v in item_schema_part.items() if k not in ["allOf", "properties"]}
            final_schema["properties"] = merged_properties
            if "type" not in final_schema: final_schema["type"] = "object"
            return final_schema
        
        return item_schema_part

    def _get_schema_property(self, path_keys: List[str]) -> Dict[str, Any]:
        current_schema_level = self.model_schema_definition
        if not path_keys: return current_schema_level.get("properties", {})
        if "properties" in current_schema_level: current_schema_level = current_schema_level["properties"]
        else: self.logger.error("Root 'properties' key not found in schema definition."); return {}

        for i, key in enumerate(path_keys):
            current_schema_level = self._get_effective_item_schema(current_schema_level) # Resolve at current node before key access
            if isinstance(current_schema_level, dict) and key in current_schema_level:
                current_schema_level = current_schema_level[key]
            else:
                self.logger.warning(f"Schema path key '{key}' not found. Path: {path_keys}.")
                return {}
            if i < len(path_keys) - 1: # If not the last key
                current_schema_level = self._get_effective_item_schema(current_schema_level) # Resolve before descending
                if isinstance(current_schema_level, dict) and "properties" in current_schema_level:
                    current_schema_level = current_schema_level["properties"]
                elif isinstance(current_schema_level, dict) and current_schema_level.get("type") == "array" and "items" in current_schema_level:
                    # If we need to navigate into an array's item schema (e.g. for tuple validation access)
                    # This branch might need refinement based on how path_keys refers to array items
                    pass # current_schema_level is the array schema; next key might be an index or sub-property of items.
                else:
                    self.logger.warning(f"Cannot descend into properties for key '{key}' in path {path_keys}.")
                    return {}
        
        current_schema_level = self._get_effective_item_schema(current_schema_level) # Resolve final target
        if isinstance(current_schema_level, dict):
            if current_schema_level.get("type") == "object" and "properties" in current_schema_level:
                return current_schema_level["properties"]
            return current_schema_level
        return {}

    def _initialize_model_from_schema_recursive(self, schema_part: Dict[str, Any], data_part: Union[Dict, List]):
        schema_part_effective = self._get_effective_item_schema(schema_part)
        schema_type = schema_part_effective.get("type")

        if schema_type == "object":
            if not isinstance(data_part, dict): self.logger.error(f"Type mismatch: Expected dict, got {type(data_part)} for schema {schema_part_effective.get('title', 'N/A')}"); return
            for prop_name, prop_schema_orig in schema_part_effective.get("properties", {}).items():
                prop_schema = self._get_effective_item_schema(prop_schema_orig)
                default_value = prop_schema.get("default")
                prop_type = prop_schema.get("type")

                if prop_type == "object":
                    data_part[prop_name] = {}
                    self._initialize_model_from_schema_recursive(prop_schema, data_part[prop_name])
                elif prop_type == "array":
                    data_part[prop_name] = []
                    items_schema = prop_schema.get("items")
                    if isinstance(items_schema, list): # Tuple validation
                        for sub_item_schema_orig in items_schema:
                            sub_item_schema = self._get_effective_item_schema(sub_item_schema_orig)
                            if sub_item_schema.get("type") == "object":
                                new_obj = {}
                                self._initialize_model_from_schema_recursive(sub_item_schema, new_obj) # Recursive call to populate based on sub-item's schema
                                data_part[prop_name].append(new_obj)
                            # Add handling for other types if arrays can contain non-objects
                    elif default_value is not None: data_part[prop_name] = list(default_value)
                else: # Primitives (or properties with 'const')
                    if "const" in prop_schema:
                        data_part[prop_name] = prop_schema["const"]
                    elif default_value is None and prop_type == "string":
                        data_part[prop_name] = ""
                    elif default_value is None and prop_type == "number":
                        data_part[prop_name] = 0 # Or None, depends on schema nullability
                    elif default_value is None and prop_type == "boolean":
                        data_part[prop_name] = False # Or None
                    else:
                        data_part[prop_name] = default_value
        elif schema_type == "array":
            if not isinstance(data_part, list): return
            # Handle initialization of top-level arrays if needed, though model root is object.

    def _find_financial_statement_anchor_pages(self, parsed_doc_content: Dict[str, Any], max_pages_to_check: Optional[int] = None) -> List[int]:
        anchor_keywords_primary = {
            "financial statements", "санхүүгийн тайлан", "income statement", "орлогын дэлгэрэнгүй тайлан", "орлогын тайлан",
            "balance sheet", "тайлан баланс", "санхүүгийн байдлын тайлан", "cash flow statement", "мөнгөн гүйлгээний тайлан",
            "statement of changes in equity", "өмчийн өөрчлөлтйин тайлан", # Corrected typo
        }
        anchor_keywords_secondary = { 
            "notes to the financial statements", "санхүүгийн тайлангийн тодруулга", "assets", "хөрөнгө", 
            "liabilities", "өр төлбөр", "equity", "эздийн өмч", "revenue", "орлого", 
            "net income", "цэвэр ашиг", "operating activities", "үндсэн үйл ажиллагааны"
        }
        anchor_pages = []
        pages_content = parsed_doc_content.get("pages_content", [])
        num_pages = len(pages_content)
        if max_pages_to_check is not None: num_pages = min(num_pages, max_pages_to_check)

        for i in range(num_pages):
            page_data = pages_content[i]
            page_text_parts = [pt for pt in [page_data.get("text_pymupdf"), page_data.get("page_full_ocr_text")] if pt]
            page_text_content_lower = "\n".join(page_text_parts).lower()
            if not page_text_content_lower.strip(): continue

            primary_hits = sum(1 for kw in anchor_keywords_primary if kw in page_text_content_lower)
            secondary_hits = sum(1 for kw in anchor_keywords_secondary if kw in page_text_content_lower)
            if (primary_hits >= 1 and secondary_hits >=1) or primary_hits >=2:
                if page_data['page_number'] not in anchor_pages: anchor_pages.append(page_data['page_number'])
        
        anchor_pages.sort()
        self.logger.info(f"Identified potential financial statement anchor pages: {anchor_pages}")
        return anchor_pages

    def _get_context_for_item(
        self, parsed_doc_content: Dict[str, Any], anchor_pages: List[int],
        item_name_english: Optional[str] = None, item_name_mongolian: Optional[str] = None,
        period_label: Optional[str] = None, section_hint: Optional[str] = None,
        task_type: Literal["metadata", "period_id", "line_item"] = "line_item",
        max_context_length: int = 15000
    ) -> str:
        log_item_name = item_name_english or section_hint or "Unknown Item"
        self.logger.debug(f"Gathering context for: Task='{task_type}', Item='{log_item_name}', Period='{period_label or 'N/A'}'")
        context_parts: List[Tuple[int, str]] = []
        all_keywords = [kw.lower() for kw in [item_name_english, item_name_mongolian, period_label, section_hint] if kw and isinstance(kw, str) and kw.strip()]

        pages_content_map = {p['page_number']: p for p in parsed_doc_content.get("pages_content", [])}

        if task_type in ["line_item", "period_id"] and anchor_pages:
            for anchor_pn in anchor_pages:
                for page_offset in range(-2, 4): # Window: anchor -2, -1, 0, +1, +2, +3
                    target_pn = anchor_pn + page_offset
                    page_data = pages_content_map.get(target_pn)
                    if page_data:
                        page_text = "\n".join(pt for pt in [page_data.get("text_pymupdf"), page_data.get("page_full_ocr_text")] if pt)
                        if not page_text.strip(): continue
                        
                        relevant_for_task = True
                        if task_type == "line_item" and all_keywords and not any(kw in page_text.lower() for kw in all_keywords):
                            relevant_for_task = False
                        
                        if relevant_for_task:
                            snippet = f"\n--- Page {page_data['page_number']} (Anchor Window) ---\n{page_text[:3000]}"
                            context_parts.append((1, snippet))

        include_initial_pages = task_type == "metadata" or (not anchor_pages and task_type != "line_item")
        initial_page_scan_count = 7 if task_type == "metadata" else 3 # More pages for metadata
        for i in range(min(len(parsed_doc_content.get("pages_content", [])), initial_page_scan_count)):
            page_data = parsed_doc_content["pages_content"][i]
            if any(f"--- Page {page_data['page_number']}" in cp_text for _, cp_text in context_parts): continue # Avoid re-adding if in anchor window
            page_text = "\n".join(pt for pt in [page_data.get("text_pymupdf"), page_data.get("page_full_ocr_text")] if pt)
            if not page_text.strip(): continue

            if include_initial_pages or (all_keywords and any(kw in page_text.lower() for kw in all_keywords)):
                snippet = f"\n--- Page {page_data['page_number']} (Initial Scan) ---\n{page_text[:2000]}"
                context_parts.append((2 if include_initial_pages else 3, snippet))

        if all_keywords and task_type == "line_item":
            for page_num_key in sorted(pages_content_map.keys()): # Iterate all pages by number
                page_data = pages_content_map[page_num_key]
                if any(f"--- Page {page_data['page_number']}" in cp_text for _, cp_text in context_parts): continue
                page_text = "\n".join(pt for pt in [page_data.get("text_pymupdf"), page_data.get("page_full_ocr_text")] if pt)
                if not page_text.strip(): continue
                if any(kw in page_text.lower() for kw in all_keywords):
                    snippet = f"\n--- Page {page_data['page_number']} (Keyword Match) ---\n{page_text[:1500]}"
                    context_parts.append((4, snippet))

        content_pages_in_context_texts = {cp_text for _, cp_text in context_parts}
        for table_data in parsed_doc_content.get("all_tables", []):
            table_page_num = table_data['page_number']
            table_str_content = ""
            for r_idx, row in enumerate(table_data.get("rows",[])): table_str_content += f"R{r_idx}: {' | '.join([str(c) if c is not None else '' for c in row])}\n"
            if not table_str_content.strip(): continue
            
            is_near_anchor = any(abs(table_page_num - anchor_pn) <= 2 for anchor_pn in anchor_pages)
            has_keywords_in_table = all_keywords and any(kw in table_str_content.lower() for kw in all_keywords)
            
            # Only add table if its page is already selected, or it's near an anchor, or it has keywords
            page_already_selected = any(f"--- Page {table_page_num} " in text for text in content_pages_in_context_texts)
            if page_already_selected or is_near_anchor or has_keywords_in_table:
                snippet = f"\n--- Page {table_page_num}, Table {table_data['table_index_on_page']} ---\n{table_str_content[:1500]}"
                priority = 1 if is_near_anchor and has_keywords_in_table else (2 if is_near_anchor or has_keywords_in_table else 5)
                context_parts.append((priority, snippet))
        
        context_parts.sort(key=lambda x: x[0])
        final_context_str = ""
        seen_core_snippets = set()
        for _, snippet_full in context_parts:
            core_content = snippet_full.split("---\n", 1)[-1] if "---\n" in snippet_full else snippet_full
            if core_content not in seen_core_snippets:
                if len(final_context_str) + len(snippet_full) > max_context_length:
                    remaining_len = max_context_length - len(final_context_str)
                    if remaining_len > 200: final_context_str += snippet_full[:remaining_len]
                    self.logger.warning(f"Context for '{log_item_name}' (Task: {task_type}) truncated during prioritized build.")
                    break 
                final_context_str += snippet_full
                seen_core_snippets.add(core_content)
        
        if not final_context_str.strip():
            final_context_str = "No relevant context found or constructed. Document might be empty or keywords did not match."
            self.logger.warning(f"No context constructed for '{log_item_name}' (Task: {task_type}).")
        self.logger.debug(f"Final context for '{log_item_name}' (Task: {task_type}, len {len(final_context_str)}):\n{final_context_str[:200]}...")
        return final_context_str
    
    def _get_line_item_mongolian_name(self, item_schema: Dict[str, Any]) -> str:
        item_properties = self._get_effective_item_schema(item_schema).get("properties", {}) # Use effective schema
        name_mn_schema = item_properties.get("name_mn", {})
        name_mn = name_mn_schema.get("const", name_mn_schema.get("default"))

        if name_mn and isinstance(name_mn, str) and name_mn.strip():
            return name_mn
        
        name_en_schema = item_properties.get("name", {})
        name_en = name_en_schema.get("const", name_en_schema.get("default", "UnknownLineItem_MN_Missing"))
        # self.logger.debug(f"Mongolian name ('name_mn') not found or empty for item using English name '{name_en}'.")
        return str(name_en)

    async def _call_llm_with_retry(self, base_prompt_key: str, prompt_args: Dict[str, Any], response_model: Optional[Type[BaseModel]] = None, max_attempts: int = 3) -> Optional[Dict[str, Any]]:
        current_prompt_key = base_prompt_key
        for attempt in range(1, max_attempts + 1):
            item_name_for_log = prompt_args.get('line_item_english_name', prompt_args.get('section_hint', 'N/A'))
            period_for_log = prompt_args.get('period_label', 'N/A')
            self.logger.info(f"LLM call attempt {attempt}/{max_attempts} for: Item='{item_name_for_log}', Period='{period_for_log}', PromptKey='{current_prompt_key}'")
            
            prompt_template_obj = self.prompts.get(current_prompt_key)
            if not prompt_template_obj or "prompt_template" not in prompt_template_obj:
                self.logger.error(f"Prompt template for key '{current_prompt_key}' not found or invalid in YAML."); return None
            prompt_template = prompt_template_obj["prompt_template"]
            
            try: formatted_prompt = prompt_template.format(**prompt_args)
            except KeyError as e: self.logger.error(f"Missing key for template '{current_prompt_key}': {e}. Args: {list(prompt_args.keys())}"); return None
            
            self.logger.debug(f"Formatted prompt (attempt {attempt}, first 500 chars):\n{formatted_prompt[:500]}...")
            llm_call_args = {"prompt": formatted_prompt, "json_mode": True} # Default to JSON mode expectation
            
            try: raw_response = await self.llm_service.invoke(**llm_call_args)
            except Exception as e:
                self.logger.error(f"LLM invocation failed for '{item_name_for_log}' (attempt {attempt}): {e}", exc_info=True)
                if attempt == max_attempts: self.logger.error(f"Max LLM attempts reached for {item_name_for_log}."); return None
                await asyncio.sleep(2 ** attempt); current_prompt_key = base_prompt_key; continue # Exponential backoff, reset prompt key
            
            self.logger.debug(f"Raw LLM response for '{item_name_for_log}' (attempt {attempt}, first 500 chars):\n{raw_response[:500]}...")
            try:
                cleaned_response_str = raw_response.strip()
                if cleaned_response_str.startswith("```json"): cleaned_response_str = cleaned_response_str[len("```json"):].strip()
                if cleaned_response_str.endswith("```"): cleaned_response_str = cleaned_response_str[:-len("```")].strip()
                if not cleaned_response_str: self.logger.warning(f"LLM response for '{item_name_for_log}' empty (attempt {attempt})."); raise json.JSONDecodeError("Empty response", "", 0)
                
                parsed_json = json.loads(cleaned_response_str)
                if response_model: response_model(**parsed_json) # Pydantic validation
                self.logger.info(f"Successfully parsed & validated LLM JSON for '{item_name_for_log}' (attempt {attempt}).")
                return parsed_json
            except json.JSONDecodeError as e_json: self.logger.warning(f"Failed to decode JSON for '{item_name_for_log}' (attempt {attempt}): {e_json}. Response snippet: {cleaned_response_str[:300]}"); current_prompt_key = self.prompts.get(f"{base_prompt_key}_re_prompt_invalid_json", {}).get("next_key", "re_prompt_invalid_json")
            except ValidationError as e_val: self.logger.warning(f"Pydantic validation failed for '{item_name_for_log}' (attempt {attempt}): {e_val}"); current_prompt_key = self.prompts.get(f"{base_prompt_key}_re_prompt_missing_keys", {}).get("next_key", "re_prompt_missing_keys")
            except Exception as e_parse: self.logger.error(f"Unexpected error parsing LLM response for '{item_name_for_log}' (attempt {attempt}): {e_parse}", exc_info=True); current_prompt_key = base_prompt_key # Reset to base for generic error
            
            if attempt == max_attempts: self.logger.error(f"Max LLM parsing/validation attempts reached for {item_name_for_log}."); return None
            await asyncio.sleep(attempt) # Simple linear backoff for parsing retries
        return None

    async def _extract_metadata(self, parsed_doc_content: Dict[str, Any], document_type: str):
        self.logger.info("Extracting model metadata...")
        anchor_pages_for_meta = self._find_financial_statement_anchor_pages(parsed_doc_content, max_pages_to_check=15) # Quick scan
        context = self._get_context_for_item(
            parsed_doc_content, anchor_pages=anchor_pages_for_meta, 
            section_hint="company identification, currency, fiscal year, reporting period / компанийн танилцуулга, валют, санхүүгийн жил, тайлант үе",
            task_type="metadata", max_context_length=8000 
        )
        prompt_args = {"context": context, "document_type": document_type}
        metadata_json = await self._call_llm_with_retry("metadata_extraction", prompt_args, response_model=ExtractedMetadata)
        
        model_meta_data_ptr = self.financial_model.setdefault("model_metadata", {})
        if metadata_json:
            model_meta_data_ptr.update(metadata_json)
            fye_schema = self._get_effective_item_schema(self.model_schema_definition.get("properties",{}).get("model_metadata",{}).get("properties",{}).get("fiscal_year_end",{}))
            if "fiscal_year_end" in model_meta_data_ptr and model_meta_data_ptr["fiscal_year_end"] is None:
                if fye_schema.get("type") == "string" and "null" not in (fye_schema.get("type") if isinstance(fye_schema.get("type"), list) else [fye_schema.get("type")]):
                    self.logger.warning("LLM returned null for fiscal_year_end; schema requires string. Setting to empty string.")
                    model_meta_data_ptr["fiscal_year_end"] = ""
            self._log_extraction_success("Model Metadata", "N/A", metadata_json)
        else:
            self.logger.warning("Failed to extract model metadata."); self._log_extraction_failure("Model Metadata", "N/A", "LLM failed")
            if model_meta_data_ptr.get("fiscal_year_end") is None: model_meta_data_ptr["fiscal_year_end"] = ""

    async def _identify_historical_periods(self, parsed_doc_content: Dict[str, Any], document_type: str) -> List[str]:
        self.logger.info("Identifying historical periods...")
        anchor_pages = self._find_financial_statement_anchor_pages(parsed_doc_content)
        context = self._get_context_for_item(
            parsed_doc_content, anchor_pages=anchor_pages,
            section_hint="financial statements column headers years periods table of contents for statements / санхүүгийн тайлангийн баганын гарчиг он үеүд гарчиг",
            task_type="period_id", max_context_length=12000 
        )
        prompt_args = {"context": context, "document_type": document_type}
        periods_json = await self._call_llm_with_retry("historical_period_identification", prompt_args, response_model=IdentifiedPeriods)
        
        identified_periods = []
        if periods_json and isinstance(periods_json.get("historical_period_labels"), list):
            raw_periods = periods_json["historical_period_labels"]
            # Validate and sanitize period labels
            validated_periods = []
            for p_label_any in raw_periods:
                p_label = str(p_label_any).strip()
                if p_label and (p_label.upper().startswith("FY") or \
                                p_label.upper().startswith("Q") or \
                                p_label.upper().startswith("H") or \
                                p_label.replace(" ","").replace("он","").replace("оны","").isnumeric() or \
                                (p_label.count('-') == 1 and all(part.strip().isnumeric() for part in p_label.split('-'))) or \
                                (p_label.count('/') == 1 and all(part.strip().isnumeric() for part in p_label.split('/')))
                                ):
                    validated_periods.append(p_label)
                else:
                    self.logger.warning(f"Invalid period label format '{p_label_any}' received from LLM. Skipping.")

            if len(validated_periods) != len(raw_periods):
                self.logger.warning(f"Some identified periods might be invalid or filtered. Original: {raw_periods}, Validated: {validated_periods}")
            identified_periods = validated_periods
            self.logger.info(f"Identified and validated historical periods: {identified_periods}")
            self._log_extraction_success("Historical Period Labels", "N/A", identified_periods)
        else:
            self.logger.warning(f"Failed to identify historical periods or invalid format. JSON: {periods_json}")
            self._log_extraction_failure("Historical Period Labels", "N/A", f"LLM failed or returned invalid format. Got: {periods_json}")
        
        fsc_ptr = self.financial_model.setdefault("financial_statements_core", {})
        fsc_ptr["historical_period_labels"] = identified_periods 
        return identified_periods

    async def _extract_historical_line_item_data(
        self, item_schema: Dict[str, Any], period_label: str, parsed_doc_content: Dict[str, Any], anchor_pages: List[int]
    ) -> Dict[str, Any]:
        item_props = self._get_effective_item_schema(item_schema).get("properties", {})
        item_name_en = item_props.get("name", {}).get("const", item_props.get("name", {}).get("default", "Unknown Item"))
        item_name_mn = self._get_line_item_mongolian_name(item_schema)
        item_guidance = item_props.get("source_guidance_historical", {}).get("default", "Refer to standard financial definitions.")
        
        self.logger.info(f"Extracting: '{item_name_en}' (MN: '{item_name_mn}') for period '{period_label}'")
        context = self._get_context_for_item(
            parsed_doc_content, anchor_pages=anchor_pages, item_name_english=item_name_en,
            item_name_mongolian=item_name_mn, period_label=period_label,
            section_hint=f"{item_name_en} value or table / {item_name_mn} утга эсвэл хүснэгт",
            task_type="line_item", max_context_length=15000
        )
        prompt_args = {"period_label": period_label, "line_item_english_name": item_name_en, "line_item_mongolian_name": item_name_mn, "line_item_definition": item_guidance, "context": context}
        extracted_json = await self._call_llm_with_retry("single_line_item_extraction", prompt_args, response_model=ExtractedLineItemDetail)
        
        output_period_data = {"period_label": period_label, "value": None, "source_reference": None, "extraction_status": "FAILURE_UNKNOWN"}
        if extracted_json:
            try:
                validated = ExtractedLineItemDetail(**extracted_json)
                output_period_data["source_reference"] = validated.source_reference
                if validated.status == "EXTRACTED_SUCCESSFULLY":
                    norm_val, norm_curr, norm_unit = self.utils.normalize_value(validated.value, validated.unit, validated.currency)
                    output_period_data["value"] = norm_val
                    # Could also store norm_curr, norm_unit if needed in period object
                    output_period_data["extraction_status"] = "SUCCESS"
                    self._log_extraction_success(item_name_en, period_label, validated.model_dump())
                elif validated.status == "CONFIRMED_NOT_FOUND":
                    output_period_data["extraction_status"] = "NOT_FOUND_PER_LLM"
                    self._log_extraction_success(item_name_en, period_label, validated.model_dump(), status_override="CONFIRMED_NOT_FOUND")
                else: 
                    output_period_data["extraction_status"] = f"FAILURE_UNEXPECTED_LLM_STATUS:{validated.status}"
                    self._log_extraction_failure(item_name_en, period_label, f"Unexpected LLM status: {validated.status}", validated.model_dump())
            except Exception as e: 
                output_period_data["extraction_status"] = "FAILURE_PROCESSING_ERROR"
                self.logger.error(f"Error processing extracted data for {item_name_en}/{period_label}: {e}", exc_info=True)
                self._log_extraction_failure(item_name_en, period_label, f"Processing error: {e}", extracted_json)
        else: 
            output_period_data["extraction_status"] = "FAILURE_MAX_ATTEMPTS"
            self.logger.warning(f"Failed all LLM attempts for '{item_name_en}'/'{period_label}'.")
            self._log_extraction_failure(item_name_en, period_label, "LLM failed all attempts")
        return output_period_data

    async def _initialize_single_line_item_object_from_schema(
        self, line_item_data_ptr: Dict[str,Any], item_schema: Dict[str, Any], 
        historical_periods: List[str], parsed_doc_content: Dict[str, Any], 
        anchor_pages: List[int], path_prefix: str, is_direct_call: bool
    ):
        item_props = self._get_effective_item_schema(item_schema).get("properties", {}) # Use effective
        item_name = item_props.get("name", {}).get("const", item_props.get("name", {}).get("default", path_prefix.split('.')[-1]))
        item_name_mn = self._get_line_item_mongolian_name(item_schema) # Use effective schema
        is_calculated = item_props.get("is_calculated", {}).get("const", item_props.get("is_calculated", {}).get("default", False))

        line_item_data_ptr["name"] = item_name
        if item_name_mn and item_name_mn != item_name : line_item_data_ptr["name_mn"] = item_name_mn # Add if defined and different
        line_item_data_ptr["is_calculated"] = is_calculated
        line_item_data_ptr["periods"] = []
        
        # Initialize all static properties defined in financial_line_item_base_schema or the specific item
        for static_prop_key in ["data_type", "calculation_logic_description", "source_guidance_historical", 
                                "ai_instructions", "ai_instructions_projected", "notes", "unit", 
                                "target_value", "tolerance"]: # Added target_value, tolerance for balance_sheet_check
            if static_prop_key in item_props:
                prop_detail_schema = item_props[static_prop_key]
                value = prop_detail_schema.get("const", prop_detail_schema.get("default"))
                if value is not None:
                    line_item_data_ptr[static_prop_key] = value
                elif prop_detail_schema.get("type") == "string" and "null" not in (prop_detail_schema.get("type") if isinstance(prop_detail_schema.get("type"), list) else [prop_detail_schema.get("type")]):
                     line_item_data_ptr[static_prop_key] = "" # Default to empty string if schema requires string and no default/const

        if is_calculated:
            self.logger.info(f"Setting up calculated single item: {item_name} at {path_prefix}")
            self._log_extraction_success(item_name, "N/A", "Calculated item", status_override="CALCULATED_ITEM")
        elif not is_direct_call:
            self.logger.info(f"Extracting data for single non-calculated item: {item_name} at {path_prefix}")
            for period in historical_periods:
                extracted_period_data = await self._extract_historical_line_item_data(
                    item_schema, period, parsed_doc_content, anchor_pages
                )
                line_item_data_ptr["periods"].append(extracted_period_data)
        elif is_direct_call and not is_calculated:
             self.logger.warning(f"Item {item_name} at {path_prefix} was expected to be calculated by direct call but schema says is_calculated=False. Extraction not performed by this initialization path.")

    async def _process_statement_section(self,
                                         section_data_model_container: Dict[str, Any], 
                                         section_array_schema: Dict[str, Any], 
                                         historical_periods: List[str],
                                         parsed_doc_content: Dict[str, Any],
                                         anchor_pages: List[int],
                                         path_prefix: str):
        line_items_schema_list = self._get_effective_item_schema(section_array_schema).get("items", [])
        if not isinstance(line_items_schema_list, list):
            self.logger.warning(f"In _process_statement_section for '{path_prefix}', expected 'items' to be a list. Got {type(line_items_schema_list)}. Skipping.")
            section_data_model_container.setdefault("line_items", [])
            return

        target_line_items_list_in_model = section_data_model_container.setdefault("line_items", [])
        target_line_items_list_in_model.clear() 

        for item_idx, specific_item_schema_orig in enumerate(line_items_schema_list):
            item_schema_effective = self._get_effective_item_schema(specific_item_schema_orig)
            item_props = item_schema_effective.get("properties", {})
            item_name = item_props.get("name", {}).get("const", item_props.get("name", {}).get("default", f"UnnamedItem_{path_prefix}_{item_idx}"))
            item_name_mn = self._get_line_item_mongolian_name(item_schema_effective)
            is_calculated = item_props.get("is_calculated", {}).get("const", item_props.get("is_calculated", {}).get("default", False))
            
            current_line_item_data_for_model = {"name": item_name, "is_calculated": is_calculated, "periods": []}
            if item_name_mn and item_name_mn != item_name: current_line_item_data_for_model["name_mn"] = item_name_mn
            
            for static_prop_key in ["data_type", "calculation_logic_description", "source_guidance_historical", 
                                    "ai_instructions", "ai_instructions_projected", "notes", "unit"]:
                if static_prop_key in item_props:
                    prop_detail = item_props[static_prop_key]
                    value = prop_detail.get("const", prop_detail.get("default"))
                    if value is not None: current_line_item_data_for_model[static_prop_key] = value
            
            if is_calculated:
                self.logger.info(f"Skipping extraction for calculated item: {item_name} in {path_prefix}")
                self._log_extraction_success(item_name, "N/A", "Calculated item", status_override="CALCULATED_ITEM")
            else:
                for period in historical_periods:
                    extracted_period_data = await self._extract_historical_line_item_data(
                        item_schema_effective, period, parsed_doc_content, anchor_pages
                    )
                    current_line_item_data_for_model["periods"].append(extracted_period_data)
            target_line_items_list_in_model.append(current_line_item_data_for_model)

    async def run_phase_a_extraction(self, parsed_doc_content: Dict[str, Any], document_type_hint: str = "Financial Report") -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        self.logger.info("Starting Phase A: Historical Data Extraction (Schema-Driven v2.3 - Context Revamp).")
        anchor_pages = self._find_financial_statement_anchor_pages(parsed_doc_content)
        await self._extract_metadata(parsed_doc_content, document_type_hint)
        historical_periods = await self._identify_historical_periods(parsed_doc_content, document_type_hint) 
        
        if not historical_periods:
            self.logger.error("No valid historical periods identified by LLM. Aborting line item extraction.")
            return self.financial_model, self.extraction_log

        fsc_data_model_ptr = self.financial_model.setdefault("financial_statements_core", {})
        fsc_data_model_ptr["historical_period_labels"] = historical_periods

        fsc_schema_properties = self._get_schema_property(["financial_statements_core"])
        if not fsc_schema_properties:
            self.logger.error("Schema for 'financial_statements_core.properties' not found. Cannot process.")
            return self.financial_model, self.extraction_log

        for stmt_key, stmt_schema_orig in fsc_schema_properties.items():
            if not isinstance(stmt_schema_orig, dict): self.logger.debug(f"Skipping non-dict schema entry '{stmt_key}' under FSC."); continue
            if stmt_key in ["historical_period_labels", "projected_period_labels", "section_description", "ai_instructions_statement"]: continue
            
            self.logger.info(f"Processing statement object property: financial_statements_core.{stmt_key}")
            stmt_schema_effective = self._get_effective_item_schema(stmt_schema_orig)
            stmt_data_model_ptr = fsc_data_model_ptr.setdefault(stmt_key, {})

            if stmt_schema_effective.get("type") != "object": 
                self.logger.warning(f"Schema for '{stmt_key}' is not 'object' type. Skipping detailed processing."); continue
            
            stmt_actual_properties = stmt_schema_effective.get("properties", {})
            if "line_items" in stmt_actual_properties:
                line_items_array_schema = self._get_effective_item_schema(stmt_actual_properties["line_items"])
                await self._process_statement_section(
                    stmt_data_model_ptr, line_items_array_schema, 
                    historical_periods, parsed_doc_content, anchor_pages,
                    f"financial_statements_core.{stmt_key}"
                )
            elif stmt_key == "balance_sheet":
                bs_main_categories = stmt_actual_properties
                for bs_main_cat_key, bs_main_cat_schema_orig in bs_main_categories.items():
                    if not isinstance(bs_main_cat_schema_orig, dict) or bs_main_cat_key == "ai_instructions_statement": continue
                    
                    bs_main_cat_schema_effective = self._get_effective_item_schema(bs_main_cat_schema_orig)
                    bs_main_cat_data_ptr = stmt_data_model_ptr.setdefault(bs_main_cat_key, {})
                    
                    if bs_main_cat_schema_effective.get("type") == "object" and "properties" in bs_main_cat_schema_effective:
                        bs_sub_categories = bs_main_cat_schema_effective.get("properties",{})
                        for bs_sub_cat_key, bs_sub_cat_schema_orig in bs_sub_categories.items():
                            if not isinstance(bs_sub_cat_schema_orig, dict): continue
                            if bs_main_cat_key == "balance_sheet_check" and bs_sub_cat_key == "periods": continue # Handled by init/QC
                            
                            bs_sub_cat_schema_effective = self._get_effective_item_schema(bs_sub_cat_schema_orig)
                            current_path_prefix = f"financial_statements_core.{stmt_key}.{bs_main_cat_key}.{bs_sub_cat_key}"
                            if bs_sub_cat_schema_effective.get("type") == "array":
                                bs_sub_cat_data_list_ptr = bs_main_cat_data_ptr.setdefault(bs_sub_cat_key, [])
                                if not isinstance(bs_sub_cat_data_list_ptr, list): bs_sub_cat_data_list_ptr = []; bs_main_cat_data_ptr[bs_sub_cat_key] = bs_sub_cat_data_list_ptr
                                temp_list_wrapper = {"line_items": bs_sub_cat_data_list_ptr}
                                await self._process_statement_section(
                                    temp_list_wrapper, bs_sub_cat_schema_effective, 
                                    historical_periods, parsed_doc_content, anchor_pages,
                                    current_path_prefix
                                )
                                bs_main_cat_data_ptr[bs_sub_cat_key] = temp_list_wrapper["line_items"]
                            elif bs_sub_cat_schema_effective.get("type") == "object":
                                single_item_data_ptr = bs_main_cat_data_ptr.setdefault(bs_sub_cat_key, {})
                                item_is_calc = bs_sub_cat_schema_effective.get("properties", {}).get("is_calculated", {}).get("const", False)
                                await self._initialize_single_line_item_object_from_schema(
                                    single_item_data_ptr, bs_sub_cat_schema_effective, 
                                    historical_periods, parsed_doc_content, anchor_pages,
                                    current_path_prefix, is_direct_call=item_is_calc 
                                )
                    elif bs_main_cat_schema_effective.get("type") == "object" and bs_main_cat_key == "balance_sheet_check":
                         is_calc_check = bs_main_cat_schema_effective.get("properties", {}).get("is_calculated", {}).get("const", True)
                         await self._initialize_single_line_item_object_from_schema(
                             bs_main_cat_data_ptr, bs_main_cat_schema_effective, 
                             historical_periods, parsed_doc_content, anchor_pages,
                             f"financial_statements_core.{stmt_key}.{bs_main_cat_key}", is_direct_call=is_calc_check
                         )
            elif stmt_key == "cash_flow_statement":
                cfs_sections = stmt_actual_properties
                for cfs_section_key, cfs_section_schema_orig in cfs_sections.items():
                    if not isinstance(cfs_section_schema_orig, dict) or cfs_section_key == "ai_instructions_statement": continue
                    
                    cfs_section_schema_effective = self._get_effective_item_schema(cfs_section_schema_orig)
                    current_path_prefix = f"financial_statements_core.{stmt_key}.{cfs_section_key}"
                    cfs_section_data_model_ptr = stmt_data_model_ptr.setdefault(cfs_section_key, {})
                    
                    if cfs_section_schema_effective.get("type") == "array":
                        list_to_populate = cfs_section_data_model_ptr if isinstance(cfs_section_data_model_ptr, list) else cfs_section_data_model_ptr.setdefault("line_items", [])
                        if not isinstance(list_to_populate, list): list_to_populate = []; # Ensure it's a list
                        
                        temp_list_wrapper = {"line_items": list_to_populate}
                        await self._process_statement_section(
                            temp_list_wrapper, cfs_section_schema_effective,
                            historical_periods, parsed_doc_content, anchor_pages,
                            current_path_prefix
                        )
                        # Update based on how it was wrapped/passed
                        if isinstance(cfs_section_data_model_ptr, list):
                            stmt_data_model_ptr[cfs_section_key] = temp_list_wrapper["line_items"]
                        else: # It was an object, update its 'line_items'
                            cfs_section_data_model_ptr["line_items"] = temp_list_wrapper["line_items"]
                            
                    elif cfs_section_schema_effective.get("type") == "object":
                        is_calc_cfs_item = cfs_section_schema_effective.get("properties",{}).get("is_calculated",{}).get("const", True) # Default true for single CFS items unless specified
                        await self._initialize_single_line_item_object_from_schema(
                            cfs_section_data_model_ptr, cfs_section_schema_effective, 
                            historical_periods, parsed_doc_content, anchor_pages,
                            current_path_prefix, is_direct_call=is_calc_cfs_item 
                        )
        self.logger.info("Phase A: Historical Data Extraction finished.")
        return self.financial_model, self.extraction_log

    def _log_extraction_success(self, item:str, period:str, data:Any, status_override:Optional[str]=None): 
        self.extraction_log.append({"item_name":item,"period":period,"status":status_override or "SUCCESS","data":data})
    def _log_extraction_failure(self, item:str, period:str, reason:str, raw:Optional[Any]=None): 
        self.extraction_log.append({"item_name":item,"period":period,"status":"FAILURE","reason":reason,"raw_data":raw})

class GeminiLlmService(BaseLlmService): # Ensure BaseLlmService is defined/imported
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.logger = logging.getLogger(__name__ + ".GeminiLlmService")

        self.api_key = config.GOOGLE_API_KEY
        self.model_name = model_name or config.ACTIVE_GOOGLE_MODEL
        
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            self.logger.info(f"GeminiLlmService initialized with model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini GenerativeModel: {e}", exc_info=True)
            raise

    async def invoke(
        self, prompt: str, model_name: Optional[str] = None,
        temperature: float = 0.2, # Adjusted for more deterministic extraction
        max_output_tokens: int = 8192, # Gemini 1.5 Flash has large context
        json_mode: bool = True, # Default to True for financial extraction
        **kwargs 
    ) -> str:
        current_model_name = model_name or self.model_name
        model_to_use = self.model
        if model_name and model_name != self.model_name: # If a different model is requested dynamically
            self.logger.info(f"Switching to model: {model_name} for this call.")
            try:
                model_to_use = genai.GenerativeModel(model_name)
            except Exception as e:
                self.logger.error(f"Failed to switch to Gemini model {model_name}: {e}. Using default {self.model_name}")
                model_to_use = self.model
        
        self.logger.debug(f"Invoking Gemini model {current_model_name} with prompt (first 200 chars): {prompt[:200]}...")

        generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        if json_mode:
            # For Gemini, JSON mode is enabled by specific instruction in the prompt
            # and ensuring the model output is clean JSON.
            # Actual API parameter for JSON mode might be different or implicit.
            # For "gemini-1.5-flash-latest" and similar, you often instruct it
            # in the prompt to "Respond only with a valid JSON object."
            # For some models, response_mime_type="application/json" in GenerationConfig works.
            # Check Gemini API docs for the specific model if direct JSON mode is supported.
            # For gemini-1.5-flash, it's usually through prompt engineering.
            # generation_config.response_mime_type = "application/json" # May not be supported by all gemini models explicitly this way
            pass # JSON mode is typically by prompt instruction for latest Gemini models


        # Define safety settings to be less restrictive for financial data
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        try:
            response = await model_to_use.generate_content_async(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Log reasons if blocked
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                self.logger.error(f"Prompt blocked by Gemini. Reason: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason}")
                raise ToolException(f"Gemini prompt blocked: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason}")

            # Check for empty candidates or parts
            if not response.candidates or not response.candidates[0].content.parts:
                self.logger.warning(f"Gemini response empty or has no parts. Candidates: {response.candidates}")
                # Check if it was blocked for safety, even if block_reason was not in prompt_feedback
                if response.candidates and response.candidates[0].finish_reason == genai.types.FinishReason.SAFETY:
                    safety_ratings_str = ", ".join([f"{sr.category.name}: {sr.probability.name}" for sr in response.candidates[0].safety_ratings])
                    self.logger.error(f"Content generation stopped due to safety. Ratings: {safety_ratings_str}")
                    raise ToolException(f"Gemini content generation blocked due to safety settings. Ratings: {safety_ratings_str}")
                return "" # Or raise an error for empty response

            response_text = response.text # Safely access .text
            self.logger.debug(f"Gemini response text (first 200 chars): {response_text[:200]}")
            return response_text
        except Exception as e:
            self.logger.error(f"Error during Gemini LLM call: {e}", exc_info=True)
            # Re-raise as ToolException or a custom LLMServiceException
            if isinstance(e, ToolException): raise 
            raise ToolException(f"Gemini API call failed: {str(e)}") from e

async def _orchestrator_standalone_test():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - [%(levelname)s] - %(name)s - %(module)s:%(lineno)d - %(message)s')
    # logging.getLogger("__main__.PhaseAOrchestrator").setLevel(logging.DEBUG) # More specific debug
    
    current_dir = Path(__file__).parent.parent.parent
    schema_p = current_dir / "fund" / "financial_model_schema.json" # Should be the v2.1 schema
    prompts_p = current_dir / "prompts" / "financial_modeling_prompts.yaml"
    log_p = current_dir / "logs" / "financial_modeling_orchestrator_v2_1_test.log"
    if not schema_p.exists() or not prompts_p.exists(): print(f"Error: Schema or Prompts missing. Check paths."); return

    llm_service_instance = GeminiLlmService()
    mock_utils = FinancialModelingUtils()
    orchestrator = PhaseAOrchestrator(schema_p, prompts_p, llm_service_instance, mock_utils, log_p)
    
    mock_parsed_content = {
        "status": "success", "file_name": "dummy.pdf", "total_pages": 2, "processed_pages_count": 2,
        "pages_content": [{"page_number": 1, "text_pymupdf": "Cover Page Info..."}, {"page_number": 2, "text_pymupdf": "Financials..."}],
        "all_tables": [], "all_images_ocr_results": []
    }
    final_model, extraction_summary = await orchestrator.run_phase_a_extraction(mock_parsed_content)
    
    # print("\n--- Final Populated Model (Summary) ---")
    # print(f"Metadata: {json.dumps(final_model.get('model_metadata'), ensure_ascii=False, indent=2)}")
    fsc = final_model.get("financial_statements_core", {})
    # print(f"Historical Periods: {fsc.get('historical_period_labels')}")
    is_items = fsc.get("income_statement", {}).get("line_items", [])
    # print(f"Income Statement Items ({len(is_items)}):")
    for item in is_items[:3]: print(f"  - {item.get('name')}: {json.dumps(item.get('periods'), ensure_ascii=False, indent=2)}")
    if len(is_items) > 3: print(f"    ... and {len(is_items)-3} more IS items.")

    bs_assets_ca = fsc.get("balance_sheet", {}).get("assets", {}).get("current_assets", [])
    # print(f"BS Current Assets Items ({len(bs_assets_ca)}):")
    for item in bs_assets_ca[:2]: print(f"  - {item.get('name')}: {json.dumps(item.get('periods'), ensure_ascii=False, indent=2)}")

    print("\n--- First 5 Extraction Log Entries ---")
    for entry in extraction_summary[:5]: print(entry)
    if len(extraction_summary) > 5: print(f"... and {len(extraction_summary)-5} more log entries.")

    # Optionally, save the full model to a file for inspection
    # with open("test_output_financial_model.json", "w", encoding="utf-8") as f:
    #    json.dump(final_model, f, ensure_ascii=False, indent=2)
    # print("\nFull model saved to test_output_financial_model.json")


if __name__ == "__main__":
    asyncio.run(_orchestrator_standalone_test())