import asyncio
import json
import logging
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Literal, Union

from pydantic import BaseModel, ValidationError

# --- LLM Service Placeholder ---
class BaseLlmService:
    async def invoke(
        self, prompt: str, model_name: Optional[str] = "gemini-1.5-flash-latest",
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

# --- Orchestrator Class ---
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
        logger_instance.setLevel(logging.INFO) # Default INFO, can be overridden by test
        if not logger_instance.handlers:
            fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
            fh.setFormatter(formatter)
            logger_instance.addHandler(fh)
        return logger_instance

# In class PhaseAOrchestrator:

    def _get_schema_property(self, path_keys: List[str]) -> Dict[str, Any]:
        """
        Navigates the loaded schema definition to retrieve a specific schema part.
        - If path_keys is empty, returns the root 'properties' of the entire schema.
        - Otherwise, navigates through nested 'properties' for each key in path_keys.
        - For the FINAL key in path_keys, it returns the schema definition of that property.
          If that final property's schema is an object, this method will return the
          dictionary found under ITS 'properties' key (i.e., its children's schemas).
        Resolves $refs along the way.
        """
        current_schema_level = self.model_schema_definition
        
        if not path_keys: # Requesting root properties of the entire schema
            return current_schema_level.get("properties", {})

        # Start navigation from the root "properties" if path_keys is not empty
        if "properties" in current_schema_level:
            current_schema_level = current_schema_level["properties"]
        else:
            self.logger.error("Root 'properties' key not found in schema definition.")
            return {}


        for i, key in enumerate(path_keys):
            # Resolve $ref at the current level
            if isinstance(current_schema_level, dict) and "$ref" in current_schema_level:
                current_schema_level = self._resolve_ref(current_schema_level["$ref"])
            
            # Access the specific key
            if isinstance(current_schema_level, dict) and key in current_schema_level:
                current_schema_level = current_schema_level[key]
            else:
                self.logger.warning(f"Schema path key '{key}' not found in current schema level. Path: {path_keys}. Current level: {list(current_schema_level.keys()) if isinstance(current_schema_level,dict) else 'Not a dict'}")
                return {} 
            
            # If it's not the last key and the current level is an object schema,
            # descend into its 'properties' for the next iteration.
            if i < len(path_keys) - 1:
                if isinstance(current_schema_level, dict) and "$ref" in current_schema_level: # Resolve if next level is $ref
                    current_schema_level = self._resolve_ref(current_schema_level["$ref"])
                if isinstance(current_schema_level, dict) and "properties" in current_schema_level:
                    current_schema_level = current_schema_level.get("properties", {})
                # If it's an array, and we expect to navigate further by index (not typical for property names)
                # elif isinstance(current_schema_level, dict) and "items" in current_schema_level and isinstance(current_schema_level["items"], list):
                #     pass # current_schema_level remains the schema for the array; next key might be an index.
                
        # After navigating to the final targeted schema part (e.g., schema for 'financial_statements_core')
        # resolve any $ref it might have.
        if isinstance(current_schema_level, dict) and "$ref" in current_schema_level:
            current_schema_level = self._resolve_ref(current_schema_level["$ref"])

        # If the final schema part is an object, return its 'properties' for iteration by the caller.
        if isinstance(current_schema_level, dict):
            if current_schema_level.get("type") == "object" and "properties" in current_schema_level:
                return current_schema_level["properties"]
            # If it's an array schema (e.g. for current_assets), return the schema for the array itself.
            # The caller (_process_statement_section) will get its "items".
            elif current_schema_level.get("type") == "array" and "items" in current_schema_level:
                 return current_schema_level # The caller will look for "items" within this.
            return current_schema_level # Return the schema object itself if not an object with properties for direct iteration
        
        return {}

    async def run_phase_a_extraction(self, parsed_doc_content: Dict[str, Any], document_type_hint: str = "Financial Report") -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        self.logger.info("Starting Phase A: Historical Data Extraction (Schema-Driven v2.2 - Corrected).")
        await self._extract_metadata(parsed_doc_content, document_type_hint)
        historical_periods = await self._identify_historical_periods(parsed_doc_content, document_type_hint)
        
        if not historical_periods:
            self.logger.error("No historical periods identified. Aborting line item extraction.")
            return self.financial_model, self.extraction_log

        fsc_data_model_ptr = self.financial_model.setdefault("financial_statements_core", {})
        fsc_data_model_ptr["historical_period_labels"] = historical_periods

        # Get the schema definition for the "financial_statements_core" object
        financial_statements_core_object_schema = self._get_schema_property(["financial_statements_core"])
        
        # The 'properties' of this object are the actual statements (income_statement, balance_sheet, etc.)
        statement_definitions_schemas = financial_statements_core_object_schema.get("properties", {})

        if not statement_definitions_schemas:
            self.logger.error("Properties for 'financial_statements_core' (i.e., statement definitions) not found in schema. Cannot process statements.")
            return self.financial_model, self.extraction_log

        for stmt_key, stmt_level_schema_orig in statement_definitions_schemas.items():
            # stmt_key examples: "income_statement", "balance_sheet", "cash_flow_statement", 
            # "section_description", "historical_period_labels", etc.
            # stmt_level_schema_orig is the schema for this specific property.

            if not isinstance(stmt_level_schema_orig, dict):
                self.logger.debug(f"Skipping non-dictionary entry '{stmt_key}' under financial_statements_core's properties.")
                continue

            # These are descriptive or already handled, not statement structures with line items
            if stmt_key in ["section_description", "historical_period_labels", "projected_period_labels", "ai_instructions_statement"]:
                # Ensure defaults for descriptive fields are set if not already by _initialize_model_from_schema_recursive
                if "default" in stmt_level_schema_orig and stmt_key not in fsc_data_model_ptr:
                     fsc_data_model_ptr[stmt_key] = stmt_level_schema_orig["default"]
                continue
            
            stmt_schema_effective = self._get_effective_item_schema(stmt_level_schema_orig)
            stmt_data_model_ptr = fsc_data_model_ptr.setdefault(stmt_key, {}) # e.g., model[...]["income_statement"] = {}

            if stmt_schema_effective.get("type") != "object":
                self.logger.warning(f"Property '{stmt_key}' under financial_statements_core is type '{stmt_schema_effective.get('type')}', not 'object' as expected for a statement. Skipping.")
                continue
            
            self.logger.info(f"Processing statement structure: financial_statements_core.{stmt_key}")

            # Standard statements like Income Statement, or complex objects like Balance Sheet
            statement_actual_properties = stmt_schema_effective.get("properties", {})

            if "line_items" in statement_actual_properties: # Directly contains line_items (e.g., Income Statement)
                line_items_array_schema = self._get_effective_item_schema(statement_actual_properties["line_items"])
                await self._process_statement_section(
                    stmt_data_model_ptr, # This is the dict that will hold 'line_items' list (e.g., model[...]['income_statement'])
                    line_items_array_schema, # Schema for the 'line_items' array itself
                    historical_periods, 
                    parsed_doc_content,
                    f"financial_statements_core.{stmt_key}"
                )
            elif stmt_key == "balance_sheet": # Balance Sheet has nested categories
                for bs_main_cat_key, bs_main_cat_schema_orig in statement_actual_properties.items():
                    # bs_main_cat_key is "assets", "liabilities_and_equity", "balance_sheet_check", "ai_instructions_statement"
                    if not isinstance(bs_main_cat_schema_orig, dict) or bs_main_cat_key == "ai_instructions_statement": continue

                    bs_main_cat_schema_effective = self._get_effective_item_schema(bs_main_cat_schema_orig)
                    bs_main_cat_data_ptr = stmt_data_model_ptr.setdefault(bs_main_cat_key, {})
                    
                    if bs_main_cat_schema_effective.get("type") == "object" and "properties" in bs_main_cat_schema_effective:
                        # For "assets" and "liabilities_and_equity" which contain further sub-categories
                        for bs_sub_cat_key, bs_sub_cat_schema_orig in bs_main_cat_schema_effective.get("properties", {}).items():
                            # bs_sub_cat_key is "current_assets", "non_current_assets", "total_assets" etc.
                            if not isinstance(bs_sub_cat_schema_orig, dict): continue
                            
                            bs_sub_cat_schema_effective = self._get_effective_item_schema(bs_sub_cat_schema_orig)
                            current_path_prefix = f"financial_statements_core.{stmt_key}.{bs_main_cat_key}.{bs_sub_cat_key}"

                            if bs_sub_cat_schema_effective.get("type") == "array":
                                # ... (existing correct logic for arrays like current_assets) ...
                                bs_sub_cat_data_list_ptr = bs_main_cat_data_ptr.setdefault(bs_sub_cat_key, [])
                                # ... (ensure list type) ...
                                temp_list_wrapper = {"line_items": bs_sub_cat_data_list_ptr}
                                await self._process_statement_section(
                                    temp_list_wrapper, bs_sub_cat_schema_effective, # Pass schema for the array
                                    historical_periods, parsed_doc_content, current_path_prefix
                                )
                                bs_main_cat_data_ptr[bs_sub_cat_key] = temp_list_wrapper["line_items"]
                            
                            elif bs_sub_cat_schema_effective.get("type") == "object":
                                # This is for single objects like "total_assets"
                                single_item_data_ptr = bs_main_cat_data_ptr.setdefault(bs_sub_cat_key, {})
                                # Directly call initialize for this single object.
                                # is_direct_call should be true if we are just setting up calculated items,
                                # false if we expect to extract non-calculated single items here.
                                # For total_assets, it's calculated.
                                item_is_calculated = bs_sub_cat_schema_effective.get("properties", {}).get("is_calculated", {}).get("const", False)

                                await self._initialize_single_line_item_object_from_schema(
                                    single_item_data_ptr, 
                                    bs_sub_cat_schema_effective, # Pass the schema of the object itself
                                    historical_periods, 
                                    parsed_doc_content, 
                                    current_path_prefix,
                                    is_direct_call=item_is_calculated # True if calculated, False if extraction needed
                                )
                    
                    elif bs_main_cat_schema_effective.get("type") == "object" and bs_main_cat_key == "balance_sheet_check":
                         # bs_main_cat_data_ptr is already the dict for "balance_sheet_check"
                         current_path_prefix_check = f"financial_statements_core.{stmt_key}.{bs_main_cat_key}"
                         item_is_calculated_check = bs_main_cat_schema_effective.get("properties", {}).get("is_calculated", {}).get("const", True)
                         await self._initialize_single_line_item_object_from_schema(
                             bs_main_cat_data_ptr, # This is the dict for balance_sheet_check
                             bs_main_cat_schema_effective, 
                             historical_periods, 
                             parsed_doc_content, 
                             current_path_prefix_check,
                             is_direct_call=item_is_calculated_check # True since balance_sheet_check is calculated
                         )

            elif stmt_key == "cash_flow_statement":
                cfs_properties_schema = stmt_schema_effective.get("properties", {})
                for cfs_section_key, cfs_section_schema_orig in cfs_properties_schema.items():
                    # cfs_section_key e.g., "cash_flow_from_operations_cfo", "net_change_in_cash"
                    if not isinstance(cfs_section_schema_orig, dict) or cfs_section_key == "ai_instructions_statement": continue

                    cfs_section_schema_effective = self._get_effective_item_schema(cfs_section_schema_orig)
                    current_path_prefix = f"financial_statements_core.{stmt_key}.{cfs_section_key}"
                    
                    if cfs_section_schema_effective.get("type") == "array": # e.g. "cash_flow_from_operations_cfo" (array of line items)
                        cfs_section_data_list_ptr = stmt_data_model_ptr.setdefault(cfs_section_key, [])
                        if not isinstance(cfs_section_data_list_ptr, list):
                             cfs_section_data_list_ptr = []; stmt_data_model_ptr[cfs_section_key] = cfs_section_data_list_ptr
                        
                        temp_list_wrapper = {"line_items": cfs_section_data_list_ptr}
                        await self._process_statement_section(
                            temp_list_wrapper, cfs_section_schema_effective,
                            historical_periods, parsed_doc_content, current_path_prefix
                        )
                        stmt_data_model_ptr[cfs_section_key] = temp_list_wrapper["line_items"]

                    elif cfs_section_schema_effective.get("type") == "object": # e.g. "net_change_in_cash" (single line item object)
                        single_item_data_ptr = stmt_data_model_ptr.setdefault(cfs_section_key, {})
                        await self._initialize_single_line_item_object_from_schema(single_item_data_ptr, cfs_section_schema_effective, historical_periods, parsed_doc_content, current_path_prefix, is_direct_call=False) # is_direct_call=False if extraction needed

        self.logger.info("Phase A: Historical Data Extraction finished.")
        return self.financial_model, self.extraction_log

    async def _initialize_single_line_item_object_from_schema(self, line_item_data_ptr: Dict[str,Any], item_schema: Dict[str, Any], historical_periods: List[str], parsed_doc_content: Dict[str, Any], path_prefix: str, is_direct_call: bool):
        """ Helper to initialize and potentially extract data for a single line item object (not in an array)."""
        item_props = item_schema.get("properties", {})
        item_name = item_props.get("name", {}).get("const", item_props.get("name", {}).get("default", path_prefix.split('.')[-1]))
        is_calculated = item_props.get("is_calculated", {}).get("const", item_props.get("is_calculated", {}).get("default", False))

        line_item_data_ptr["name"] = item_name
        line_item_data_ptr["is_calculated"] = is_calculated
        line_item_data_ptr["periods"] = []
        for static_prop in ["data_type", "calculation_logic_description", "source_guidance_historical", "unit", "target_value", "tolerance"]:
            if static_prop in item_props:
                prop_detail_schema = item_props[static_prop]
                value = prop_detail_schema.get("const", prop_detail_schema.get("default"))
                if value is not None:
                    line_item_data_ptr[static_prop] = value
        
        if is_calculated:
            self.logger.info(f"Setting up calculated single item: {item_name} at {path_prefix}")
            self._log_extraction_success(item_name, "N/A", "Calculated item", status_override="CALCULATED_ITEM")
        elif not is_direct_call: # Only extract if not called directly for a calculated item like balance_sheet_check
            self.logger.info(f"Extracting data for single non-calculated item: {item_name} at {path_prefix}")
            for period in historical_periods:
                extracted_period_data = await self._extract_historical_line_item_data(
                    item_schema, period, parsed_doc_content # Pass the item's own schema
                )
                line_item_data_ptr["periods"].append(extracted_period_data)
        elif is_direct_call and not is_calculated: # Called for an item like "balance_sheet_check" that turned out not to be calculated
             self.logger.warning(f"Item {item_name} at {path_prefix} was expected to be calculated by direct call but schema says is_calculated=False. Extraction not performed by this path.")

    def _resolve_ref(self, ref_path: str) -> Dict[str, Any]:
        if not ref_path.startswith("#/"): raise ValueError(f"Unsupported $ref format: {ref_path}")
        parts = ref_path[2:].split('/')
        current = self.model_schema_definition
        for part in parts:
            if isinstance(current, dict) and part in current: current = current[part]
            else: raise ValueError(f"$ref path not found: {ref_path}")
        return current

    def _get_effective_item_schema(self, item_schema_part: Dict[str, Any]) -> Dict[str, Any]:
        """Resolves $ref and merges allOf to get the effective schema properties for an item."""
        if "$ref" in item_schema_part:
            base_schema = self._resolve_ref(item_schema_part["$ref"])
            # If the item_schema_part itself has other properties, they override/extend the $ref base
            # This simple merge won't handle deep merging of properties, but is often sufficient
            effective_schema = {**base_schema, **{k:v for k,v in item_schema_part.items() if k != "$ref"}}
            return self._get_effective_item_schema(effective_schema) # Recurse in case $ref pointed to another $ref/allOf

        if "allOf" in item_schema_part:
            merged_properties = {}
            # Collect all properties from allOf parts
            for sub_schema_wrapper in item_schema_part["allOf"]:
                # Resolve sub-schema which might itself be a $ref or have allOf
                resolved_sub_schema = self._get_effective_item_schema(sub_schema_wrapper)
                if "properties" in resolved_sub_schema:
                    merged_properties.update(resolved_sub_schema["properties"])
            
            # Override with properties directly in item_schema_part, if any
            if "properties" in item_schema_part:
                merged_properties.update(item_schema_part["properties"])
            
            # Construct the final schema using a base type (usually object) and merged properties
            # Also copy other keywords like 'type', 'required', 'description' from the main item_schema_part
            final_schema = {k: v for k, v in item_schema_part.items() if k not in ["allOf", "properties"]}
            final_schema["properties"] = merged_properties
            if "type" not in final_schema: final_schema["type"] = "object" # Assume object if not specified
            return final_schema
        
        return item_schema_part # No $ref or allOf at this level

    def _initialize_model_from_schema_recursive(self, schema_part: Dict[str, Any], data_part: Union[Dict, List]):
        schema_part = self._get_effective_item_schema(schema_part) # Resolve $ref/allOf first
        schema_type = schema_part.get("type")

        if schema_type == "object":
            if not isinstance(data_part, dict): # Should not happen if called correctly
                self.logger.error(f"Type mismatch: Expected dict for data_part, got {type(data_part)} for schema {schema_part.get('title', 'N/A')}")
                return

            for prop_name, prop_schema_orig in schema_part.get("properties", {}).items():
                prop_schema = self._get_effective_item_schema(prop_schema_orig)
                default_value = prop_schema.get("default")
                prop_type = prop_schema.get("type")

                if prop_type == "object":
                    data_part[prop_name] = {}
                    self._initialize_model_from_schema_recursive(prop_schema, data_part[prop_name])
                elif prop_type == "array":
                    data_part[prop_name] = []
                    # If 'items' is an array (tuple validation), initialize based on that structure
                    if isinstance(prop_schema.get("items"), list) and prop_schema.get("items"):
                        for sub_item_schema_orig in prop_schema["items"]:
                            sub_item_schema = self._get_effective_item_schema(sub_item_schema_orig)
                            if sub_item_schema.get("type") == "object":
                                new_obj = {}
                                # Populate with const/default names from schema for specific line items
                                if "properties" in sub_item_schema and "name" in sub_item_schema["properties"]:
                                    name_schema = sub_item_schema["properties"]["name"]
                                    new_obj["name"] = name_schema.get("const", name_schema.get("default", "Unknown Line Item"))
                                # Initialize other fields based on sub_item_schema
                                self._initialize_model_from_schema_recursive(sub_item_schema, new_obj)
                                data_part[prop_name].append(new_obj)
                            # Add handling for other types if arrays can contain non-objects
                    elif default_value is not None: # Use default if provided for array (e.g. empty list)
                         data_part[prop_name] = list(default_value) # Ensure it's a new list
                    # else, it remains an empty list initialized above
                else: # Primitives
                    data_part[prop_name] = default_value # which can be None
        
        elif schema_type == "array": # Top level is array, usually not for the whole model
            if not isinstance(data_part, list): return # Error or skip
            # Similar logic as above if schema_part["items"] is complex

    async def _call_llm_with_retry(self, base_prompt_key: str, prompt_args: Dict[str, Any], response_model: Optional[Type[BaseModel]] = None, max_attempts: int = 3) -> Optional[Dict[str, Any]]:
        # (Identical to the corrected version from your last successful run with _LINE_ITEM_DEFINITIONS)
        current_prompt_key = base_prompt_key
        for attempt in range(1, max_attempts + 1):
            item_name_for_log = prompt_args.get('line_item_english_name', prompt_args.get('section_hint', 'N/A'))
            period_for_log = prompt_args.get('period_label', 'N/A')
            self.logger.info(f"LLM call attempt {attempt}/{max_attempts} for: {item_name_for_log} (Period: {period_for_log}) using prompt key: '{current_prompt_key}'")
            prompt_template_obj = self.prompts.get(current_prompt_key); prompt_template = prompt_template_obj.get("prompt_template") if prompt_template_obj else None
            if not prompt_template: self.logger.error(f"Prompt template for key '{current_prompt_key}' not found."); return None
            try: formatted_prompt = prompt_template.format(**prompt_args)
            except KeyError as e: self.logger.error(f"Missing key for template '{current_prompt_key}': {e}. Args: {prompt_args}"); return None
            self.logger.debug(f"Formatted prompt (attempt {attempt}):\n{formatted_prompt[:1000]}...")
            try:
                raw_response = await self.llm_service.invoke(prompt=formatted_prompt, model_name="gemini-1.5-flash-latest", json_mode=True)
                self.logger.debug(f"Raw LLM response (attempt {attempt}):\n{raw_response[:1000]}...")
            except Exception as e:
                self.logger.error(f"LLM invocation failed (attempt {attempt}): {e}", exc_info=True)
                if attempt == max_attempts: return None
                await asyncio.sleep(1 + attempt); current_prompt_key = base_prompt_key; continue
            try:
                cleaned_response_str = raw_response.strip()
                if cleaned_response_str.startswith("```json"): cleaned_response_str = cleaned_response_str[len("```json"):].strip()
                if cleaned_response_str.endswith("```"): cleaned_response_str = cleaned_response_str[:-len("```")].strip()
                if not cleaned_response_str: self.logger.warning(f"LLM response empty (attempt {attempt})."); raise json.JSONDecodeError("Empty response", "", 0)
                parsed_json = json.loads(cleaned_response_str)
                if response_model: response_model(**parsed_json)
                self.logger.info(f"Successfully parsed & validated LLM JSON for {item_name_for_log} (attempt {attempt}).")
                return parsed_json
            except json.JSONDecodeError as e_json: self.logger.warning(f"Failed to decode JSON for {item_name_for_log} (attempt {attempt}): {e_json}. Response: {cleaned_response_str[:200]}"); current_prompt_key = "re_prompt_invalid_json"
            except ValidationError as e_val: self.logger.warning(f"Pydantic validation failed for {item_name_for_log} (attempt {attempt}): {e_val}"); current_prompt_key = "re_prompt_missing_keys"
            except Exception as e_parse: self.logger.error(f"Unexpected error parsing for {item_name_for_log} (attempt {attempt}): {e_parse}", exc_info=True); current_prompt_key = base_prompt_key
            if attempt == max_attempts: return None
            await asyncio.sleep(attempt)
        return None

    def _get_context_for_item(self, parsed_document_content: Dict[str, Any], item_name_english: Optional[str] = None, item_name_mongolian: Optional[str] = None, period_label: Optional[str] = None, section_hint: Optional[str] = None, max_context_length: int = 15000) -> str:
        # (Identical to corrected version from previous iteration - ensure it's suitable)
        log_item_name = item_name_english or section_hint or "Unknown Item"
        self.logger.debug(f"Gathering context for: Item='{log_item_name}', Period='{period_label or 'N/A'}'")
        context_parts = []
        num_pages_to_scan = min(parsed_document_content.get("processed_pages_count", len(parsed_document_content.get("pages_content",[]))), 25)
        keywords = [kw.lower() for kw in [item_name_english, item_name_mongolian, period_label, section_hint] if kw and isinstance(kw, str)]
        for i in range(num_pages_to_scan):
            if i >= len(parsed_document_content.get("pages_content", [])): break
            page_data = parsed_document_content["pages_content"][i]; page_text_parts = []
            if page_data.get("text_pymupdf"): page_text_parts.append(page_data["text_pymupdf"])
            if page_data.get("page_full_ocr_text"): page_text_parts.append(page_data["page_full_ocr_text"])
            page_text_content = "\n".join(page_text_parts)
            take_page = i < 5 or (keywords and any(kw in page_text_content.lower() for kw in keywords))
            if take_page: context_parts.append(f"\n--- Page {page_data['page_number']} ---\n{page_text_content[:2000]}")
        for table_data in parsed_document_content.get("all_tables", [])[:15]:
            table_repr = f"\n--- Page {table_data['page_number']}, Table {table_data['table_index_on_page']} ---\n"; table_str = ""
            for r_idx, row in enumerate(table_data.get("rows",[])): table_str += f"R{r_idx}: {' | '.join([str(c) if c is not None else '' for c in row])}\n"
            take_table = table_data['page_number'] <= 5 or (keywords and any(kw in table_str.lower() for kw in keywords))
            if take_table: context_parts.append(table_repr + table_str[:1500])
        for img_ocr in parsed_document_content.get("all_images_ocr_results",[])[:15]:
            if img_ocr.get("ocr_text"):
                img_text = img_ocr["ocr_text"]
                take_img = img_ocr['page_number'] <= 5 or (keywords and any(kw in img_text.lower() for kw in keywords))
                if take_img: context_parts.append(f"\n--- Page {img_ocr['page_number']}, ImgOCR ---\n{img_text[:1000]}")
        combined_context = "".join(context_parts) or "No specific context found."
        if len(combined_context) > max_context_length: self.logger.warning(f"Context for '{log_item_name}' truncated."); combined_context = combined_context[:max_context_length]
        self.logger.debug(f"Context for '{log_item_name}' (len {len(combined_context)}):\n{combined_context[:200]}...")
        return combined_context

    async def _extract_metadata(self, parsed_doc_content: Dict[str, Any], document_type: str):
        self.logger.info("Extracting model metadata...")
        model_meta_data_ptr = self.financial_model.setdefault("model_metadata", {})
        context = self._get_context_for_item(parsed_doc_content, section_hint="company identification, currency, fiscal year", max_context_length=5000)
        prompt_args = {"context": context, "document_type": document_type}
        metadata_json = await self._call_llm_with_retry("metadata_extraction", prompt_args, response_model=ExtractedMetadata)
        if metadata_json: model_meta_data_ptr.update(metadata_json); self._log_extraction_success("Model Metadata", "N/A", metadata_json)
        else: self.logger.warning("Failed to extract model metadata."); self._log_extraction_failure("Model Metadata", "N/A", "LLM failed")

    async def _identify_historical_periods(self, parsed_doc_content: Dict[str, Any], document_type: str) -> List[str]:
        self.logger.info("Identifying historical periods...")
        fsc_ptr = self.financial_model.setdefault("financial_statements_core", {})
        current_labels = fsc_ptr.setdefault("historical_period_labels", [])
        context = self._get_context_for_item(parsed_doc_content, section_hint="financial statements column headers years periods", max_context_length=10000)
        prompt_args = {"context": context, "document_type": document_type}
        periods_json = await self._call_llm_with_retry("historical_period_identification", prompt_args, response_model=IdentifiedPeriods)
        identified_periods = []
        if periods_json and isinstance(periods_json.get("historical_period_labels"), list):
            identified_periods = periods_json["historical_period_labels"]
            self.logger.info(f"Identified historical periods: {identified_periods}")
            fsc_ptr["historical_period_labels"] = identified_periods # Override previous default
            self._log_extraction_success("Historical Period Labels", "N/A", identified_periods)
        else:
            self.logger.warning(f"Failed to identify historical periods. JSON: {periods_json}")
            self._log_extraction_failure("Historical Period Labels", "N/A", f"LLM failed. Got: {periods_json}")
        return identified_periods

    def _get_line_item_mongolian_name(self, item_schema: Dict[str, Any]) -> str: # Now takes full item schema
        # Try to get from a specific property if added to schema, e.g. "title_mn"
        return item_schema.get("properties", {}).get("name_mn", {}).get("const",
               item_schema.get("properties", {}).get("name", {}).get("const", "Unknown Line Item"))


    async def _extract_historical_line_item_data(
        self, item_schema: Dict[str, Any], period_label: str, parsed_doc_content: Dict[str, Any]
    ) -> Dict[str, Any]: # Returns a period_object_value_only structure
        item_props = self._get_effective_item_schema(item_schema).get("properties", {})
        item_name_en = item_props.get("name", {}).get("const", item_props.get("name", {}).get("default", "Unknown Item"))
        item_name_mn = self._get_line_item_mongolian_name(item_schema) # Use effective schema here too
        item_guidance = item_props.get("source_guidance_historical", {}).get("default", "Refer to standard financial definitions.")
        
        self.logger.info(f"Extracting: '{item_name_en}' for period '{period_label}'")
        context = self._get_context_for_item(parsed_doc_content, item_name_english=item_name_en, item_name_mongolian=item_name_mn, period_label=period_label, section_hint=f"{item_name_en} value")
        prompt_args = {"period_label": period_label, "line_item_english_name": item_name_en, "line_item_mongolian_name": item_name_mn, "line_item_definition": item_guidance, "context": context}
        extracted_json = await self._call_llm_with_retry("single_line_item_extraction", prompt_args, response_model=ExtractedLineItemDetail)
        
        output_period_data = {"period_label": period_label, "value": None, "source_reference": None, "extraction_status": "FAILURE_UNKNOWN"}
        if extracted_json:
            try:
                validated = ExtractedLineItemDetail(**extracted_json)
                output_period_data["source_reference"] = validated.source_reference
                if validated.status == "EXTRACTED_SUCCESSFULLY":
                    norm_val, _, _ = self.utils.normalize_value(validated.value, validated.unit, validated.currency)
                    output_period_data["value"] = norm_val
                    output_period_data["extraction_status"] = "SUCCESS"
                    self._log_extraction_success(item_name_en, period_label, validated.model_dump())
                elif validated.status == "CONFIRMED_NOT_FOUND":
                    output_period_data["extraction_status"] = "NOT_FOUND_PER_LLM"
                    self._log_extraction_success(item_name_en, period_label, validated.model_dump(), status_override="CONFIRMED_NOT_FOUND")
                else: output_period_data["extraction_status"] = f"FAILURE_UNEXPECTED_LLM_STATUS:{validated.status}"; self._log_extraction_failure(item_name_en, period_label, f"Unexpected LLM status: {validated.status}", validated.model_dump())
            except Exception as e: output_period_data["extraction_status"] = "FAILURE_PROCESSING_ERROR"; self.logger.error(f"Error processing {item_name_en}/{period_label}: {e}", exc_info=True); self._log_extraction_failure(item_name_en, period_label, f"Processing error: {e}", extracted_json)
        else: output_period_data["extraction_status"] = "FAILURE_MAX_ATTEMPTS"; self.logger.warning(f"Failed extraction for '{item_name_en}'/'{period_label}'."); self._log_extraction_failure(item_name_en, period_label, "LLM failed")
        return output_period_data

    def _log_extraction_success(self, item:str, period:str, data:Any, status_override:Optional[str]=None): self.extraction_log.append({"item_name":item,"period":period,"status":status_override or "SUCCESS","data":data})
    def _log_extraction_failure(self, item:str, period:str, reason:str, raw:Optional[Any]=None): self.extraction_log.append({"item_name":item,"period":period,"status":"FAILURE","reason":reason,"raw_data":raw})

# In class PhaseAOrchestrator:

    async def _process_statement_section(self,
                                         # section_data_model_container is the dict in financial_model
                                         # that *contains* (or will contain) the "line_items" list.
                                         # E.g., model[...]["income_statement"]
                                         section_data_model_container: Dict[str, Any], 
                                         # section_schema is the schema for the array property itself
                                         # (e.g., schema for "income_statement.line_items" or "current_assets")
                                         section_array_schema: Dict[str, Any], 
                                         historical_periods: List[str],
                                         parsed_doc_content: Dict[str, Any],
                                         path_prefix: str):
        
        # section_array_schema is the schema for the array property itself (e.g., for 'line_items' under income_statement,
        # or for 'current_assets' array directly).
        # We need the "items" part of this array schema, which should be a list for tuple validation.
        line_items_schema_list = section_array_schema.get("items", [])
        
        if not isinstance(line_items_schema_list, list):
            self.logger.warning(f"In _process_statement_section for '{path_prefix}', "
                                f"expected 'items' in section_array_schema to be a list of schemas (tuple validation). "
                                f"Got type: {type(line_items_schema_list)}. Skipping this section.")
            # Ensure the container in the model at least has an empty line_items list if it was expected.
            # If section_array_schema was intended to be an array (e.g. current_assets directly),
            # then section_data_model_container is ALREADY that list.
            # This logic needs to be careful.
            # Let's assume the caller ensures section_data_model_container is the dict where 'line_items' lives, OR
            # if it's a direct list, it's handled correctly by the caller.
            # The temp_list_wrapper in run_phase_a_extraction for BS direct arrays aims to standardize this.
            if "line_items" in section_data_model_container: # If it was an object like income_statement
                 section_data_model_container.setdefault("line_items", [])
            return

        # target_line_items_list_in_model is the actual list in self.financial_model
        # For Income Statement: section_data_model_container is model[...]["income_statement"], so this gets/creates its "line_items" list.
        # For BS current_assets (via temp_list_wrapper): section_data_model_container is {"line_items": model[...]["current_assets"]}, so this gets that list.
        target_line_items_list_in_model = section_data_model_container.setdefault("line_items", [])
        target_line_items_list_in_model.clear() 

        for item_idx, specific_item_schema_orig in enumerate(line_items_schema_list):
            item_schema_effective = self._get_effective_item_schema(specific_item_schema_orig)
            item_props = item_schema_effective.get("properties", {})
            
            item_name = item_props.get("name", {}).get("const", item_props.get("name", {}).get("default", f"UnnamedItem_{path_prefix}_{item_idx}"))
            is_calculated = item_props.get("is_calculated", {}).get("const", item_props.get("is_calculated", {}).get("default", False))

            current_line_item_data_for_model = {
                "name": item_name, "is_calculated": is_calculated, "periods": [],
                "data_type": item_props.get("data_type", {}).get("default", "currency_value"),
                "calculation_logic_description": item_props.get("calculation_logic_description", {}).get("default"),
                "source_guidance_historical": item_props.get("source_guidance_historical", {}).get("default"),
                "ai_instructions": item_props.get("ai_instructions", {}).get("default"),
                "ai_instructions_projected": item_props.get("ai_instructions_projected", {}).get("default"),
                "notes": item_props.get("notes", {}).get("default"),
                "unit": item_props.get("unit", {}).get("default")
            }
            # Remove keys with None values to keep model clean if default was None
            current_line_item_data_for_model = {k:v for k,v in current_line_item_data_for_model.items() if v is not None}


            if is_calculated:
                self.logger.info(f"Skipping extraction for calculated item: {item_name} in {path_prefix}")
                self._log_extraction_success(item_name, "N/A", "Calculated item", status_override="CALCULATED_ITEM")
            else:
                for period in historical_periods:
                    extracted_period_data = await self._extract_historical_line_item_data(
                        item_schema_effective, period, parsed_doc_content # Pass effective schema of the item
                    )
                    current_line_item_data_for_model["periods"].append(extracted_period_data)
            
            target_line_items_list_in_model.append(current_line_item_data_for_model)

    async def run_phase_a_extraction(self, parsed_doc_content: Dict[str, Any], document_type_hint: str = "Financial Report") -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        self.logger.info("Starting Phase A: Historical Data Extraction (Schema-Driven v2.2).")
        await self._extract_metadata(parsed_doc_content, document_type_hint)
        historical_periods = await self._identify_historical_periods(parsed_doc_content, document_type_hint)
        
        if not historical_periods:
            self.logger.error("No historical periods identified. Aborting line item extraction.")
            return self.financial_model, self.extraction_log

        # Ensure financial_statements_core and its historical_period_labels are initialized
        fsc_data_model_ptr = self.financial_model.setdefault("financial_statements_core", {})
        fsc_data_model_ptr["historical_period_labels"] = historical_periods # Override default with identified ones

# In PhaseAOrchestrator.run_phase_a_extraction:

        # ...
        fsc_schema = self._get_schema_property(["financial_statements_core"]) 

        if not fsc_schema: # fsc_schema is the "properties" dict of financial_statements_core
            self.logger.error("Schema for 'financial_statements_core.properties' not found or is empty. Cannot process statements.")
            return self.financial_model, self.extraction_log

        for stmt_key, stmt_schema_orig in fsc_schema.items():
            # stmt_key is "income_statement", "balance_sheet", "cash_flow_statement", "section_description", etc.
            # stmt_schema_orig is the schema definition for that specific property.

            # --- ADD THIS CHECK ---
            if not isinstance(stmt_schema_orig, dict):
                self.logger.debug(f"Skipping non-dict schema entry '{stmt_key}' under financial_statements_core.properties (value: {stmt_schema_orig})")
                continue
            # --- END ADDED CHECK ---
            
            if stmt_key in ["historical_period_labels", "projected_period_labels", "section_description"]:
                # These are handled differently or are just descriptive.
                # Ensure ai_instructions_statement is copied if present in schema (already handled by _initialize_model_from_schema_recursive if it has a default)
                # self.financial_model["financial_statements_core"].setdefault(stmt_key, stmt_schema_orig.get("default")) # Initialize if not already
                continue 
            
            # Check for ai_instructions_statement separately, as it's a property of the statement objects, not a statement itself
            if stmt_key == "ai_instructions_statement": # This property is inside income_statement etc. not directly under fsc_schema.items()
                continue


            self.logger.info(f"Processing statement object property: {stmt_key}") # Changed log message for clarity
            stmt_schema_effective = self._get_effective_item_schema(stmt_schema_orig)
            stmt_data_model_ptr = fsc_data_model_ptr.setdefault(stmt_key, {})

            if stmt_schema_effective.get("type") != "object":
                self.logger.warning(f"Schema for property '{stmt_key}' is not an object type as expected for a statement structure. Type is '{stmt_schema_effective.get('type')}'. Skipping detailed processing for this key.")
                continue
                        
            # Case 1: Statement object directly contains a "line_items" array property (e.g., Income Statement)
            if "line_items" in stmt_schema_effective.get("properties", {}):
                line_items_array_schema = stmt_schema_effective["properties"]["line_items"]
                # Pass the schema for the line_items array itself to _process_statement_section
                await self._process_statement_section(
                    stmt_data_model_ptr, # This is the dict that should contain 'line_items' list
                    line_items_array_schema, 
                    historical_periods, 
                    parsed_doc_content,
                    f"financial_statements_core.{stmt_key}"
                )
            # Case 2: Statement object contains nested structures like Balance Sheet ("assets", "liabilities_and_equity")
            elif stmt_key == "balance_sheet":
                bs_properties_schema = stmt_schema_effective.get("properties", {})
                for bs_main_category_key, bs_main_category_schema_orig in bs_properties_schema.items():
                    # bs_main_category_key is "assets", "liabilities_and_equity", "balance_sheet_check"
                    if bs_main_category_key == "ai_instructions_statement": continue
                    
                    bs_main_category_schema_effective = self._get_effective_item_schema(bs_main_category_schema_orig)
                    # bs_main_cat_data_ptr points to e.g., self.financial_model[...]["balance_sheet"]["assets"]
                    bs_main_cat_data_ptr = stmt_data_model_ptr.setdefault(bs_main_category_key, {})

                    if bs_main_category_schema_effective.get("type") == "object" and "properties" in bs_main_category_schema_effective:
                        # This is for "assets" and "liabilities_and_equity" objects
                        for bs_sub_category_key, bs_sub_category_schema_orig in bs_main_category_schema_effective.get("properties", {}).items():
                            # bs_sub_category_key is "current_assets", "non_current_assets", "total_assets", etc.
                            bs_sub_category_schema_effective = self._get_effective_item_schema(bs_sub_category_schema_orig)
                            
                            # Path for logging/debugging
                            current_path_prefix = f"financial_statements_core.{stmt_key}.{bs_main_category_key}.{bs_sub_category_key}"

                            if bs_sub_category_schema_effective.get("type") == "array":
                                # This is for arrays like "current_assets", "non_current_assets"
                                # bs_sub_cat_data_list_ptr points to the list itself, e.g., model[...]["assets"]["current_assets"]
                                bs_sub_cat_data_list_ptr = bs_main_cat_data_ptr.setdefault(bs_sub_category_key, [])
                                if not isinstance(bs_sub_cat_data_list_ptr, list): # Ensure it's a list
                                    self.logger.warning(f"Correcting type for {current_path_prefix} to list in model.")
                                    bs_sub_cat_data_list_ptr = []
                                    bs_main_cat_data_ptr[bs_sub_category_key] = bs_sub_cat_data_list_ptr
                                
                                # We pass a temporary dict wrapper because _process_statement_section expects to set a 'line_items' key
                                temp_list_wrapper = {"line_items": bs_sub_cat_data_list_ptr}
                                await self._process_statement_section(
                                    temp_list_wrapper, # The dict that _process_statement_section will operate on
                                    bs_sub_category_schema_effective, # Schema for the array (e.g., "current_assets")
                                    historical_periods,
                                    parsed_doc_content,
                                    current_path_prefix
                                )
                                # Update the original list in the model with the processed list
                                bs_main_cat_data_ptr[bs_sub_category_key] = temp_list_wrapper["line_items"]

                            elif bs_sub_category_schema_effective.get("type") == "object":
                                # This is for single objects like "total_assets", "total_liabilities", "balance_sheet_check"
                                # These are single line items, not arrays of them.
                                single_item_data_ptr = bs_main_cat_data_ptr.setdefault(bs_sub_category_key, {})
                                item_props = bs_sub_category_schema_effective.get("properties", {})
                                item_name = item_props.get("name", {}).get("const", item_props.get("name", {}).get("default", bs_sub_category_key))
                                is_calculated = item_props.get("is_calculated", {}).get("const", item_props.get("is_calculated", {}).get("default", False))

                                single_item_data_ptr["name"] = item_name
                                single_item_data_ptr["is_calculated"] = is_calculated
                                single_item_data_ptr["periods"] = [] # Initialize periods array
                                # Populate other static fields from schema
                                for static_prop in ["data_type", "calculation_logic_description", "source_guidance_historical", "unit"]:
                                     if static_prop in item_props and "default" in item_props[static_prop]:
                                         single_item_data_ptr[static_prop] = item_props[static_prop]["default"]
                                
                                if is_calculated:
                                    self.logger.info(f"Skipping extraction for calculated single item: {item_name} at {current_path_prefix}")
                                    self._log_extraction_success(item_name, "N/A", "Calculated item", status_override="CALCULATED_ITEM")
                                else:
                                    # If a single object item is NOT calculated, we'd extract its periods here
                                    # This logic is similar to _extract_historical_line_item_data but for a single object
                                    self.logger.warning(f"Non-calculated single object item '{item_name}' at {current_path_prefix}. Extraction for such direct objects needs explicit handling if required beyond calculated flags.")
                                    # For now, just log it. If extraction needed, call a modified _extract_historical_line_item_data
                                    for period in historical_periods:
                                        extracted_period_data = await self._extract_historical_line_item_data(
                                            bs_sub_category_schema_effective, # Pass the item's own schema
                                            period, 
                                            parsed_doc_content
                                        )
                                        single_item_data_ptr["periods"].append(extracted_period_data)
                    elif bs_main_category_schema_effective.get("type") == "object" and bs_main_category_key == "balance_sheet_check":
                        # Handle balance_sheet_check specifically as it's a direct object not a category of arrays
                        check_item_data_ptr = bs_main_cat_data_ptr # It's the dict for balance_sheet_check
                        check_item_props = bs_main_category_schema_effective.get("properties", {})
                        check_item_data_ptr["name"] = check_item_props.get("name", {}).get("const", "Balance Sheet Check")
                        check_item_data_ptr["is_calculated"] = check_item_props.get("is_calculated", {}).get("const", True)
                        check_item_data_ptr["periods"] = []
                        self.logger.info(f"Setting up calculated item: {check_item_data_ptr['name']}")
                        self._log_extraction_success(check_item_data_ptr['name'], "N/A", "Calculated item", status_override="CALCULATED_ITEM")


            # Case 3: Cash Flow Statement (has nested objects which then contain line_items arrays)
            elif stmt_key == "cash_flow_statement":
                cfs_properties_schema = stmt_schema_effective.get("properties", {})
                for cfs_section_key, cfs_section_schema_orig in cfs_properties_schema.items():
                    # cfs_section_key e.g., "cash_flow_from_operations_cfo", "net_change_in_cash"
                    if cfs_section_key == "ai_instructions_statement": continue

                    cfs_section_schema_effective = self._get_effective_item_schema(cfs_section_schema_orig)
                    cfs_section_data_model_ptr = stmt_data_model_ptr.setdefault(cfs_section_key, {})
                    current_path_prefix = f"financial_statements_core.{stmt_key}.{cfs_section_key}"

                    if cfs_section_schema_effective.get("type") == "array": # e.g. cash_flow_from_operations_cfo
                        # This is an array of line items
                        temp_list_wrapper = {"line_items": cfs_section_data_model_ptr} # if schema makes cfs_section_data_model_ptr the list directly
                        if not isinstance(cfs_section_data_model_ptr, list): # If schema made it an object with a line_items list inside
                            cfs_section_data_model_ptr.setdefault("line_items", [])
                            temp_list_wrapper = cfs_section_data_model_ptr


                        await self._process_statement_section(
                            temp_list_wrapper, 
                            cfs_section_schema_effective, # Schema for the array itself
                            historical_periods, 
                            parsed_doc_content,
                            current_path_prefix
                        )
                        if isinstance(stmt_data_model_ptr.get(cfs_section_key), dict) and "line_items" in temp_list_wrapper : # if it was wrapped
                             stmt_data_model_ptr[cfs_section_key] = temp_list_wrapper["line_items"] # unwrap if needed
                        elif isinstance(stmt_data_model_ptr.get(cfs_section_key), list): # if it was direct list
                             pass # it's already updated
                             

                    elif cfs_section_schema_effective.get("type") == "object":
                        # For single objects like "net_change_in_cash"
                        item_props = cfs_section_schema_effective.get("properties", {})
                        item_name = item_props.get("name", {}).get("const", cfs_section_key)
                        is_calculated = item_props.get("is_calculated", {}).get("const", True)
                        
                        cfs_section_data_model_ptr["name"] = item_name
                        cfs_section_data_model_ptr["is_calculated"] = is_calculated
                        cfs_section_data_model_ptr["periods"] = []
                        # Populate other static fields
                        for static_prop in ["data_type", "calculation_logic_description", "unit"]:
                             if static_prop in item_props and ("default" in item_props[static_prop] or "const" in item_props[static_prop]):
                                 cfs_section_data_model_ptr[static_prop] = item_props[static_prop].get("const", item_props[static_prop].get("default"))

                        if is_calculated:
                            self.logger.info(f"Skipping extraction for calculated CFS item: {item_name}")
                            self._log_extraction_success(item_name, "N/A", "Calculated item", status_override="CALCULATED_ITEM")
                        else:
                             self.logger.warning(f"Non-calculated single object item '{item_name}' in CFS. Extraction logic needs extension if required.")
                             for period in historical_periods:
                                extracted_period_data = await self._extract_historical_line_item_data(
                                    cfs_section_schema_effective, period, parsed_doc_content
                                )
                                cfs_section_data_model_ptr["periods"].append(extracted_period_data)


        self.logger.info("Phase A: Historical Data Extraction finished.")
        return self.financial_model, self.extraction_log

# --- Standalone Test (Updated Mock and Structure) ---
class MockLlmService(BaseLlmService):
    async def invoke(self, prompt: str, **kwargs) -> str:
        self.log_call(prompt) # Ensure this method exists or add it
        if "extract the following metadata" in prompt.lower(): return json.dumps({"target_company_name": "Mock Company Монгол ХХК", "ticker_symbol": "MCK", "currency": "MNT", "fiscal_year_end": "12-31"})
        if "identify the distinct historical financial periods" in prompt.lower(): return json.dumps({"historical_period_labels": ["FY2023", "FY2022"]})
        if "extract a specific financial data point" in prompt.lower():
            def get_val(item, period):
                if item=="Revenue" and period=="FY2023": return 125000000
                if item=="Cost of Goods Sold (COGS)" and period=="FY2023": return 80000000
                if item=="SG&A Expenses" and period=="FY2023": return 15000000
                if item=="Revenue" and period=="FY2022": return 100000000
                # Add more mock data here for other line items as needed by your schema
                return 500000 # Default small value for other items
            
            for line_item_name in ["Revenue", "Cost of Goods Sold (COGS)", "SG&A Expenses", "Depreciation & Amortization (D&A)", "Interest Expense", "Interest Income", "Income Tax Expense", "Cash & Cash Equivalents", "Accounts Receivable", "Inventory", "Net PP&E", "Goodwill", "Intangible Assets (Net)", "Accounts Payable", "Short-Term Debt & Current Portion of LTD", "Long-Term Debt", "Common Stock & APIC", "Retained Earnings"]:
                if line_item_name in prompt:
                    for p_label in ["FY2023", "FY2022"]:
                        if p_label in prompt:
                            return json.dumps({"value": get_val(line_item_name, p_label), "currency": "MNT", "unit": "actuals", "source_reference": f"Mock {line_item_name} PXX", "status": "EXTRACTED_SUCCESSFULLY"})
            return json.dumps({"value": None, "source_reference": "Mock Default Not Found", "status": "CONFIRMED_NOT_FOUND"}) # Default to not found for unmocked items
        return json.dumps({"error": "Mock LLM: Unknown prompt content for structured response"})
    def log_call(self, prompt): print(f"\n--- Mock LLM Call ---\n{prompt[:500] + ('...' if len(prompt)>500 else '')}\n--- End Mock LLM Call ---\n")

async def _orchestrator_standalone_test():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - [%(levelname)s] - %(name)s - %(module)s:%(lineno)d - %(message)s')
    # logging.getLogger("__main__.PhaseAOrchestrator").setLevel(logging.DEBUG) # More specific debug
    
    current_dir = Path(__file__).parent.parent.parent
    schema_p = current_dir / "fund" / "financial_model_schema.json" # Should be the v2.1 schema
    prompts_p = current_dir / "prompts" / "financial_modeling_prompts.yaml"
    log_p = current_dir / "logs" / "financial_modeling_orchestrator_v2_1_test.log"
    if not schema_p.exists() or not prompts_p.exists(): print(f"Error: Schema or Prompts missing. Check paths."); return

    mock_llm = MockLlmService()
    mock_utils = FinancialModelingUtils()
    orchestrator = PhaseAOrchestrator(schema_p, prompts_p, mock_llm, mock_utils, log_p)
    
    mock_parsed_content = {
        "status": "success", "file_name": "dummy.pdf", "total_pages": 2, "processed_pages_count": 2,
        "pages_content": [{"page_number": 1, "text_pymupdf": "Cover Page Info..."}, {"page_number": 2, "text_pymupdf": "Financials..."}],
        "all_tables": [], "all_images_ocr_results": []
    }
    final_model, extraction_summary = await orchestrator.run_phase_a_extraction(mock_parsed_content)
    
    print("\n--- Final Populated Model (Summary) ---")
    print(f"Metadata: {json.dumps(final_model.get('model_metadata'), ensure_ascii=False, indent=2)}")
    fsc = final_model.get("financial_statements_core", {})
    print(f"Historical Periods: {fsc.get('historical_period_labels')}")
    is_items = fsc.get("income_statement", {}).get("line_items", [])
    print(f"Income Statement Items ({len(is_items)}):")
    for item in is_items[:3]: print(f"  - {item.get('name')}: {json.dumps(item.get('periods'), ensure_ascii=False, indent=2)}")
    if len(is_items) > 3: print(f"    ... and {len(is_items)-3} more IS items.")

    bs_assets_ca = fsc.get("balance_sheet", {}).get("assets", {}).get("current_assets", [])
    print(f"BS Current Assets Items ({len(bs_assets_ca)}):")
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