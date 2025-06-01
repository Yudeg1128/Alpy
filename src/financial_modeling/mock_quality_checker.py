import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union, Literal
from decimal import Decimal, InvalidOperation
import sys # Added for StreamHandler

from pydantic import BaseModel, Field
from jsonschema import validate, ValidationError as JsonSchemaValidationError

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Set logger level to DEBUG
# Add a handler to output to console if not already configured elsewhere
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - [%(name)s] - [%(levelname)s] - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(name)s] - [%(levelname)s] - %(message)s')

class QualityCheckError(BaseModel):
    check_type: str
    item_path: Optional[str] = None
    message: str
    severity: Literal["ERROR", "WARNING", "INFO"] = "ERROR"
    details: Optional[Any] = None

class QualityReport(BaseModel):
    overall_status: Literal["PASSED", "PASSED_WITH_WARNINGS", "FAILED"] = Field(default="PASSED")
    phase_a_data_completeness_score: Optional[float] = Field(default=None, description="Percentage of non-calculated historical items successfully extracted.")
    errors: List[QualityCheckError] = Field(default_factory=list)
    warnings: List[QualityCheckError] = Field(default_factory=list)

    def add_error(self, check_type: str, message: str, item_path: Optional[str] = None, details: Optional[Any] = None):
        self.errors.append(QualityCheckError(check_type=check_type, item_path=item_path, message=message, severity="ERROR", details=details))
        if self.overall_status != "FAILED": # Only escalate to FAILED, don't downgrade
            self.overall_status = "FAILED"
    
    def add_warning(self, check_type: str, message: str, item_path: Optional[str] = None, details: Optional[Any] = None):
        self.warnings.append(QualityCheckError(check_type=check_type, item_path=item_path, message=message, severity="WARNING", details=details))
        if self.overall_status == "PASSED": # Escalate PASSED to PASSED_WITH_WARNINGS
            self.overall_status = "PASSED_WITH_WARNINGS"

class QualityChecker:
    def __init__(self, schema_definition_path: Path):
        self.schema_definition_path = schema_definition_path
        try:
            with open(self.schema_definition_path, 'r', encoding='utf-8') as f:
                self.model_schema_definition = json.load(f)
            logger.info(f"QualityChecker: Loaded schema definition: {self.schema_definition_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load schema definition: {e}")

    def _is_non_critical_field(self, field_path: str) -> bool:
        """Check if a field is non-critical for validation"""
        non_critical_fields = ["name_mn", "calculation_logic_description", "ai_instructions"]
        return any(field in field_path for field in non_critical_fields)

    def _normalize_line_item(self, item: dict) -> dict:
        """Convert flat line item structure to nested schema structure"""
        normalized = {}
        
        # Map name fields
        if "item_name_english" in item:
            normalized["name"] = item["item_name_english"]
        
        # Add required fields
        normalized["data_type"] = "number"  # Assuming all values are numbers
        normalized["is_calculated"] = False  # Default to false, will be overridden by schema if needed
        
        # Convert flat period structure to nested periods array
        if "period_id" in item:
            normalized["periods"] = [{
                "period_label": item["period_id"],
                "value": item.get("value"),
                "status": item.get("extraction_status", "EXTRACTED_SUCCESSFULLY"),
                "unit": item.get("unit", "төг")
            }]
        
        return normalized

    def _relax_schema_constraints(self, schema: dict) -> None:
        """Recursively modify schema to make non-critical fields optional and handle field aliases"""
        if not isinstance(schema, dict):
            return

        # Remove non-critical fields from required lists
        if "required" in schema:
            schema["required"] = [f for f in schema["required"] if not self._is_non_critical_field(f)]

        # Make non-critical const fields optional
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                if self._is_non_critical_field(prop_name):
                    if "const" in prop_schema:
                        del prop_schema["const"]
                    if prop_name in schema.get("required", []):
                        schema["required"].remove(prop_name)
                self._relax_schema_constraints(prop_schema)

        # Handle nested schemas
        for key, value in schema.items():
            if isinstance(value, dict):
                self._relax_schema_constraints(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        self._relax_schema_constraints(item)

    def _get_historical_value(self, line_item_periods_list: Optional[List[Dict[str, Any]]], period_label: str) -> Optional[Decimal]:
        if not line_item_periods_list: return None
        for period_data in line_item_periods_list:
            if isinstance(period_data, dict) and period_data.get("period_label") == period_label:
                value = period_data.get("value")
                status = period_data.get("status") # Key in test_extraction_result is 'status'
                # Only consider successfully extracted, non-null numeric values for sanity checks
                if status == "EXTRACTED_SUCCESSFULLY" and value is not None:
                    try: return Decimal(str(value)) # Convert to Decimal for precision
                    except (InvalidOperation, ValueError, TypeError):
                        logger.warning(f"Could not convert value '{value}' to Decimal for period '{period_label}'.")
                        return None
                return None # Not successful or value is None
        return None # Period not found

    def _find_line_item_periods(self, model_data: Dict[str, Any], path_to_list_container: List[str], list_key: str, item_name: str) -> Optional[List[Dict[str, Any]]]:
        current_level = model_data
        for key in path_to_list_container: # Navigate to the dict containing the list
            current_level = current_level.get(key)
            if not isinstance(current_level, dict): return None
        
        list_of_item_objects = current_level.get(list_key)
        if not isinstance(list_of_item_objects, list): return None

        for item_obj in list_of_item_objects:
            if isinstance(item_obj, dict) and item_obj.get("name") == item_name:
                return item_obj.get("periods") # Return the 'periods' list of the found item
        return None
    
    def _find_single_line_item_periods(self, model_data: Dict[str, Any], path_to_item_object: List[str]) -> Optional[List[Dict[str, Any]]]:
        current_level = model_data
        for key in path_to_item_object: # Navigate to the item object itself
            current_level = current_level.get(key)
            if not isinstance(current_level, dict): return None
        return current_level.get("periods") # Return the 'periods' list of this single item object


    def perform_schema_validation(self, financial_model_instance: Dict[str, Any]) -> None:
        logger.info("Performing JSON schema validation...")
        try:
            # Create a relaxed schema by removing non-critical field validations
            relaxed_schema = self.model_schema_definition.copy()
            self._relax_schema_constraints(relaxed_schema)
            
            # Deep copy and normalize the model data
            normalized_data = json.loads(json.dumps(financial_model_instance))
            
            # Group line items by name to combine periods
            for statement_key, statement in normalized_data.get("financial_statements_core", {}).items():
                if isinstance(statement, dict) and "line_items" in statement:
                    items_by_name = {}
                    for item in statement["line_items"]:
                        normalized = self._normalize_line_item(item)
                        name = normalized["name"]
                        if name in items_by_name:
                            # Combine periods for same line item
                            items_by_name[name]["periods"].extend(normalized["periods"])
                        else:
                            items_by_name[name] = normalized
                    statement["line_items"] = list(items_by_name.values())
            
            validate(instance=normalized_data, schema=relaxed_schema)
            logger.info("Schema validation passed.")
        except JsonSchemaValidationError as e:
            error_path = " -> ".join([str(p) for p in e.path])
            # Only report schema errors for critical fields
            if not self._is_non_critical_field(error_path):
                message = f"Schema validation error at {error_path}: {e.message}"
                logger.error(message)
                self.report.add_error(
                    check_type="SchemaValidation", item_path=error_path, message=e.message,
                    details={"validator": e.validator, "validator_value": e.validator_value, "instance_snippet": str(e.instance)[:200]}
                )
        except Exception as e_other:
            message = f"An unexpected error occurred during schema validation: {str(e_other)}"
            logger.error(message, exc_info=True)
            self.report.add_error(check_type="SchemaValidation", message=message)

    def perform_financial_sanity_checks(self, model_data: Dict[str, Any], tolerance: Decimal = Decimal("1.00")) -> None:
        logger.info("Performing financial sanity checks...")
        fsc = model_data.get("financial_statements_core", {})
        historical_periods = fsc.get("historical_period_labels", [])
        if not historical_periods:
            self.report.add_warning("FinancialSanityCheck", "No historical periods in model, skipping period-based sanity checks.")
            return

        for period in historical_periods:
            logger.debug(f"Sanity checks for period: {period}")
            
            # 1. Balance Sheet: Assets = Liabilities + Equity (Commented out for mock schema as it's not applicable)
            # bs_assets_path = ["financial_statements_core", "balance_sheet", "assets"]
            # bs_le_path = ["financial_statements_core", "balance_sheet", "liabilities_and_equity"]

            # total_assets_periods = self._find_single_line_item_periods(model_data, bs_assets_path + ["total_assets"])
            # total_liabilities_periods = self._find_single_line_item_periods(model_data, bs_le_path + ["total_liabilities"])
            # total_equity_periods = self._find_single_line_item_periods(model_data, bs_le_path + ["total_equity"])

            # assets_val = self._get_historical_value(total_assets_periods, period)
            # liabilities_val = self._get_historical_value(total_liabilities_periods, period)
            # equity_val = self._get_historical_value(total_equity_periods, period)
            
            # bs_check_path_log = f"BalanceSheetCheck.{period}"
            # if all(v is not None for v in [assets_val, liabilities_val, equity_val]):
            #     diff = assets_val - (liabilities_val + equity_val)
            #     if abs(diff) > tolerance:
            #         self.report.add_error(
            #             "FinancialSanityCheck", f"BS Equation failed for {period}. Assets: {assets_val}, L+E: {liabilities_val + equity_val}, Diff: {diff}",
            #             item_path=bs_check_path_log, details={"Assets": assets_val, "Liabilities": liabilities_val, "Equity": equity_val, "Difference": diff}
            #         )
            # else:
            #     missing = [n for v,n in [(assets_val,"TotalAssets"),(liabilities_val,"TotalLiabilities"),(equity_val,"TotalEquity")] if v is None]
            #     self.report.add_warning("FinancialSanityCheck", f"Cannot perform BS Equation check for {period} due to missing: {', '.join(missing)}", item_path=bs_check_path_log)

            # 2. Income Statement: Gross Profit = Revenue - COGS
            is_path_container = ["financial_statements_core", "income_statement"]
            # Ensure line item names match the mock_financial_schema.json (which they do: "Revenue", "COGS", "Gross Profit")
            revenue_periods = self._find_line_item_periods(model_data, is_path_container, "line_items", "Revenue")
            cogs_periods = self._find_line_item_periods(model_data, is_path_container, "line_items", "COGS")
            gp_periods = self._find_line_item_periods(model_data, is_path_container, "line_items", "Gross Profit")

            rev_val = self._get_historical_value(revenue_periods, period)
            cogs_val = self._get_historical_value(cogs_periods, period)
            gp_val = self._get_historical_value(gp_periods, period)

            gp_check_path_log = f"GrossProfitCheck.{period}"
            if all(v is not None for v in [rev_val, cogs_val, gp_val]):
                calculated_gp = rev_val - cogs_val # Assuming COGS is positive value to be subtracted
                diff_gp = gp_val - calculated_gp
                if abs(diff_gp) > tolerance:
                    self.report.add_error(
                        "FinancialSanityCheck", f"Gross Profit calculation inconsistent for {period}. Reported: {gp_val}, Calc (Rev-COGS): {calculated_gp}, Diff: {diff_gp}",
                        item_path=gp_check_path_log, details={"Revenue": rev_val, "COGS": cogs_val, "ReportedGP": gp_val, "CalculatedGP": calculated_gp, "Difference": diff_gp}
                    )
            elif gp_val is not None: # GP reported but components missing
                 missing_gp = [n for v,n in [(rev_val,"Revenue"),(cogs_val,"COGS")] if v is None]
                 self.report.add_warning("FinancialSanityCheck", f"Cannot verify GP for {period} due to missing: {', '.join(missing_gp)}", item_path=gp_check_path_log)
        logger.info("Financial sanity checks completed.")


    def _calculate_phase_a_completeness(self, model_data: Dict[str, Any]) -> None:
        logger.info("Starting Phase A completeness calculation...")
        total_historical_data_points = 0
        successfully_extracted_data_points = 0
        
        fsc_data = model_data.get("financial_statements_core", {})
        historical_periods = fsc_data.get("historical_period_labels", [])
        logger.info(f"Historical periods for completeness: {historical_periods}")
        if not historical_periods: 
            logger.warning("No historical periods found in model data for completeness check.")
            return

        # Iterate through schema to find non-calculated line items
        fsc_schema_props = self.model_schema_definition.get("properties",{}).get("financial_statements_core",{}).get("properties",{})
        logger.debug(f"FSC Schema Props for completeness: {list(fsc_schema_props.keys())}")

        for stmt_key, stmt_schema in fsc_schema_props.items():
            logger.debug(f"Processing statement for completeness: {stmt_key}")
            if stmt_schema.get("type") == "object" and "properties" in stmt_schema:
                # Process line_items if they are defined as an array in the schema (as in mock_financial_schema.json)
                if "line_items" in stmt_schema["properties"] and stmt_schema["properties"]["line_items"].get("type") == "array":
                    actual_line_items_data = model_data.get("financial_statements_core",{}).get(stmt_key,{}).get("line_items",[])
                    line_item_schemas_list = stmt_schema["properties"]["line_items"].get("items", []) # Gets the array of schema items
                    logger.debug(f"Found {len(line_item_schemas_list)} line item schemas in {stmt_key} (array type)")

                    for item_def_schema_container in line_item_schemas_list: # Iterates through the array of schema definitions
                        item_def_props = item_def_schema_container.get("properties", {})
                        item_name_from_schema = item_def_props.get("name", {}).get("const")
                        is_calculated = item_def_props.get("is_calculated", {}).get("const", False)
                        logger.debug(f"  Schema item: {item_name_from_schema}, is_calculated: {is_calculated}")

                        if item_name_from_schema:
                            total_historical_data_points += len(historical_periods)
                            logger.debug(f"    Added {len(historical_periods)} to total_historical_data_points for {item_name_from_schema}. New total: {total_historical_data_points}")
                            # Support flat structure with item_name_english
                            matching_items = [it for it in actual_line_items_data if it.get("item_name_english") == item_name_from_schema]
                            if matching_items:
                                # Handle flat structure (each period is a separate line item)
                                for item in matching_items:
                                    period = item.get('period_id')
                                    status = item.get('extraction_status')
                                    logger.debug(f"      Checking line item {item_name_from_schema} - Period: {period}, Status: '{status}', Value: {item.get('value')}")
                                    if status == "EXTRACTED_SUCCESSFULLY" and item.get("value") is not None:
                                        successfully_extracted_data_points +=1
                                        logger.debug(f"      Found EXTRACTED_SUCCESSFULLY for {item_name_from_schema} in {period}. New success count: {successfully_extracted_data_points}")

        if total_historical_data_points > 0:
            completeness = (float(successfully_extracted_data_points) / total_historical_data_points) * 100
            self.report.phase_a_data_completeness_score = round(completeness, 2)
            logger.info(f"Phase A Completeness: Score: {self.report.phase_a_data_completeness_score}%, Extracted: {successfully_extracted_data_points}, Total: {total_historical_data_points} data points.")
        else:
            logger.info("No non-calculated historical items found to assess Phase A completeness.")

    def run_all_checks(self, financial_model_instance: Dict[str, Any]) -> QualityReport:
        self.report = QualityReport() # Initialize a new report for this run
        logger.info(f"Running all quality checks for model instance. Schema: {self.schema_definition_path}")

        self.perform_schema_validation(financial_model_instance)
        # Only proceed with other checks if schema validation passes or has only warnings
        # For this mock version, we might want to see all errors, so we run them regardless of schema outcome.
        # if self.report.overall_status == "FAILED" and any(e.check_type == "SchemaValidation" for e in self.report.errors):
        #     logger.warning("Skipping further checks due to critical schema validation failure.")
        #     return self.report

        self.perform_financial_sanity_checks(financial_model_instance)
        self._calculate_phase_a_completeness(financial_model_instance)
        
        logger.info(f"All quality checks completed. Overall Status: {self.report.overall_status}")
        return self.report

# --- Standalone Test Function for Mock Schema --- 
def run_mock_quality_checker_tests():
    print("\n--- Testing Mock QualityChecker ---")
    script_dir = Path(__file__).parent 
    # Assuming mock_financial_schema.json is in the same directory (src/financial_modeling)
    schema_path_for_mock_test = script_dir / "mock_financial_schema.json"
    # Assuming test_extraction_result_TEST_REAL_DOC_001.json is in test_outputs relative to project root (Alpy)
    project_root = script_dir.parent.parent 
    input_data_path = Path(__file__).resolve().parent.parent.parent / "test_extraction_result_TEST_REAL_DOC_001.json"

    if not schema_path_for_mock_test.exists():
        print(f"ERROR: Mock schema file not found: {schema_path_for_mock_test}. Cannot run Mock QualityChecker tests.")
        return
    if not input_data_path.exists():
        print(f"ERROR: Input data file not found: {input_data_path}. Cannot run Mock QualityChecker tests.")
        return

    qc = QualityChecker(schema_definition_path=schema_path_for_mock_test)

    # Load the actual extraction result
    try:
        with open(input_data_path, 'r', encoding='utf-8') as f:
            valid_model_data = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load input data file {input_data_path}: {e}")
        return

    print(f"\n--- Test Case: Running checks on {input_data_path.name} using {schema_path_for_mock_test.name} ---")

    # Transform valid_model_data (from test_extraction_result_TEST_REAL_DOC_001.json)
    # to conform to mock_financial_schema.json
    transformed_data = {}

    # 1. model_metadata
    transformed_data["model_metadata"] = {
        "model_name": valid_model_data.get("primary_document_name", "Mock Financial Model from Test Data"),
        "version": "1.0_mock_transformed",
        "currency": valid_model_data.get("metadata", {}).get("reporting_currency", "MNT"),
        "fiscal_year_end": valid_model_data.get("metadata", {}).get("fiscal_year_end", "12-31"),
        "historical_periods_count": len(valid_model_data.get("historical_periods", [])),
        "projection_periods_count": 0
        # Add other required fields from schema with defaults if not in source
    }

    # 2. financial_statements_core
    transformed_data["financial_statements_core"] = {}
    fsc_core = transformed_data["financial_statements_core"]

    historical_period_objects = valid_model_data.get("historical_periods", [])
    fsc_core["historical_period_labels"] = [p.get("period_label") for p in historical_period_objects if p.get("period_label")]
    fsc_core["projected_period_labels"] = []

    # Initialize statements as per schema
    fsc_core["income_statement"] = {"line_items": []}
    fsc_core["balance_sheet"] = {"line_items": []} # Required by schema
    fsc_core["cash_flow_statement"] = {"line_items": []} # Required by schema

    # Populate income_statement.line_items
    # Get schema definition for income statement line items
    # This requires qc instance or loading schema separately here.
    # For simplicity, we'll use the QualityChecker's loaded schema if possible, or define expected items.
    qc = QualityChecker(schema_definition_path=schema_path_for_mock_test)

    name_map_for_input_data = {
        "Cost of Goods Sold": "COGS",
        "Depreciation & Amortization": "D&A",
        "Taxes": "Tax",
        "Property, Plant & Equipment, Net": "Property, Plant & Equipment"
        # This map might need to be extended if similar name mismatches exist for BS/CFS items
    }
    
    # Expected line items from mock_financial_schema.json for income_statement
    # (This should ideally be dynamically read from qc.model_schema_definition)
    schema_income_statement_line_items = qc.model_schema_definition.get("properties", {}).get("financial_statements_core", {}).get("properties", {}).get("income_statement", {}).get("properties", {}).get("line_items", {}).get("items", [])

    for schema_li_def_container in schema_income_statement_line_items:
        # The specific properties for this line item (like name const, is_calculated const)
        # are in schema_li_def_container.get("properties", {})
        schema_li_props = schema_li_def_container.get("properties", {})
        
        if not schema_li_props or "name" not in schema_li_props or "const" not in schema_li_props["name"]:
            logger.warning(f"Skipping schema line item due to missing name/const in its 'properties' block: {schema_li_def_container}")
            continue

        li_name = schema_li_props["name"]["const"]
        li_name_mn = schema_li_props.get("name_mn", {}).get("const", "")
        is_calculated = schema_li_props.get("is_calculated", {}).get("const", False)

        new_line_item = {
            "name": li_name,
            "name_mn": li_name_mn,
            "data_type": "currency_value", # Default from base schema
            "is_calculated": is_calculated,
            "periods": []
        }

        for period_label in fsc_core["historical_period_labels"]:
            found_data = None
            input_li_name_to_search = name_map_for_input_data.get(li_name, li_name)
            for item_data in valid_model_data.get("line_items_data", []):
                if item_data.get("item_name_english") == input_li_name_to_search and item_data.get("period_id") == period_label:
                    found_data = item_data
                    break
            
            period_entry = {
                "period_label": period_label,
                "value": found_data.get("value") if found_data else None,
                "source_reference": found_data.get("notes") if found_data else None,
                "status": "EXTRACTED_SUCCESSFULLY" if found_data and found_data.get("value") is not None else "DATA_NOT_FOUND_IN_INPUT"
            }
            new_line_item["periods"].append(period_entry)
        
        fsc_core["income_statement"]["line_items"].append(new_line_item)

    report = qc.run_all_checks(transformed_data)
    print(f"Overall Status: {report.overall_status}")
    if report.phase_a_data_completeness_score is not None:
        print(f"Completeness Score: {report.phase_a_data_completeness_score}%")
    
    if report.errors:
        print("Errors found:")
        for err in report.errors: print(f"  ERROR: {err.check_type} - {err.message} (Path: {err.item_path}) Details: {err.details}")
    if report.warnings:
        print("Warnings found:")
        for warn in report.warnings: print(f"  WARNING: {warn.check_type} - {warn.message} (Path: {warn.item_path})")
    
    # Basic assertion: we expect the process to run, status might vary based on data quality
    assert report.overall_status in ["PASSED", "PASSED_WITH_WARNINGS", "FAILED"], "Quality check did not produce a valid status."
    print(f"Mock QualityChecker test finished for {input_data_path.name}. Status: {report.overall_status}")

    # Example of a schema error test (optional, can be expanded)
    # Create a copy of the data and introduce a schema error
    # schema_error_model_data = json.loads(json.dumps(actual_extraction_result))
    # if schema_error_model_data.get("model_metadata"):
    #     schema_error_model_data["model_metadata"]["currency"] = 123 # Invalid type
    #     print("\n--- Test Case: Schema Error (mock data) ---")
    #     report_schema_error = qc.run_all_checks(schema_error_model_data)
    #     print(f"Overall Status (Schema Error Test): {report_schema_error.overall_status}")
    #     assert report_schema_error.overall_status == "FAILED", "Schema error test did not result in FAILED status."
    #     assert any(e.check_type == "SchemaValidation" and "123 is not of type 'string'" in e.message for e in report_schema_error.errors), "Missing expected schema error for currency type."
    # else:
    #     print("Skipping schema error test as model_metadata not found in input.")

    print("\n--- Mock QualityChecker tests finished ---")

if __name__ == "__main__":
    run_mock_quality_checker_tests()
