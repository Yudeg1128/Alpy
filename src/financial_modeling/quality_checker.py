import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union, Literal
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, Field
from jsonschema import validate, ValidationError as JsonSchemaValidationError

# Setup a logger for this module
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
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
            logger.error(f"QualityChecker: Failed to load schema definition: {e}", exc_info=True)
            raise
        self.report: QualityReport # Will be initialized in run_all_checks

    def _get_historical_value(self, line_item_periods_list: Optional[List[Dict[str, Any]]], period_label: str) -> Optional[Decimal]:
        if not line_item_periods_list: return None
        for period_data in line_item_periods_list:
            if isinstance(period_data, dict) and period_data.get("period_label") == period_label:
                value = period_data.get("value")
                status = period_data.get("extraction_status")
                # Only consider successfully extracted, non-null numeric values for sanity checks
                if status == "SUCCESS" and value is not None:
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
            validate(instance=financial_model_instance, schema=self.model_schema_definition)
            logger.info("JSON schema validation PASSED.")
        except JsonSchemaValidationError as e:
            error_path = " -> ".join(map(str, e.path)) if e.path else "Root"
            message = f"Schema validation failed at '{error_path}': {e.message}"
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
            
            # 1. Balance Sheet: Assets = Liabilities + Equity
            bs_assets_path = ["financial_statements_core", "balance_sheet", "assets"]
            bs_le_path = ["financial_statements_core", "balance_sheet", "liabilities_and_equity"]

            total_assets_periods = self._find_single_line_item_periods(model_data, bs_assets_path + ["total_assets"])
            total_liabilities_periods = self._find_single_line_item_periods(model_data, bs_le_path + ["total_liabilities"])
            total_equity_periods = self._find_single_line_item_periods(model_data, bs_le_path + ["total_equity"])

            assets_val = self._get_historical_value(total_assets_periods, period)
            liabilities_val = self._get_historical_value(total_liabilities_periods, period)
            equity_val = self._get_historical_value(total_equity_periods, period)
            
            bs_check_path_log = f"BalanceSheetCheck.{period}"
            if all(v is not None for v in [assets_val, liabilities_val, equity_val]):
                diff = assets_val - (liabilities_val + equity_val)
                if abs(diff) > tolerance:
                    self.report.add_error(
                        "FinancialSanityCheck", f"BS Equation failed for {period}. Assets: {assets_val}, L+E: {liabilities_val + equity_val}, Diff: {diff}",
                        item_path=bs_check_path_log, details={"Assets": assets_val, "Liabilities": liabilities_val, "Equity": equity_val, "Difference": diff}
                    )
            else:
                missing = [n for v,n in [(assets_val,"TotalAssets"),(liabilities_val,"TotalLiabilities"),(equity_val,"TotalEquity")] if v is None]
                self.report.add_warning("FinancialSanityCheck", f"Cannot perform BS Equation check for {period} due to missing: {', '.join(missing)}", item_path=bs_check_path_log)

            # 2. Income Statement: Gross Profit = Revenue - COGS
            is_path_container = ["financial_statements_core", "income_statement"]
            revenue_periods = self._find_line_item_periods(model_data, is_path_container, "line_items", "Revenue")
            cogs_periods = self._find_line_item_periods(model_data, is_path_container, "line_items", "Cost of Goods Sold (COGS)")
            gp_periods = self._find_line_item_periods(model_data, is_path_container, "line_items", "Gross Profit")

            rev_val = self._get_historical_value(revenue_periods, period)
            cogs_val = self._get_historical_value(cogs_periods, period)
            gp_val = self._get_historical_value(gp_periods, period)

            gp_check_path_log = f"GrossProfitCheck.{period}"
            if all(v is not None for v in [rev_val, cogs_val, gp_val]):
                calculated_gp = rev_val - cogs_val
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
        total_historical_data_points = 0
        successfully_extracted_data_points = 0
        
        fsc_data = model_data.get("financial_statements_core", {})
        historical_periods = fsc_data.get("historical_period_labels", [])
        if not historical_periods: return

        # Iterate through schema to find non-calculated line items
        fsc_schema_props = self.model_schema_definition.get("properties",{}).get("financial_statements_core",{}).get("properties",{})

        for stmt_key, stmt_schema in fsc_schema_props.items():
            if stmt_schema.get("type") == "object" and "properties" in stmt_schema:
                # Direct line_items (e.g., Income Statement)
                if "line_items" in stmt_schema["properties"] and stmt_schema["properties"]["line_items"].get("type") == "array":
                    actual_line_items_data = model_data.get("financial_statements_core",{}).get(stmt_key,{}).get("line_items",[])
                    for item_idx, item_def_schema in enumerate(stmt_schema["properties"]["line_items"].get("items",[])):
                        is_calculated = item_def_schema.get("properties",{}).get("is_calculated",{}).get("const", False)
                        if not is_calculated:
                            total_historical_data_points += len(historical_periods)
                            item_name = item_def_schema.get("properties",{}).get("name",{}).get("const")
                            item_data_in_model = next((it for it in actual_line_items_data if it.get("name") == item_name), None)
                            if item_data_in_model:
                                for p_data in item_data_in_model.get("periods",[]):
                                    if p_data.get("extraction_status") == "SUCCESS" and p_data.get("value") is not None:
                                        successfully_extracted_data_points +=1
                # Nested structures (e.g., Balance Sheet)
                elif stmt_key == "balance_sheet":
                    bs_props = stmt_schema.get("properties",{})
                    for bs_main_cat_key, bs_main_cat_schema in bs_props.items(): # assets, liabilities_and_equity
                        if bs_main_cat_schema.get("type") == "object" and "properties" in bs_main_cat_schema:
                            for bs_sub_cat_key, bs_sub_cat_array_schema in bs_main_cat_schema.get("properties",{}).items(): # current_assets, etc.
                                if bs_sub_cat_array_schema.get("type") == "array" and "items" in bs_sub_cat_array_schema:
                                    actual_sub_cat_data = model_data.get("financial_statements_core",{}).get(stmt_key,{}).get(bs_main_cat_key,{}).get(bs_sub_cat_key,[])
                                    for item_idx, item_def_schema in enumerate(bs_sub_cat_array_schema.get("items",[])):
                                        is_calculated = item_def_schema.get("properties",{}).get("is_calculated",{}).get("const", False)
                                        if not is_calculated:
                                            total_historical_data_points += len(historical_periods)
                                            item_name = item_def_schema.get("properties",{}).get("name",{}).get("const")
                                            item_data_in_model = next((it for it in actual_sub_cat_data if it.get("name") == item_name), None)
                                            if item_data_in_model:
                                                for p_data in item_data_in_model.get("periods",[]):
                                                    if p_data.get("extraction_status") == "SUCCESS" and p_data.get("value") is not None:
                                                        successfully_extracted_data_points +=1
        if total_historical_data_points > 0:
            completeness = (float(successfully_extracted_data_points) / total_historical_data_points) * 100
            self.report.phase_a_data_completeness_score = round(completeness, 2)
            logger.info(f"Phase A Completeness: {self.report.phase_a_data_completeness_score}% ({successfully_extracted_data_points}/{total_historical_data_points} data points).")
        else:
            logger.info("No non-calculated historical items found to assess Phase A completeness.")


    def run_all_checks(self, financial_model_instance: Dict[str, Any]) -> QualityReport:
        self.report = QualityReport() # Initialize a fresh report for this run
        self.perform_schema_validation(financial_model_instance)
        
        # Only run detailed financial sanity checks if basic schema is OK enough
        # A major schema failure might make sanity checks impossible or misleading
        if not any(e.severity == "ERROR" and e.check_type == "SchemaValidation" for e in self.report.errors):
            self.perform_financial_sanity_checks(financial_model_instance)
            self._calculate_phase_a_completeness(financial_model_instance)
        else:
            logger.warning("Skipping financial sanity checks and completeness due to critical schema validation errors.")
            self.report.add_warning("QualityLogic", "Skipped financial sanity checks and completeness due to schema errors.")
            if self.report.overall_status != "FAILED": self.report.overall_status = "FAILED" # Ensure overall status reflects schema failure

        logger.info(f"Quality Check Finished. Overall Status: {self.report.overall_status}")
        return self.report

# --- Standalone Test Function ---
def run_quality_checker_tests():
    print("\n--- Testing QualityChecker ---")
    script_dir = Path(__file__).parent 
    project_root = script_dir.parent.parent 
    schema_path_for_test = project_root / "fund" / "financial_model_schema.json"
    
    if not schema_path_for_test.exists():
        print(f"ERROR: Test schema file not found: {schema_path_for_test}. Cannot run QualityChecker tests.")
        return

    qc = QualityChecker(schema_definition_path=schema_path_for_test)

    # Test Case 1: Valid model data, meticulously built to match schema v2.2
    valid_model_data = {
        "model_metadata": {
            "model_name": "Debt Instrument Issuer Financial Model",
            "version": "2.2_fully_expanded", # Match schema default or a valid string
            "target_company_name": "Test Co Valid",
            "ticker_symbol": "VALID",
            "currency": "USD",
            "fiscal_year_end": "12-31",
            "historical_periods_count": 1,
            "projection_periods_count": 0,
            "analyst_name": "QC Test Agent",
            "date_created": "2024-01-01T10:00:00Z",
            "ai_instructions_general": "Process this valid model.",
            "model_focus": "Fixed Income Securities Analysis"
        },
        "global_assumptions": [], # Schema default
        "company_specific_assumptions_debt_focus": { # Schema default is {}, ensure required sub-objects if any
            "section_description": "Test assumptions",
            "ai_instructions_section": "Follow instructions",
            "revenue_drivers": [], "cost_and_margin_drivers": [], "working_capital_drivers": [],
            "capital_expenditure_and_asset_drivers": [],
            "financing_and_debt_assumptions": {
                "existing_debt_refinancing_strategy": None, "new_debt_issuance_plans": [],
                "assumed_interest_rate_on_new_debt": None, "dividend_policy_and_payout_ratio": [],
                "share_repurchases_policy": [], "debt_paydown_priority": None,
                "cash_sweep_mechanism_for_excess_cash_flow": None, "minimum_cash_balance_target": None
            },
            "tax_assumptions": {"effective_tax_rate": [], "cash_tax_rate_if_different": []}
        },
        "financial_statements_core": {
            "section_description": "Core financial statements for test.",
            "historical_period_labels": ["FY2023"],
            "projected_period_labels": [],
            "income_statement": {
                "ai_instructions_statement": "IS for QC test.",
                "line_items": [
                    {"name": "Revenue", "data_type": "currency_value", "is_calculated": False, "periods": [{"period_label": "FY2023", "value": 1000.0, "extraction_status": "SUCCESS", "source_reference": "TestSrc"}]},
                    {"name": "Cost of Goods Sold (COGS)", "data_type": "currency_value", "is_calculated": False, "periods": [{"period_label": "FY2023", "value": 600.0, "extraction_status": "SUCCESS"}]},
                    {"name": "Gross Profit", "data_type": "currency_value", "is_calculated": True, "calculation_logic_description": "Revenue - COGS", "periods": [{"period_label": "FY2023", "value": 400.0, "extraction_status": "SUCCESS"}]},
                    {"name": "SG&A Expenses", "data_type": "currency_value", "is_calculated": False, "periods": [{"period_label": "FY2023", "value": 100.0, "extraction_status": "SUCCESS"}]},
                    {"name": "Other Operating Expenses/Income", "data_type": "currency_value", "is_calculated": False, "periods": [{"period_label": "FY2023", "value": 0.0, "extraction_status": "SUCCESS"}]},
                    {"name": "EBITDA (Earnings Before Interest, Taxes, Depreciation & Amortization)", "data_type": "currency_value", "is_calculated": True, "calculation_logic_description": "Calc", "periods": [{"period_label": "FY2023", "value": 300.0, "extraction_status": "SUCCESS"}]}, # GP(400) - SG&A(100) - OtherOpEx(0)
                    {"name": "Depreciation & Amortization (D&A)", "data_type": "currency_value", "is_calculated": False, "periods": [{"period_label": "FY2023", "value": 50.0, "extraction_status": "SUCCESS"}]},
                    {"name": "EBIT (Earnings Before Interest & Taxes)", "data_type": "currency_value", "is_calculated": True, "calculation_logic_description": "EBITDA - D&A", "periods": [{"period_label": "FY2023", "value": 250.0, "extraction_status": "SUCCESS"}]},
                    {"name": "Interest Expense", "data_type": "currency_value", "is_calculated": False, "periods": [{"period_label": "FY2023", "value": 20.0, "extraction_status": "SUCCESS"}]},
                    {"name": "Interest Income", "data_type": "currency_value", "is_calculated": False, "periods": [{"period_label": "FY2023", "value": 5.0, "extraction_status": "SUCCESS"}]},
                    {"name": "EBT (Earnings Before Tax)", "data_type": "currency_value", "is_calculated": True, "calculation_logic_description": "EBIT - IntExp + IntInc", "periods": [{"period_label": "FY2023", "value": 235.0, "extraction_status": "SUCCESS"}]},
                    {"name": "Income Tax Expense", "data_type": "currency_value", "is_calculated": False, "periods": [{"period_label": "FY2023", "value": 47.0, "extraction_status": "SUCCESS"}]}, # Assuming 20% tax
                    {"name": "Net Income", "data_type": "currency_value", "is_calculated": True, "calculation_logic_description": "EBT - Tax", "periods": [{"period_label": "FY2023", "value": 188.0, "extraction_status": "SUCCESS"}]}
                ]
            },
            "balance_sheet": { 
                "ai_instructions_statement": "BS for QC test.",
                "assets": { 
                    "current_assets": [
                        {"name": "Cash & Cash Equivalents", "data_type": "currency_value", "is_calculated": False, "periods": [{"period_label": "FY2023", "value": 100.0, "extraction_status": "SUCCESS"}]},
                        {"name": "Accounts Receivable", "data_type": "currency_value", "is_calculated": False, "periods": [{"period_label": "FY2023", "value": 200.0, "extraction_status": "SUCCESS"}]},
                        {"name": "Inventory", "data_type": "currency_value", "is_calculated": False, "periods": [{"period_label": "FY2023", "value": 150.0, "extraction_status": "SUCCESS"}]}
                    ],
                    "non_current_assets": [
                        {"name": "Net PP&E", "data_type": "currency_value", "is_calculated": False, "periods": [{"period_label": "FY2023", "value": 1000.0, "extraction_status": "SUCCESS"}]},
                        {"name": "Goodwill", "data_type": "currency_value", "is_calculated": False, "periods": [{"period_label": "FY2023", "value": 50.0, "extraction_status": "SUCCESS"}]},
                        {"name": "Intangible Assets (Net)", "data_type": "currency_value", "is_calculated": False, "periods": [{"period_label": "FY2023", "value": 0.0, "extraction_status": "SUCCESS"}]}
                    ],
                    "total_assets": {"name": "Total Assets", "data_type": "currency_value", "is_calculated": True, "calculation_logic_description": "Sum", "periods": [{"period_label": "FY2023", "value": 1500.0, "extraction_status": "SUCCESS"}]}
                }, 
                "liabilities_and_equity": { 
                    "current_liabilities": [
                        {"name": "Accounts Payable", "data_type": "currency_value", "is_calculated": False, "periods": [{"period_label": "FY2023", "value": 150.0, "extraction_status": "SUCCESS"}]},
                        {"name": "Short-Term Debt & Current Portion of LTD", "data_type": "currency_value", "is_calculated": False, "periods": [{"period_label": "FY2023", "value": 50.0, "extraction_status": "SUCCESS"}]}
                    ],
                    "non_current_liabilities": [
                        {"name": "Long-Term Debt", "data_type": "currency_value", "is_calculated": False, "periods": [{"period_label": "FY2023", "value": 400.0, "extraction_status": "SUCCESS"}]}
                    ],
                    "total_liabilities": {"name": "Total Liabilities", "data_type": "currency_value", "is_calculated": True, "calculation_logic_description": "Sum", "periods": [{"period_label": "FY2023", "value": 600.0, "extraction_status": "SUCCESS"}]},
                    "equity": [ 
                        {"name": "Common Stock & APIC", "data_type": "currency_value", "is_calculated": False, "periods": [{"period_label": "FY2023", "value": 500.0, "extraction_status": "SUCCESS"}]},
                        {"name": "Retained Earnings", "data_type": "currency_value", "is_calculated": False, "periods": [{"period_label": "FY2023", "value": 400.0, "extraction_status": "SUCCESS"}]}
                    ],
                    "total_equity": {"name": "Total Equity", "data_type": "currency_value", "is_calculated": True, "calculation_logic_description": "Sum", "periods": [{"period_label": "FY2023", "value": 900.0, "extraction_status": "SUCCESS"}]},
                    "total_liabilities_and_equity": {"name": "Total Liabilities & Equity", "data_type": "currency_value", "is_calculated": True, "calculation_logic_description": "Sum", "periods": [{"period_label": "FY2023", "value": 1500.0, "extraction_status": "SUCCESS"}]}
                },
                "balance_sheet_check": {"name": "Balance Sheet Check (Assets - L&E)", "data_type":"currency_value", "is_calculated": True, "periods": [{"period_label":"FY2023", "value":0.0, "extraction_status": "SUCCESS"}]}
            },
            "cash_flow_statement": {
                 "ai_instructions_statement": "CFS for QC test.",
                 "cash_flow_from_operations_cfo": [
                    {"name": "Net Income", "data_type": "currency_value", "is_calculated": False, "periods": [{"period_label": "FY2023", "value": 188.0, "extraction_status": "SUCCESS"}]},
                    {"name": "Depreciation & Amortization", "data_type": "currency_value", "is_calculated": False, "periods": [{"period_label": "FY2023", "value": 50.0, "extraction_status": "SUCCESS"}]},
                    {"name": "Changes in Accounts Receivable", "data_type": "currency_value", "is_calculated": True, "periods": []}, # Placeholder for calculated
                    {"name": "Changes in Inventory", "data_type": "currency_value", "is_calculated": True, "periods": []},
                    {"name": "Changes in Accounts Payable", "data_type": "currency_value", "is_calculated": True, "periods": []},
                    {"name": "Net Cash Flow from Operations", "data_type": "currency_value", "is_calculated": True, "periods": []}
                 ],
                 "cash_flow_from_investing_cfi": [
                    {"name": "Capital Expenditures (Capex)", "data_type": "currency_value", "is_calculated": False, "periods": [{"period_label": "FY2023", "value": -70.0, "extraction_status": "SUCCESS"}]}, # Capex is outflow
                    {"name": "Net Cash Flow from Investing", "data_type": "currency_value", "is_calculated": True, "periods": []}
                 ],
                 "cash_flow_from_financing_cff": [
                    {"name": "Net Debt Issued/(Repaid)", "data_type": "currency_value", "is_calculated": False, "periods": [{"period_label": "FY2023", "value": 10.0, "extraction_status": "SUCCESS"}]}, # Example: Net borrowing
                    {"name": "Net Equity Issued/(Repurchased)", "data_type": "currency_value", "is_calculated": False, "periods": [{"period_label": "FY2023", "value": 0.0, "extraction_status": "SUCCESS"}]},
                    {"name": "Dividends Paid", "data_type": "currency_value", "is_calculated": False, "periods": [{"period_label": "FY2023", "value": -20.0, "extraction_status": "SUCCESS"}]}, # Dividend is outflow
                    {"name": "Net Cash Flow from Financing", "data_type": "currency_value", "is_calculated": True, "periods": []}
                 ],
                 "net_change_in_cash": {"name": "Net Change in Cash", "is_calculated": True, "data_type": "currency_value", "periods": []},
                 "cash_beginning_of_period": {"name": "Cash at Beginning of Period", "is_calculated": False, "data_type": "currency_value", "periods": [{"period_label": "FY2023", "value": 50.0, "extraction_status": "SUCCESS"}]}, # Example
                 "cash_end_of_period": {"name": "Cash at End of Period", "is_calculated": True, "data_type": "currency_value", "periods": []} # Will link to BS Cash
            }
        },
        # Ensure all other top-level required properties from schema are present
        "supporting_schedules_debt_focus": {}, # Default if required by schema
        "credit_analysis_metrics_and_ratios": {}, # Default if required
        "valuation_context_for_debt": {}, # Default if required
        "sensitivity_and_scenario_analysis_debt_focus": {}, # Default if required
        "ai_agent_summary_and_confidence_debt_focus": {} # Default if required
    }
    
    # Dynamically fill any remaining missing required root properties with schema defaults
    # This ensures the test data is as complete as possible against the schema definition.
    root_schema_props = qc.model_schema_definition.get("properties", {})
    for req_prop in qc.model_schema_definition.get("required", []):
        if req_prop not in valid_model_data:
            prop_schema = root_schema_props.get(req_prop, {})
            # Use schema "default" if available, otherwise use type-based empty default
            if "default" in prop_schema:
                valid_model_data[req_prop] = prop_schema["default"]
            elif prop_schema.get("type") == "object": valid_model_data[req_prop] = {}
            elif prop_schema.get("type") == "array": valid_model_data[req_prop] = []
            else: valid_model_data[req_prop] = None # Or raise error if no suitable default

    print("\n--- Test Case 1: Valid Model ---")
    report1 = qc.run_all_checks(valid_model_data)
    print(f"Overall Status: {report1.overall_status}")
    print(f"Completeness Score: {report1.phase_a_data_completeness_score}%")
    if report1.errors:
        print("Errors found in Test Case 1:")
        for err in report1.errors: print(f"  ERROR Test1: {err.check_type} - {err.message} (Path: {err.item_path}) Details: {err.details}")
    if report1.warnings:
        print("Warnings found in Test Case 1:")
        for warn in report1.warnings: print(f"  WARNING Test1: {warn.check_type} - {warn.message} (Path: {warn.item_path})")
    assert report1.overall_status == "PASSED", f"Test 1 FAILED: Expected PASSED, got {report1.overall_status}."


    # Test Case 2: Model with Schema Error (e.g., metadata.currency is wrong type)
    schema_error_model_data = json.loads(json.dumps(valid_model_data)) # deepcopy
    schema_error_model_data["model_metadata"]["currency"] = 123 # Invalid type
    print("\n--- Test Case 2: Schema Error Model ---")
    report2 = qc.run_all_checks(schema_error_model_data)
    print(f"Overall Status: {report2.overall_status}")
    assert report2.overall_status == "FAILED", "Test Case 2 FAILED: Schema error not detected as FAILED"
    assert any(e.check_type == "SchemaValidation" and "123 is not of type 'string'" in e.message for e in report2.errors), "Test Case 2: Missing expected schema error for currency type."
    if not any(e.check_type == "SchemaValidation" and "123 is not of type 'string'" in e.message for e in report2.errors):
        print("Unexpected errors in Test Case 2:", report2.errors)


    # Test Case 3: Model with Financial Sanity Error (BS doesn't balance)
    financial_error_model_data = json.loads(json.dumps(valid_model_data)) # deepcopy
    # Make BS not balance: Assets = 1505, L+E remains 1500
    financial_error_model_data["financial_statements_core"]["balance_sheet"]["assets"]["total_assets"]["periods"][0]["value"] = 1505.0 
    print("\n--- Test Case 3: Financial Sanity Error Model ---")
    report3 = qc.run_all_checks(financial_error_model_data)
    print(f"Overall Status: {report3.overall_status}")
    assert report3.overall_status == "FAILED", f"Test Case 3 FAILED: Financial error not detected as FAILED. Errors: {report3.errors}"
    assert any(e.check_type == "FinancialSanityCheck" and "BS Equation failed" in e.message for e in report3.errors), "Test Case 3: Missing expected BS equation error."
    if not any(e.check_type == "FinancialSanityCheck" and "BS Equation failed" in e.message for e in report3.errors):
        print("Unexpected errors in Test Case 3:", report3.errors)


    print("\n--- QualityChecker tests finished ---")

if __name__ == "__main__":
    run_quality_checker_tests()