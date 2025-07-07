"""
MCP Server for Credit Intelligence Transformation Phase
"""
import asyncio
import copy
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "financial_analyst"))
from security_folder_utils import require_security_folder, get_subfolder, get_security_file
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP
from mcp.types import Implementation
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

# Import financial statement validator
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import traceback


# Import config with fallback for API keys
try:
    import config
except ImportError:
    # Create a minimal config object with required attributes
    class MinimalConfig:
        GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
        LLM_TOP_P = float(os.environ.get("LLM_TOP_P", 1.0))
    config = MinimalConfig()

# Ensure config.py is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "financial_analyst"))

# Define weights for each plug type and their targets using standardized field names from UnifiedFinancialStatement schema
PLUG_WEIGHTS = {
    # Balance Sheet plugs - Assets
    "reported_vs_calculated_total_assets_plug": {
        "net_property_plant_equipment": 0.5,  # Standardized field name
        "other_non_current_assets": 0.3,      # Standardized field name
        "other_current_assets": 0.2           # Standardized field name
    },
    # Balance Sheet plugs - Liabilities
    "reported_vs_subtotals_total_liabilities_plug": {
        "other_current_liabilities": 0.6,     # Standardized field name
        "other_non_current_liabilities": 0.4   # Standardized field name
    },
    # Balance Sheet plugs - Equity
    "balance_sheet_equation_equity_plug": {
        "retained_earnings": 0.8,  # Standardized field name
        "other_equity": 0.2         # Standardized field name
    },
    # Income Statement plugs - Operating Expenses
    "reported_vs_calculated_operating_expenses_plug": {
        "personnel_expenses": 0.4,          # Standardized field name
        "administrative_expenses": 0.3,        # Standardized field name
        "depreciation_amortization": 0.2,      # Standardized field name
        "other_operating_expenses": 0.1        # Standardized field name
    },
    # Income Statement plugs - Profit Before Tax
    "reported_vs_calculated_profit_before_tax_plug": {
        "operating_profit": 0.7,            # Standardized field name
        "other_income": 0.15,                # Standardized field name
        "other_expenses": 0.15               # Standardized field name
    },
    # Cash Flow plugs - Assets
    "change_in_bs_asset_plug": {
        "change_in_net_loans": 0.6,            # Standardized field name
        "change_in_other_operating_assets": 0.4  # Standardized field name
    },
    # Cash Flow plugs - Liabilities
    "change_in_bs_liability_plug": {
        "change_in_other_operating_liabilities": 0.6,  # Standardized field name
        "repayments_of_long_term_debt": 0.4           # Standardized field name
    },
    # Cash Flow plugs - Equity
    "change_in_bs_equity_plug": {
        "proceeds_from_issuance_of_common_stock": 0.7,  # Standardized field name
        "other_financing_activities": 0.3             # Standardized field name
    }
}

# Set up logging to file
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True, parents=True)
log_file = log_dir / "debt_reconciliation.log"

# Clear any existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create file handler
file_handler = logging.FileHandler(log_file, mode='a')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

# Create console handler - set to WARNING level to reduce console spam
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
# Set both handlers to INFO level
file_handler.setLevel(logging.INFO)
console_handler.setLevel(logging.INFO)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)

logger = logging.getLogger("MCPCreditIntelligenceTransformer")
logger.info("=" * 80)
logger.info("Starting MCP Credit Intelligence Transformer server")
logger.info(f"Logging to {log_file.absolute()}")
logger.info("=" * 80)

# --- Helper: safe get ---
def safe_get(dct, *keys):
    for k in keys:
        if dct is None or k not in dct:
            return None
        dct = dct[k]
    return dct

def get_best_assumption(assumptions_by_cat, metric_name, period, default=None):
    """Get assumption value for a specific metric and period from assumptions_by_category"""
    for category_name, assumptions_list in assumptions_by_cat.items():
        for assumption in assumptions_list:
            if assumption.get("metric_name") == metric_name:
                periods = assumption.get("projection_periods", [])
                values = assumption.get("values", [])
                if period in periods and len(values) > periods.index(period):
                    return values[periods.index(period)]
    return default

mcp_app = FastMCP(
    name="MCPCreditIntelligenceTransformerServer",
    version="0.1.0",
    description="MCP server for transforming extracted bond data into actionable credit intelligence"
)

# --- Input Schemas ---
class AssumptionsGeneratorInput(BaseModel):
    # Use the data extraction schema as input
    bond_metadata: dict
    bond_financials_historical: dict
    bond_financials_projections: Optional[dict] = None
    collateral_and_protective_clauses: Optional[dict] = None
    issuer_business_profile: Optional[dict] = None
    industry_profile: Optional[dict] = None

class CreditSummaryInput(BaseModel):
    financial_model_output: dict = Field(..., description="Output from the financial model phase")
    qualitative_inputs: dict = Field(..., description="Qualitative inputs for credit analysis")
    stress_test_results: Optional[dict] = Field(None, description="Optional stress test results")


@mcp_app.tool()
async def reclassify_financial_plugs(security_id: str) -> dict:
    """
    Deterministically reclassifies financial statement plugs into appropriate line items.
    1. Loads validated historical data with plug values.
    2. Applies predefined mapping rules to reallocate plug amounts.
    3. Preserves all validated data; only redistributes plug amounts.
    4. Logs reclassification actions in 'reclassification_notes'.
    5. Returns enhanced financial data with reclassified plugs.

    Args:
        security_id: The security ID to reclassify financial plugs for.

    Returns:
        Dictionary containing the enhanced financial data with reclassified plugs,
        or an error dictionary if issues occur.
    """
    import logging
    logger = logging.getLogger("MCPCreditIntelligenceTransformer")
    import copy
    import json
    from pathlib import Path

    # Ensure require_security_folder is available in the scope
    # This might require an import if it's defined elsewhere, e.g.:
    # from .utils import require_security_folder
    # For now, assuming it's available globally or in the same file.

    logger.info(f"Starting deterministic plug reclassification for security {security_id}...")
    reclassified_data = None # Initialize to ensure it's defined in finally block if error occurs early

    try:
        security_folder = require_security_folder(security_id)
        credit_analysis_folder = security_folder / "credit_analysis"
        validated_data_path = credit_analysis_folder / "validated_historical_data.json"

        if not validated_data_path.exists():
            logger.error(f"Validated historical data not found at {validated_data_path} for security {security_id}")
            # Return an error dictionary instead of raising FileNotFoundError directly
            # to allow the MCP tool to handle it gracefully as a JSON response.
            return {"error": f"Validated historical data not found at {validated_data_path}", "status": "FileNotFoundError"}

        with open(validated_data_path, "r", encoding="utf-8") as f:
            validated_data = json.load(f)
        logger.info(f"Loaded validated historical data from {validated_data_path}")

        reclassified_data = copy.deepcopy(validated_data)

        def _apply_plug_reallocation(statement_data: dict, period_date: str, plug_item_name: str, target_candidates: list, notes_list: list):
            """Helper function to reallocate a plug value to target items using weighted distribution and log it."""
            # Initialize target to None if not found
            if plug_item_name not in statement_data:
                logger.debug(f"Plug item {plug_item_name} not found in statement data")
                return
                
            plug_value = statement_data[plug_item_name]
            logger.debug(f"\n=== Processing {plug_item_name} = {plug_value:,.2f} for {period_date} ===")
            
            if plug_value is None or plug_value == 0:
                logger.debug(f"Skipping {plug_item_name} - zero or no value")
                return

            logger.debug(f"Target candidates: {target_candidates}")
            
            # Get weights for this plug type from PLUG_WEIGHTS
            weights = PLUG_WEIGHTS.get(plug_item_name, {})
            logger.debug(f"Weights for {plug_item_name}: {weights}")
            
            # If no weights defined, use equal distribution among target_candidates
            if not weights:
                logger.debug(f"No weights defined for {plug_item_name}, using equal distribution")
                weights = {t: 1.0 for t in target_candidates}
            
            logger.debug(f"Using weights: {weights}")
            logger.debug(f"Raw statement data for {period_date}: {statement_data}")
            
            # Initialize any missing targets with zero
            for target in weights.keys():
                if target not in statement_data:
                    statement_data[target] = 0.0
                    logger.debug(f"Initialized missing target field {target} to 0.0")
                logger.debug(f"Current value of {target}: {statement_data.get(target)}")
            
            # Filter valid targets (must be in statement_data and have numeric values)
            valid_targets = {}
            for target, weight in weights.items():
                if target not in statement_data or not isinstance(statement_data[target], (int, float)):
                    logger.debug(f"Skipping invalid target: {target} (value: {statement_data.get(target)}) or not numeric")
                else:
                    valid_targets[target] = weight
                    logger.debug(f"Valid target: {target} (value: {statement_data.get(target)})")
            
            logger.debug(f"Valid weighted targets for {plug_item_name}: {valid_targets}")
            
            if not valid_targets:
                error_msg = f"No valid weighted targets found for {plug_item_name}"
                logger.error(error_msg)
                notes_list.append(f"[ERROR] {error_msg}")
                return
                
            # Calculate total weight for normalization
            total_weight = sum(valid_targets.values())
            if total_weight <= 0:
                error_msg = f"Invalid total weight {total_weight} for {plug_item_name}"
                logger.error(error_msg)
                notes_list.append(f"[ERROR] {error_msg}")
                return

            logger.debug(f"Total weight for {plug_item_name}: {total_weight}")
            if total_weight == 0:
                logger.debug(f"Total weight is zero for {plug_item_name}. Skipping reallocation.")
                return

            # Calculate distribution amounts
            total_weight = sum(valid_targets.values())
            # Sort targets by weight in descending order to handle rounding errors better
            sorted_targets = sorted(valid_targets.items(), key=lambda x: -x[1])
            
            # Distribute amounts to each target
            for i, (target, weight) in enumerate(sorted_targets):
                # Calculate the exact amount based on weight proportion
                proportional_amount = plug_value * (weight / total_weight)
                logger.debug(f"Calculated proportional amount for {target}: {proportional_amount} (weight: {weight})")
                # Round to 2 decimal places for financial accuracy
                amount_to_distribute = round(proportional_amount, 2)
                logger.debug(f"Rounded amount to distribute for {target}: {amount_to_distribute}")
                
                # Skip if amount is effectively zero (avoiding -0.0 or tiny values)
                if abs(amount_to_distribute) < 0.01:
                    continue
                
                original = statement_data[target]
                statement_data[target] = round(statement_data[target] + amount_to_distribute, 2)
                
                notes_list.append(
                    f"Reallocated {abs(amount_to_distribute):,.2f} from {plug_item_name} to {target} "
                    f"({original:,.2f} → {statement_data[target]:,.2f}) at {weight/total_weight:.1%} weight"
                )
                logger.debug(f"Reallocated {amount_to_distribute} from {plug_item_name} to {target} (weight: {weight/total_weight:.1%})")
                



                # Update total_equity if this is an equity plug
                if plug_item_name == "balance_sheet_equation_equity_plug" and plug_value != 0:
                    original_equity = statement_data.get("total_equity", 0)
                    statement_data["total_equity"] = original_equity + plug_value
                    logger.debug(f"Updated total_equity: {original_equity:,.2f} → {statement_data['total_equity']:,.2f}")
                    notes_list.append(f"Updated total_equity by adding {plug_value:,.2f} from {plug_item_name}")
                
                statement_data[plug_item_name] = 0.0  # Clear the plug

        # Define which plugs belong to which statement type
        statement_plugs = {
            'income_statement': [
                'reported_vs_calculated_operating_expenses_plug',
                'reported_vs_calculated_profit_before_tax_plug'
            ],
            'balance_sheet': [
                'reported_vs_calculated_total_assets_plug',
                'reported_vs_subtotals_total_liabilities_plug',
                'balance_sheet_equation_equity_plug'
            ],
            'cash_flow_statement': [
                'change_in_bs_asset_plug',
                'change_in_bs_liability_plug',
                'change_in_bs_equity_plug'
            ]
        }

        # Apply plug reallocation for each statement type
        for period_data in reclassified_data.get("mapped_historical_data", []):
            period_date = period_data.get('reporting_period_end_date', 'Unknown Period')
            
            # Ensure reclassification_notes is a list, initialize if not present or not a list
            period_notes = period_data.setdefault("reclassification_notes", [])
            if not isinstance(period_notes, list):
                logger.warning(f"Period: {period_date}, 'reclassification_notes' was not a list. Reinitializing.")
                period_notes = []
                period_data["reclassification_notes"] = period_notes
            
            # Process each statement type
            for stmt_type, plug_list in statement_plugs.items():
                stmt_data = period_data.get(stmt_type, {})
                if not isinstance(stmt_data, dict):
                    continue
                    
                logger.info(f"\n{'='*80}")
                logger.info(f"Processing {stmt_type.replace('_', ' ').title()} for {period_date}")
                logger.info(f"Initial {stmt_type} Keys: {list(stmt_data.keys())}")
                
                # Log initial plug values
                for plug_item in plug_list:
                    if plug_item in stmt_data:
                        logger.info(f"Initial {plug_item}: {stmt_data[plug_item]}")
                
                # Process each plug for this statement type
                for plug_item in plug_list:
                    if plug_item not in stmt_data:
                        continue
                        
                    # Get target candidates from PLUG_WEIGHTS
                    target_candidates = list(PLUG_WEIGHTS.get(plug_item, {}).keys())
                    if not target_candidates:
                        logger.warning(f"No target candidates found for plug: {plug_item}")
                        continue
                        
                    _apply_plug_reallocation(
                        statement_data=stmt_data,
                        period_date=period_date,
                        plug_item_name=plug_item,
                        target_candidates=target_candidates,
                        notes_list=period_notes
                    )
                    
                    # Log after each plug is processed
                    if plug_item in stmt_data:
                        logger.info(f"After processing {plug_item}: {stmt_data[plug_item]}")
                        logger.info(f"Notes added: {period_notes[-1] if period_notes else 'None'}")
                
                logger.info(f"Final {stmt_type} Keys: {list(stmt_data.keys())}")

        # Verify all plugs were zeroed out
        for period_data in reclassified_data.get("mapped_historical_data", []):
            period_date = period_data.get('reporting_period_end_date', 'Unknown Period') # Default for logging if date is missing
            remaining_plugs = {
                k: v for k, v in period_data.items() 
                if k.endswith('_plug') and v != 0 and v is not None
            }
            if remaining_plugs:
                error_msg = f"Failed to reallocate plugs: {remaining_plugs}"
                logger.error(error_msg)
                period_data.setdefault("reclassification_notes", []).append(f"[ERROR] {error_msg}")

        output_path = credit_analysis_folder / "reclassified_historical_data.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(reclassified_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully completed deterministic plug reclassification for security {security_id} and saved to {output_path}")
        return reclassified_data

    except FileNotFoundError as e: # This specific catch might be redundant if handled above, but good for defense
        logger.error(f"File not found during plug reclassification for {security_id}: {e}")
        return {"error": str(e), "status": "FileNotFoundError"}
    except Exception as e:
        logger.exception(f"An unexpected error occurred during plug reclassification for security {security_id}: {e}")
        # Return an error dictionary. Include partial data if it exists and might be useful.
        error_response = {"error": str(e), "status": "Error"}
        if reclassified_data is not None: # Check if reclassified_data was initialized (i.e., after deepcopy)
            error_response["reclassified_data_partial"] = reclassified_data
        return error_response
    finally:
        logger.info(f"Finished plug reclassification attempt for security {security_id}.")


@mcp_app.tool()
async def run_financial_model(security_id: str) -> dict:
    """
    Deterministic financial model for a bond security. Applies conservative logic and explicit inflation/FX adjustments using real extracted and assumptions data.
    Returns a partial output with the macroeconomic_adjustments section fully populated.
    """
    return await _run_financial_model(security_id)

@mcp_app.tool()
async def stress_test_financial_model(security_id: str) -> dict:
    """
    Generate stress test scenarios for a financial model and run the model with modified assumptions.
    Creates a stress_test subfolder with scenario definitions and results.
    
    Args:
        security_id: The security ID to stress test
        
    Returns:
        Dictionary containing the scenarios and their results
    """
    try:
        # Create stress_test subfolder
        security_folder = get_subfolder(security_id, "credit_analysis")
        stress_test_folder = security_folder / "stress_test"
        
        # Clean existing stress_test folder if it exists
        import shutil
        if stress_test_folder.exists():
            logger.info(f"Cleaning existing stress_test folder: {stress_test_folder}")
            shutil.rmtree(stress_test_folder)
        
        # Create fresh stress_test folder
        stress_test_folder.mkdir(exist_ok=True)
        
        # Load financial model data
        financial_model_path = security_folder / "financial_model.json"
        if not financial_model_path.exists():
            raise FileNotFoundError(f"Financial model not found at {financial_model_path}")
        
        with open(financial_model_path, "r", encoding="utf-8") as f:
            financial_model = json.load(f)
        
        # Load industry research
        industry_research_path = Path(__file__).parent.parent.parent / "external_reports" / "industry_research.json"
        if not industry_research_path.exists():
            logger.warning(f"Industry research not found at {industry_research_path}")
            industry_research = {}
        else:
            with open(industry_research_path, "r", encoding="utf-8") as f:
                industry_research = json.load(f)
        
        # Load country research
        country_research_path = Path(__file__).parent.parent.parent / "external_reports" / "country_research.json"
        if not country_research_path.exists():
            logger.warning(f"Country research not found at {country_research_path}")
            country_research = {}
        else:
            with open(country_research_path, "r", encoding="utf-8") as f:
                country_research = json.load(f)
        
        # Get original assumptions
        assumptions = financial_model.get("assumptions_generation_data", {})
        
        # Generate scenarios using LLM
        scenarios = await _generate_stress_scenarios_with_llm(
            financial_model=financial_model,
            industry_research=industry_research,
            country_research=country_research
        )
        
        # Save scenarios to file
        scenarios_path = stress_test_folder / "scenarios.json"
        with open(scenarios_path, "w", encoding="utf-8") as f:
            json.dump(scenarios, f, indent=2, ensure_ascii=False)
        
        # Process each scenario
        results = {
            "base_case": financial_model.get("financial_model_data", {}),
            "scenarios": []
        }
        
        for scenario in scenarios["scenarios"]:
            scenario_name = scenario["name"].replace(" ", "_").lower()
            scenario_folder = stress_test_folder / scenario_name
            scenario_folder.mkdir(exist_ok=True)
            
            # Create modified assumptions
            modified_assumptions = _modify_assumptions(
                original_assumptions=assumptions,
                modifications=scenario["assumption_modifications"]
            )
            
            # Save modified assumptions
            assumptions_path = scenario_folder / "assumptions.json"
            with open(assumptions_path, "w", encoding="utf-8") as f:
                json.dump(modified_assumptions, f, indent=2, ensure_ascii=False)
            
            # Run financial model with modified assumptions
            # We'll need to temporarily replace the assumptions file
            original_assumptions_path = security_folder / "assumptions.json"
            temp_assumptions_path = None
            
            if original_assumptions_path.exists():
                # Backup original assumptions
                temp_assumptions_path = security_folder / "assumptions.json.bak"
                original_assumptions_path.rename(temp_assumptions_path)
            
            try:
                # Copy modified assumptions to the main folder
                with open(original_assumptions_path, "w", encoding="utf-8") as f:
                    json.dump(modified_assumptions, f, indent=2, ensure_ascii=False)
                
                # Run the financial model
                stress_model_result = await _run_financial_model(security_id)
                
                # Save the result
                model_path = scenario_folder / "financial_model.json"
                with open(model_path, "w", encoding="utf-8") as f:
                    json.dump(stress_model_result, f, indent=2, ensure_ascii=False)
                
                # Add to results
                scenario_result = {
                    "name": scenario["name"],
                    "description": scenario["description"],
                    "financial_model_data": stress_model_result.get("financial_model_data", {})
                }
                results["scenarios"].append(scenario_result)
                
            finally:
                # Restore original assumptions
                if temp_assumptions_path:
                    original_assumptions_path.unlink(missing_ok=True)
                    temp_assumptions_path.rename(original_assumptions_path)
        
        # Save combined results
        results_path = stress_test_folder / "stress_test_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results
    
    except Exception as e:
        logger.error(f"Error in stress testing: {e}")
        raise


@mcp_app.tool()
async def run_deterministic_scenarios(security_id: str) -> dict:
    """
    Run predefined deterministic scenarios for a financial model and capture results for correlation analysis.
    Creates a deterministic_scenarios subfolder with scenario results.
    
    Args:
        security_id: The security ID to analyze with deterministic scenarios
        
    Returns:
        List of scenario results following the deterministic scenario output schema
    """
    try:
        # Create deterministic_scenarios subfolder
        security_folder = get_subfolder(security_id, "credit_analysis")
        det_scenarios_folder = security_folder / "deterministic_scenarios"
        
        # Clean existing deterministic_scenarios folder if it exists
        import shutil
        if det_scenarios_folder.exists():
            logger.info(f"Cleaning existing deterministic_scenarios folder: {det_scenarios_folder}")
            shutil.rmtree(det_scenarios_folder)
        
        # Create fresh deterministic_scenarios folder
        det_scenarios_folder.mkdir(exist_ok=True)
        
        # Load financial model data
        financial_model_path = security_folder / "financial_model.json"
        if not financial_model_path.exists():
            raise FileNotFoundError(f"Financial model not found at {financial_model_path}")
        
        with open(financial_model_path, "r", encoding="utf-8") as f:
            financial_model = json.load(f)
        
        # Get original assumptions
        assumptions = financial_model.get("assumptions_generation_data", {})
        
        # Load deterministic scenarios
        scenarios_path = Path(__file__).parent / "deterministic_scenarios.json"
        if not scenarios_path.exists():
            raise FileNotFoundError(f"Deterministic scenarios not found at {scenarios_path}")
        
        with open(scenarios_path, "r", encoding="utf-8") as f:
            deterministic_scenarios = json.load(f)
        
        # Process each scenario
        results = []
        
        # First run the base case (current_trend) to establish baseline
        base_scenario = deterministic_scenarios["scenarios"]["current_trend"]
        base_scenario_name = "current_trend"
        
        # Process all scenarios including base case
        for scenario_name, scenario_data in deterministic_scenarios["scenarios"].items():
            logger.info(f"Processing deterministic scenario: {scenario_name}")
            scenario_folder = det_scenarios_folder / scenario_name
            scenario_folder.mkdir(exist_ok=True)
            
            # Save scenario definition
            scenario_def_path = scenario_folder / "scenario_definition.json"
            with open(scenario_def_path, "w", encoding="utf-8") as f:
                json.dump(scenario_data, f, indent=2, ensure_ascii=False)
            
            # Create modified assumptions based on deltas
            modified_assumptions = await _apply_deterministic_scenario_deltas(
                original_assumptions=assumptions,
                scenario_deltas=scenario_data["assumptions_by_category_deltas"]
            )
            
            # Save modified assumptions
            assumptions_path = scenario_folder / "assumptions.json"
            with open(assumptions_path, "w", encoding="utf-8") as f:
                json.dump(modified_assumptions, f, indent=2, ensure_ascii=False)
            
            # Run financial model with modified assumptions
            # We'll need to temporarily replace the assumptions file
            original_assumptions_path = security_folder / "assumptions.json"
            temp_assumptions_path = None
            
            if original_assumptions_path.exists():
                # Backup original assumptions
                temp_assumptions_path = security_folder / "assumptions.json.bak"
                original_assumptions_path.rename(temp_assumptions_path)
            
            try:
                # Copy modified assumptions to the main folder
                with open(original_assumptions_path, "w", encoding="utf-8") as f:
                    json.dump(modified_assumptions, f, indent=2, ensure_ascii=False)
                
                # Run the financial model
                scenario_model_result = await _run_financial_model(security_id)
                
                # Save the result
                model_path = scenario_folder / "financial_model.json"
                with open(model_path, "w", encoding="utf-8") as f:
                    json.dump(scenario_model_result, f, indent=2, ensure_ascii=False)
                
                # Add debugging for the financial model data structure
                logging.info(f"Financial model data keys: {scenario_model_result.keys()}")
                if "financial_model_data" in scenario_model_result:
                    logging.info(f"Financial model data structure: {list(scenario_model_result['financial_model_data'].keys())}")
                    if "periods" in scenario_model_result["financial_model_data"]:
                        logging.info(f"Number of periods: {len(scenario_model_result['financial_model_data']['periods'])}")
                        logging.info(f"Period types: {[p.get('reporting_period_type', 'unknown') for p in scenario_model_result['financial_model_data']['periods']]}")
                
                # Extract required metrics for output schema
                financial_metrics = _extract_deterministic_scenario_metrics(
                    scenario_model_result.get("financial_model_data", {})
                )
                
                # Add to results
                scenario_result = {
                    "scenario_name": scenario_name,
                    "financial_metrics": financial_metrics
                }
                results.append(scenario_result)
                
            finally:
                # Restore original assumptions
                if temp_assumptions_path:
                    original_assumptions_path.unlink(missing_ok=True)
                    temp_assumptions_path.rename(original_assumptions_path)
        
        # Save combined results
        results_path = det_scenarios_folder / "deterministic_scenario_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results
    
    except Exception as e:
        logger.error(f"Error in deterministic scenario analysis: {e}")
        raise


@mcp_app.tool()
async def summarize_credit_risk(security_id: str) -> dict:
    """
    LLM interprets model output, produces narrative summary and risk assessment.
    Automatically loads financial model output and stress test results if available.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage
    import config
    
    # Define schema path
    schema_dir = Path(__file__).parent
    CREDIT_SUMMARY_OUTPUT_SCHEMA_PATH = str(schema_dir / "credit_summary_output_schema.json")
    
    logger.info(f"Generating credit risk summary for security {security_id} using LLM...")
    
    try:
        # Load financial model output from the security folder
        security_folder = require_security_folder(security_id)
        analysis_folder = security_folder / "credit_analysis"
        financial_model_output_path = analysis_folder / "financial_model.json"
        
        if not financial_model_output_path.exists():
            raise FileNotFoundError(f"Financial model output not found at {financial_model_output_path}")
            
        with open(financial_model_output_path, "r", encoding="utf-8") as f:
            financial_model_data = json.load(f)
            # Extract the financial model output from the nested structure
            financial_model_output = financial_model_data
        
        # Load qualitative inputs from the security folder
        data_extraction_folder = security_folder / "data_extraction"
        complete_data_path = data_extraction_folder / "complete_data.json"
        
        if not complete_data_path.exists():
            raise FileNotFoundError(f"Complete data not found at {complete_data_path}")
            
        with open(complete_data_path, "r", encoding="utf-8") as f:
            complete_data = json.load(f)
            
        qualitative_inputs = {
            "issuer_business_profile": complete_data.get("issuer_business_profile", {}),
            "industry_profile": complete_data.get("industry_profile", {}),
            "collateral_and_protective_clauses": complete_data.get("collateral_and_protective_clauses", {})
        }
        
        # Load stress test results if available
        stress_test_folder = analysis_folder / "stress_test"
        stress_test_results_path = stress_test_folder / "stress_test_results.json"
        
        if stress_test_results_path.exists():
            logger.info(f"Found stress test results at {stress_test_results_path}")
            with open(stress_test_results_path, "r", encoding="utf-8") as f:
                stress_test_results = json.load(f)
        else:
            logger.info(f"No stress test results found at {stress_test_results_path}")
            stress_test_results = None
        
        # Initialize LLM - use the same model as assumptions
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-04-17",
            temperature=0.0,
            top_p=getattr(config, "LLM_TOP_P", 1.0),
            google_api_key=getattr(config, "GOOGLE_API_KEY", None),
        )
        
        # Prepare stress test data for LLM
        stress_test_summary = None
        if stress_test_results is not None:
            logger.info("Stress test results provided, incorporating into credit risk summary")
            
            # Extract the most relevant stress test data
            stress_test_data = stress_test_results
            
            # Format stress test summary for LLM
            scenarios = []
            
            # Include base case key metrics
            base_case = stress_test_data.get("base_case", {})
            base_case_metrics = _extract_key_metrics(base_case)
            
            # Process each scenario
            for idx, scenario in enumerate(stress_test_data.get("scenarios", [])):
                scenario_metrics = _extract_key_metrics(scenario.get("financial_model_data", {}))
                
                # Calculate percentage changes from base case
                changes = {}
                for key, value in scenario_metrics.items():
                    if key in base_case_metrics and base_case_metrics[key] != 0:
                        pct_change = (value - base_case_metrics[key]) / base_case_metrics[key] * 100
                        changes[key] = f"{pct_change:.1f}%"
                
                scenarios.append({
                    "name": scenario.get("name", f"Scenario {idx+1}"),
                    "description": scenario.get("description", ""),
                    "key_metrics": scenario_metrics,
                    "changes_from_base": changes
                })
            
            stress_test_summary = {
                "base_case": base_case_metrics,
                "scenarios": scenarios
            }
        
        # Create prompt for LLM
        prompt = _create_credit_summary_prompt(financial_model_output, stress_test_summary)
        
        # Call LLM
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        content = response.content
        
        # Parse LLM response
        try:
            # Clean up the response if it contains markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            credit_summary = json.loads(content)
            
            # Save the credit risk summary to the security folder
            credit_summary_output = {"credit_summary": credit_summary}
            credit_summary_path = analysis_folder / "credit_risk_summary.json"
            with open(credit_summary_path, "w", encoding="utf-8") as f:
                json.dump(credit_summary_output, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Credit risk summary saved to {credit_summary_path}")
            
            return {"credit_summary": credit_summary, "_schema": CREDIT_SUMMARY_OUTPUT_SCHEMA_PATH}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Raw response: {content}")
            # Create a default structure with error message
            error_summary = {
                "credit_summary": {
                    "error": "Failed to parse LLM response",
                    "raw_response": content[:500]  # Truncate to avoid excessive output
                }
            }
            
            # Save the error summary to the security folder
            credit_summary_path = analysis_folder / "credit_risk_summary.json"
            with open(credit_summary_path, "w", encoding="utf-8") as f:
                json.dump(error_summary, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Error credit risk summary saved to {credit_summary_path}")
            
            # Return the error structure
            return {
                "credit_summary": error_summary["credit_summary"],
                "_schema": CREDIT_SUMMARY_OUTPUT_SCHEMA_PATH
            }
    
    except Exception as e:
        logger.error(f"Error in credit risk summarization: {e}")
        raise


def _extract_key_metrics(financial_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract key financial metrics from financial model data.
    Focus on the most relevant metrics for credit risk assessment.
    """
    metrics = {}
    
    # Extract unified financial statements for the most recent period
    statements = financial_data.get("unified_financial_statements", [])
    if statements:
        # Sort by date descending to get most recent first
        statements = sorted(
            statements, 
            key=lambda x: x.get("reporting_period_end_date", ""), 
            reverse=True
        )
        
        # Get the most recent statement
        recent = statements[0]
        financials = recent.get("financials", {})
        
        # Extract key credit metrics
        metrics.update({
            "net_income": financials.get("net_income", 0),
            "ebitda": financials.get("ebitda", 0),
            "total_assets": financials.get("total_assets", 0),
            "total_debt": financials.get("total_debt", 0),
            "total_equity": financials.get("total_equity", 0),
            "revenue": financials.get("revenue", 0),
            "interest_expense": financials.get("interest_expense", 0),
        })
        
        # Calculate key ratios
        if metrics["total_debt"] > 0 and metrics["ebitda"] > 0:
            metrics["debt_to_ebitda"] = metrics["total_debt"] / metrics["ebitda"]
        else:
            metrics["debt_to_ebitda"] = 0
            
        if metrics["interest_expense"] > 0 and metrics["ebitda"] > 0:
            metrics["interest_coverage"] = metrics["ebitda"] / metrics["interest_expense"]
        else:
            metrics["interest_coverage"] = 0
            
        if metrics["total_assets"] > 0:
            metrics["return_on_assets"] = metrics["net_income"] / metrics["total_assets"]
        else:
            metrics["return_on_assets"] = 0
    
    # Extract key credit metrics
    credit_metrics = financial_data.get("credit_metrics", {})
    for period, metrics_data in credit_metrics.items():
        # Just take the first period we find
        metrics.update({
            "dscr": metrics_data.get("debt_service_coverage_ratio", 0),
            "leverage_ratio": metrics_data.get("leverage_ratio", 0),
            "liquidity_ratio": metrics_data.get("liquidity_ratio", 0),
        })
        break
    
    return metrics


def _extract_deterministic_scenario_metrics(financial_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract metrics required for deterministic scenario output schema.
    
    Args:
        financial_data: Financial model data for a scenario
        
    Returns:
        Dictionary of metrics matching the deterministic scenario output schema
    """
    metrics = {
        "projected_dscr_avg": 0.0,
        "projected_dscr_min": 0.0,
        "projected_debt_to_ebitda_avg": 0.0,
        "projected_interest_coverage_avg": 0.0,
        "projected_ebitda_cagr": 0.0,
        "total_projected_net_income": 0.0,
        "ending_total_equity": 0.0,
        "cash_at_bond_maturity": 0.0,
        "projected_total_assets_end_period": 0.0
    }
    
    logging.info(f"Financial data keys in extraction function: {list(financial_data.keys())}")
    
    # Extract credit ratios from the nested list format
    credit_ratios = financial_data.get("credit_ratios", [])
    if isinstance(credit_ratios, list) and len(credit_ratios) > 0:
        if isinstance(credit_ratios[0], list) and len(credit_ratios[0]) > 0:
            ratio_dict = credit_ratios[0][0]  # Get the first dictionary
            logging.info(f"Credit ratio dict keys: {list(ratio_dict.keys())}")
            
            # Extract DSCR metrics
            if "projected_dscr_avg" in ratio_dict:
                metrics["projected_dscr_avg"] = ratio_dict["projected_dscr_avg"]
            if "projected_dscr_min" in ratio_dict:
                metrics["projected_dscr_min"] = ratio_dict["projected_dscr_min"]
            
            # Extract EBITDA CAGR
            if "projected_ebitda_cagr" in ratio_dict:
                metrics["projected_ebitda_cagr"] = ratio_dict["projected_ebitda_cagr"]
                
            # Extract cash at maturity
            if "cash_at_maturity" in ratio_dict:
                metrics["cash_at_bond_maturity"] = ratio_dict["cash_at_maturity"]
    
    # Extract data from unified financial statements
    unified_statements = financial_data.get("unified_financial_statements", [])
    if unified_statements and isinstance(unified_statements, list):
        logging.info(f"Found {len(unified_statements)} unified financial statements")
        
        # Variables to collect data
        net_income_sum = 0.0
        ebitda_values = []
        debt_to_ebitda_values = []
        interest_coverage_values = []
        projected_periods = []
        
        # Identify projected periods
        for statement in unified_statements:
            if statement.get("period_type", "").startswith("Projected") or statement.get("reporting_period_type", "").startswith("Projected"):
                projected_periods.append(statement)
        
        logging.info(f"Found {len(projected_periods)} projected periods")
        
        # Process each projected period
        for statement in projected_periods:
            # Extract net income
            if "net_income" in statement:
                net_income_sum += statement["net_income"]
            
            # Extract EBITDA for CAGR calculation
            if "ebitda" in statement:
                ebitda_values.append(statement["ebitda"])
            
            # Calculate debt to EBITDA ratio
            if "ebitda" in statement and "total_debt" in statement and statement["ebitda"] > 0:
                debt_to_ebitda = statement["total_debt"] / statement["ebitda"]
                debt_to_ebitda_values.append(debt_to_ebitda)
            
            # Calculate interest coverage ratio
            if "ebitda" in statement and "interest_expense" in statement and statement["interest_expense"] > 0:
                interest_coverage = statement["ebitda"] / statement["interest_expense"]
                interest_coverage_values.append(interest_coverage)
        
        # Set total projected net income
        metrics["total_projected_net_income"] = net_income_sum
        logging.info(f"Total projected net income: {net_income_sum}")
        
        # Calculate EBITDA CAGR if not already set and we have enough values
        if metrics["projected_ebitda_cagr"] == 0.0 and len(ebitda_values) >= 2:
            start_value = ebitda_values[0]
            end_value = ebitda_values[-1]
            n_years = len(ebitda_values) - 1
            
            if start_value > 0 and end_value > 0:
                metrics["projected_ebitda_cagr"] = (end_value / start_value) ** (1 / n_years) - 1
        
        # Calculate average debt to EBITDA ratio
        if debt_to_ebitda_values:
            metrics["projected_debt_to_ebitda_avg"] = sum(debt_to_ebitda_values) / len(debt_to_ebitda_values)
        
        # Calculate average interest coverage ratio
        if interest_coverage_values:
            metrics["projected_interest_coverage_avg"] = sum(interest_coverage_values) / len(interest_coverage_values)
        
        # Get last projected period for ending values
        if projected_periods:
            last_period = projected_periods[-1]
            
            # Extract ending total equity
            if "total_equity" in last_period:
                metrics["ending_total_equity"] = last_period["total_equity"]
                logging.info(f"Ending total equity: {last_period['total_equity']}")
            
            # Extract cash at bond maturity if not already set
            if metrics["cash_at_bond_maturity"] == 0.0 and "cash_and_equivalents" in last_period:
                metrics["cash_at_bond_maturity"] = last_period["cash_and_equivalents"]
            
            # Extract total assets at end period
            if "total_assets" in last_period:
                metrics["projected_total_assets_end_period"] = last_period["total_assets"]
                logging.info(f"Projected total assets end period: {last_period['total_assets']}")
    
    # Add debugging for the run_deterministic_scenarios function
    logging.info(f"Extracted metrics: {metrics}")
    
    return metrics


def _create_credit_summary_prompt(financial_model_data: Dict[str, Any], stress_test_summary: Optional[Dict[str, Any]] = None) -> str:
    """
    Create a prompt for the LLM to generate a credit risk summary.
    Includes stress test results if available.
    """
    # Extract basic company info
    extraction_data = financial_model_data.get("extraction_phase_data", {})
    issuer_profile = extraction_data.get("issuer_business_profile", {})
    issuer_name = issuer_profile.get("issuer_name", "")
    issuer_industry = issuer_profile.get("issuer_industry", "")
    security_id = financial_model_data.get("security_id", "")
    
    # Create base prompt
    prompt = f"""
    You are a credit analyst specializing in financial risk assessment. Your task is to analyze the financial model data for {issuer_name} ({security_id}), a company in the {issuer_industry} industry, and produce a comprehensive credit risk summary.
    
    Based on the financial model data, provide:
    1. A concise business overview
    2. Key financial strengths and weaknesses
    3. Credit risk assessment (including key ratios analysis)
    4. Debt repayment capacity evaluation
    5. Overall credit rating recommendation (using standard S&P/Moody's scale)
    """
    
    # Add stress test section if available
    if stress_test_summary is not None:
        prompt += """
        
        Additionally, analyze the provided stress test scenarios and their impact on the company's credit profile:
        1. For each scenario, assess the severity of impact on key financial metrics
        2. Evaluate which scenarios pose the greatest risk to credit quality
        3. Determine if any scenarios would trigger a credit rating downgrade
        4. Provide recommendations for mitigating risks identified in the stress tests
        """
        
        # Add stress test data
        prompt += """
        
        Here is the stress test summary data:
        """
        
        # Add base case metrics
        prompt += """
        
        Base Case Key Metrics:
        """
        for key, value in stress_test_summary.get("base_case", {}).items():
            prompt += f"\n- {key}: {value}"
        
        # Add scenarios
        for idx, scenario in enumerate(stress_test_summary.get("scenarios", [])):
            prompt += f"""
            
            Scenario {idx+1}: {scenario.get('name', '')}
            Description: {scenario.get('description', '')}
            Key Metrics Changes from Base Case:
            """
            
            for key, value in scenario.get("changes_from_base", {}).items():
                prompt += f"\n- {key}: {value}"
    
    # Add output format instructions
    prompt += """
    
    Provide your analysis in the following JSON format:
    ```json
    {
      "business_overview": "Concise description of the company's business model and market position",
      "financial_strengths": ["Strength 1", "Strength 2"],
      "financial_weaknesses": ["Weakness 1", "Weakness 2"],
      "key_ratios": {
        "debt_to_ebitda": "X.XX with interpretation",
        "interest_coverage": "X.XX with interpretation",
        "dscr": "X.XX with interpretation"
      },
      "credit_risk_assessment": "Detailed credit risk analysis",
      "debt_repayment_capacity": "Analysis of ability to meet debt obligations",
      "rating_recommendation": "BBB-",
      "rating_rationale": "Explanation for the rating"
    """
    
    if stress_test_summary is not None:
        prompt += """,
      "stress_test_analysis": {
        "scenario_impacts": [
          {
            "scenario_name": "Scenario 1 name",
            "impact_severity": "High/Medium/Low",
            "key_vulnerabilities": ["Vulnerability 1", "Vulnerability 2"],
            "potential_rating_impact": "Potential downgrade to BB+"
          }
        ],
        "most_severe_scenario": "Name of most concerning scenario",
        "risk_mitigation_recommendations": ["Recommendation 1", "Recommendation 2"]
      }
    """
    
    prompt += """
    }
    ```
    
    Ensure your analysis is data-driven, balanced, and focuses on credit risk implications. Be specific about financial metrics and their trends.
    """
    
    return prompt


async def _generate_stress_scenarios_with_llm(
    financial_model: Dict[str, Any],
    industry_research: Dict[str, Any],
    country_research: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Use LLM to generate stress test scenarios based on the provided data.
    
    Args:
        financial_model: The financial model data
        industry_research: Industry research data
        country_research: Country research data
        
    Returns:
        Dictionary containing the scenarios
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage
    import config
    
    try:
        # Initialize LLM - use the same model as assumptions
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-04-17",
            temperature=0.0,
            top_p=getattr(config, "LLM_TOP_P", 1.0),
            google_api_key=getattr(config, "GOOGLE_API_KEY", None),
        )
        
        # Extract relevant data for the prompt
        security_id = financial_model.get("security_id", "")
        issuer_name = safe_get(financial_model, "extraction_phase_data", "issuer_business_profile", "issuer_name") or ""
        issuer_industry = safe_get(financial_model, "extraction_phase_data", "issuer_business_profile", "issuer_industry") or ""
        
        # Create prompt
        prompt = f"""
        You are a financial risk analyst specializing in stress testing financial models. Your task is to create 3 coherent, realistic stress test scenarios for {issuer_name} ({security_id}), a company in the {issuer_industry} industry.

        Based on the financial model, industry research, and country research provided, identify 3 negative scenarios that could realistically impact this company. For each scenario:
        1. Provide a name and detailed description
        2. Identify specific financial assumptions that would be affected
        3. Specify exactly how each assumption should be modified (provide specific numerical changes)
        4. Explain the rationale for each modification

        # MODELING INSTRUCTIONS
        1. Use the provided historical financial data and assumptions to create a complete financial model
        2. Project financial statements for the periods specified in the assumptions
        3. Create a debt schedule for the specific bond being analyzed
        4. Calculate credit ratios and key metrics
        5. Document your methodology and approach
        6. CRITICAL: For the first projected period (transition from historical to projected):
           - If historical cash data is inconsistent, adjust other historical values to create a consistent starting point
           - NEVER carry forward inconsistencies from historical data into projections
           - Explicitly document any adjustments made to historical data to ensure consistency

        FINANCIAL MODEL SUMMARY:
        {json.dumps(safe_get(financial_model, "financial_model_data", "credit_ratios", 0) or {}, indent=2)}

        INDUSTRY RESEARCH:
        {json.dumps(industry_research, indent=2)}

        COUNTRY RESEARCH:
        {json.dumps(country_research, indent=2)}

        ORIGINAL ASSUMPTIONS:
        {json.dumps(safe_get(financial_model, "assumptions_generation_data", "assumptions_by_category") or {}, indent=2)}

        Output your response as a JSON object with the following structure:
        {{
          "scenarios": [
            {{
              "name": "Scenario name",
              "description": "Detailed scenario description",
              "assumption_modifications": [
                {{
                  "category": "revenue_assumptions",
                  "metric_name": "average_yield_on_loans",
                  "original_values": [0.4, 0.4, 0.4, 0.4, 0.4],
                  "modified_values": [0.35, 0.35, 0.35, 0.35, 0.35],
                  "rationale": "Explanation for why this assumption would change in this scenario"
                }}
              ]
            }}
          ]
        }}

        Ensure all scenarios are realistic, coherent, and based on the specific risks relevant to this company and country.
        """
        
        # Call LLM
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        content = response.content
        
        # Extract JSON from response
        try:
            # Clean up the response if it contains markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            scenarios = json.loads(content)
            return scenarios
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Raw response: {content}")
            # Return a default structure
            return {
                "scenarios": [
                    {
                        "name": "Economic Downturn",
                        "description": "Default scenario due to parsing error",
                        "assumption_modifications": []
                    }
                ]
            }
    
    except Exception as e:
        logger.error(f"Error in LLM scenario generation: {e}")
        raise


@mcp_app.tool()
async def map_historical_data_to_schema(security_id: str) -> dict:
    """
    Maps historical financial data from complete_data.json and external reports to the standardized schema.
    
    1. Loads the historical data from complete_data.json for the specified security
    2. Loads any relevant external reports for context
    3. Uses LLM to map the issuer-specific line items to our standardized schema
    4. Saves the mapped historical data to the credit_analysis folder
    
    Args:
        security_id: The security ID to map historical data for
        
    Returns:
        Dictionary containing the mapped historical financial data
    """
    logger.info("=== Starting map_historical_data_to_schema ===")
    logger.info(f"Starting historical data mapping for security_id: {security_id}")
    
    # Validate security_id and locate folders
    logger.info(f"[1/10] Validating security_id: {security_id}")
    if not security_id:
        raise ValueError("security_id argument must be provided to tool action.")
    logger.info("[1/10] Security ID validation complete")
    
    # Get security folder paths
    logger.info("[2/10] Getting security folder paths")
    security_folder = require_security_folder(security_id)
    logger.info(f"Security folder: {security_folder}")
    
    data_extraction_folder = get_subfolder(security_id, "data_extraction")
    logger.info(f"Data extraction folder: {data_extraction_folder}")
    
    credit_analysis_folder = get_subfolder(security_id, "credit_analysis")
    logger.info(f"Credit analysis folder: {credit_analysis_folder}")
    
    # Ensure credit_analysis folder exists
    logger.info("[3/10] Ensuring credit_analysis folder exists")
    credit_analysis_folder.mkdir(exist_ok=True, parents=True)
    logger.info("Credit analysis folder ready")
    
    # Load complete_data.json
    logger.info("[4/10] Loading complete_data.json")
    data_path = data_extraction_folder / "complete_data.json"
    logger.info(f"Looking for data at: {data_path}")
    
    if not data_path.exists():
        error_msg = f"complete_data.json not found for security_id {security_id} at {data_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info(f"Found complete_data.json at: {data_path}")
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)
        logger.info("Successfully loaded complete_data.json")
        logger.debug(f"Input data keys: {list(input_data.keys())}")
    except Exception as e:
        logger.error(f"Error loading complete_data.json: {str(e)}")
        raise
    
    # Load external reports for additional context
    logger.info("[5/10] Loading external reports")
    external_reports_dir = Path(__file__).parent.parent.parent / "external_reports"
    external_reports_data = {}
    
    if external_reports_dir.exists():
        report_files = list(external_reports_dir.glob("*.json"))
        logger.info(f"Found {len(report_files)} external report(s)")
        
        for report_file in report_files:
            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    external_reports_data[report_file.stem] = json.load(f)
                    logger.info(f"Successfully loaded external report: {report_file.name}")
                    logger.debug(f"Report keys: {list(external_reports_data[report_file.stem].keys())}")
            except Exception as e:
                logger.warning(f"Failed to load external report {report_file.name}: {e}")
    else:
        logger.warning(f"External reports directory not found: {external_reports_dir}")
    
    logger.info("Completed loading external reports")
    
    logger.info("[6/10] Loading bond metadata")
    metadata_path = data_extraction_folder / "bond_metadata.json"
    logger.info(f"Looking for metadata at: {metadata_path}")
    
    if not metadata_path.exists():
        error_msg = f"bond_metadata.json not found for security_id {security_id} at {metadata_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            bond_metadata = json.load(f)
        logger.info("Successfully loaded bond_metadata.json")
        logger.debug(f"Bond metadata keys: {list(bond_metadata.keys())}")
    except Exception as e:
        logger.error(f"Error loading bond_metadata.json: {str(e)}")
        raise
    
    # Get industry type from metadata (default to 'services' if not specified)
    logger.info("[7/10] Determining industry type and loading schema")
    industry_type = bond_metadata.get("bond_metadata", {}).get("industry_type", "services")
    logger.info(f"Using industry type: {industry_type}")
    
    # Map industry type to schema file
    industry_schema_map = {
        "finance": "finance_schema.json",
        "manufacturing": "manufacturing_schema.json",
        "trading": "trading_volatile_schema.json",
        "asset_heavy": "asset_heavy_schema.json",
        "services": "services_schema.json"
    }
    
    # Default to services schema if industry type not found
    schema_file = industry_schema_map.get(industry_type.lower(), "services_schema.json")
    schema_path = Path(__file__).parent.parent / "mcp_data_extractor" / "industry_schemas" / schema_file
    
    logger.info(f"Schema file determined: {schema_file}")
    logger.info(f"Schema path: {schema_path}")
    
    if not schema_path.exists():
        error_msg = f"Schema file not found: {schema_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info("Loading schema file...")
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_data = json.load(f)
        logger.info("Successfully loaded schema file")
        logger.debug(f"Schema data keys: {list(schema_data.keys())}")
    except Exception as e:
        logger.error(f"Error loading schema file: {str(e)}")
        raise
    
    # Get the industry-specific schema
    schema_key = f"{industry_type.lower()}_schema"
    industry_schema = schema_data.get(schema_key, {})
    logger.info(f"Retrieved schema for key: {schema_key}")
    logger.debug(f"Industry schema keys: {list(industry_schema.keys())}")
    
    # Create a clean copy of the schema, preserving summation_plugs
    clean_industry_schema = {}
    for statement in ['income_statement', 'balance_sheet', 'cash_flow_statement']:
        if statement in industry_schema:
            # Keep the statement data as-is, including any summation_plugs
            clean_industry_schema[statement] = industry_schema[statement].copy()
    
    # Create LLM prompt for mapping
    logger.info("[8/10] Preparing LLM prompt")
    
    prompt = f"""
    You are a financial data mapping expert. Your task is to map historical financial data from an issuer's specific format to a standardized schema.
    
    # CRITICAL INSTRUCTIONS:
    - ONLY map the data that is explicitly provided in the source for each period
    - If a field is not present in the source, set it to null
    - DO NOT create any calculated fields or attempt to maintain relationships between fields
    - DO NOT create any 'summation_plugs' or similar adjustments
    - For each period, ONLY include the fields that are present in the source data
    - If a period has limited data (e.g., only net_profit), ONLY include that specific field
    - Never copy data from other periods to fill in missing data
    - If multiple source items could map to the same target field:
        1. Choose the most appropriate source item for the main mapping
        2. In 'mapping_decisions', list ALL source items that could map to this field
        3. For each source item, include its value and a brief explanation of why it was or wasn't chosen
        4. If multiple source items should be combined (e.g., different types of interest income), sum them and document the components
    
    # STANDARDIZED SCHEMA STRUCTURE
    The standardized schema has the following structure for financial statements:
    
    ## Income Statement Items:
    {json.dumps(clean_industry_schema.get('income_statement', {}), indent=2)}
    
    ## Balance Sheet Items:
    {json.dumps(clean_industry_schema.get('balance_sheet', {}), indent=2)}
    
    ## Cash Flow Statement Items:
    {json.dumps(clean_industry_schema.get('cash_flow_statement', {}), indent=2)}
    
    # ISSUER'S HISTORICAL FINANCIAL DATA
    Here is the issuer's historical financial data that needs to be mapped to the standardized schema:
    {json.dumps(input_data.get('bond_financials_historical', {}), indent=2)}
    
    # MAPPING RULES:
    1. For each period, ONLY include fields that are present in the source data
    2. Map each field to the most appropriate field in the standardized schema
    3. If a field doesn't have a clear match, set it to null
    4. Do not calculate or infer any values that aren't explicitly provided
    5. For each mapping, include a brief explanation in the 'mapping_decisions' object
    
    # OUTPUT FORMAT:
    Return a JSON object with the following structure:
    {{
        "mapped_historical_data": [
            {{
                "reporting_period_end_date": "YYYY-MM-DD",
                "reporting_period_type": "Annual/Quarterly",
                "income_statement": {{ ... }},
                "balance_sheet": {{ ... }},
                "cash_flow_statement": {{ ... }},
                "mapping_decisions": {{ ... }},
                "period_specific_notes": "..."
            }}
        ]
    }}
    
    # CROSS-PERIOD CONSISTENCY REQUIREMENTS
    4. CRITICAL: Before mapping individual periods, analyze ALL periods together to identify patterns and components
    
    5. CONSISTENT COMPONENT MAPPING:
       - When the same or similar components appear under different labels across periods, standardize to the most specific schema category
       - Track how components are grouped into totals across different periods
       - Ensure that the sum of mapped components equals the reported total in each period
    
    6. HANDLING TOTALS AND SUBTOTALS:
       - When you encounter a total or subtotal, first try to identify and map its components
       - Only map to a total line item if you cannot reasonably break it down further
       - Document any assumptions made when breaking down totals into components
    
    7. STANDARDIZE SIGN CONVENTIONS:
       - Expenses, losses, contra-assets, and dividends should be negative
       - Revenues, gains, assets, and liabilities should be positive
    
    8. HANDLE STRUCTURAL INCONSISTENCIES:
       - When an issuer changes reporting format between periods, map to consistent components
       - Document any assumptions made about how components map across different reporting formats
    
    9. DOCUMENTATION REQUIREMENTS:
       - For each period, document how totals were broken down into components
       - Explain any assumptions made about the composition of totals
       - Note any cases where components couldn't be determined and had to be mapped to a total
    
    # OUTPUT FORMAT
    Return a JSON object with the following structure:
    {{
        "mapped_historical_data": [
            // Array of UnifiedFinancialStatement objects, one for each historical period
            {{
                "reporting_period_end_date": "YYYY-MM-DD",
                "reporting_period_type": "Annual", // or Quarterly, Semi-Annual
                "income_statement": {{ ... }}, // Mapped income statement items
                "balance_sheet": {{ ... }}, // Mapped balance sheet items
                "cash_flow_statement": {{ ... }}, // Mapped cash flow statement items
                "mapping_decisions": {{ // Explanations of mapping decisions
                    "item_name": "Explanation of why this item was mapped to this category"
                }},
                "period_specific_notes": "Any period-specific considerations or anomalies"
            }}
        ]
    }}
    """
    import config
    
    # Initialize LLM
    logger.info("[9/10] Initializing LLM client")
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-04-17",
            temperature=0.0,
            top_p=getattr(config, "LLM_TOP_P", 1.0),
            google_api_key=getattr(config, "GOOGLE_API_KEY", None),
        )
        logger.info("LLM client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize LLM client: {str(e)}")
        raise
    
    # Call LLM for mapping
    logger.info("[10/10] Calling LLM to map historical data to standardized schema...")
    logger.debug(f"Prompt length: {len(prompt)} characters")
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        logger.info("LLM API call completed successfully")
        logger.debug(f"Response type: {type(response)}")
    except Exception as e:
        logger.error(f"LLM API call failed: {str(e)}")
        raise
    response_content = response.content
    
    # Extract JSON from response
    logger.info("Processing LLM response")
    mapped_data = {}
    response_content = response.content if hasattr(response, 'content') else str(response)
    logger.debug(f"Response content type: {type(response_content)}")
    logger.debug(f"First 500 chars of response: {response_content[:500]}...")
    
    pattern = r'```(?:json)?\s*({[\s\S]*?})\s*```'
    logger.debug("Searching for JSON in response using pattern")
    matches = re.findall(pattern, response_content)
    logger.info(f"Found {len(matches)} JSON block(s) in response")
    
    if matches:
        logger.info("Attempting to parse JSON from matched block")
        try:
            mapped_data = json.loads(matches[0])
            logger.info("Successfully extracted JSON from LLM response")
            logger.debug(f"Mapped data keys: {list(mapped_data.keys())}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            # Try to extract any JSON object
            try:
                logger.info("Attempting fallback JSON extraction")
                # Find the start of a JSON object
                start_idx = response_content.find('{')
                end_idx = response_content.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = response_content[start_idx:end_idx+1]
                    logger.debug(f"Extracted JSON string (first 200 chars): {json_str[:200]}...")
                    mapped_data = json.loads(json_str)
                    logger.info("Successfully extracted JSON using fallback method")
                else:
                    logger.error("Could not find valid JSON object markers in response")
            except Exception as e2:
                logger.error(f"Fallback JSON extraction failed: {e2}")
                logger.error(f"Problematic JSON string: {json_str[:500]}..." if 'json_str' in locals() else "No JSON string was extracted")
                raise ValueError(f"Could not extract valid JSON from LLM response: {e2}")
    else:
        # If no JSON block found, try to parse the entire response as JSON
        logger.info("No JSON block markers found, attempting to parse entire response as JSON")
        try:
            mapped_data = json.loads(response_content)
            logger.info("Successfully parsed entire LLM response as JSON")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse entire response as JSON: {e}")
            logger.error(f"Response content (first 500 chars): {response_content[:500]}...")
            raise ValueError(f"LLM response did not contain valid JSON: {e}")
    
    # Save mapped data to credit_analysis folder
    logger.info("Saving mapped historical data to file")
    output_path = credit_analysis_folder / "mapped_historical_data.json"
    logger.info(f"Output path: {output_path}")
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(mapped_data, f, indent=2)
        logger.info(f"Successfully saved mapped historical data to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save mapped historical data: {str(e)}")
        raise
    
    logger.info("=== map_historical_data_to_schema completed successfully ===")
    return mapped_data
    return mapped_data


@mcp_app.tool()
async def generate_jit_validation_schema(security_id: str) -> dict:
    """
    Analyzes historical financial data and generates a bespoke JIT validation schema for that issuer.
    
    1. Loads the issuer's historical financial data.
    2. Loads the universal master schema template that defines the rules for schema generation.
    3. Uses an LLM to analyze the data and generate a bespoke validation schema.
    4. Saves the generated schema to the credit_analysis folder for use by the Data Validator.
    
    Args:
        security_id: The security ID for which to generate a validation schema.
        
    Returns:
        A dictionary containing the generated validation schema.
    """
    logger.info("=== Starting generate_jit_validation_schema ===")
    logger.info(f"Starting JIT schema generation for security_id: {security_id}")
    
    # 1. Validate security_id and get paths
    logger.info(f"[1/7] Validating security_id: {security_id}")
    if not security_id:
        raise ValueError("security_id argument must be provided")
    
    # Import from the correct relative path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from financial_analyst.security_folder_utils import find_security_folder, get_subfolder, get_security_file
    
    # Find the security folder
    security_folder = find_security_folder(security_id)
    if not security_folder:
        raise ValueError(f"Security folder not found for {security_id}")
    
    # Get credit analysis folder
    credit_analysis_folder = security_folder / "credit_analysis"
    credit_analysis_folder.mkdir(exist_ok=True, parents=True)
    
    # 2. Load historical financial data from the correct path
    logger.info("[2/7] Loading issuer's historical financial data")
    historical_data_path = security_folder / "data_extraction" / "complete_data.json"
    if not historical_data_path.exists():
        error_msg = f"complete_data.json not found for security_id {security_id} at {historical_data_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        with open(historical_data_path, "r", encoding="utf-8") as f:
            historical_data = json.load(f)
        logger.info("Successfully loaded historical data.")
    except Exception as e:
        logger.error(f"Error loading historical data: {str(e)}")
        raise

    # 3. Load industry and country research data
    logger.info("[3/7] Loading industry and country research data")
    industry_research_path = Path("/home/me/CascadeProjects/Alpy/subagents/financial_analyst/external_reports/industry_research.json")
    country_research_path = Path("/home/me/CascadeProjects/Alpy/subagents/financial_analyst/external_reports/country_research.json")
    
    industry_research = {}
    if industry_research_path.exists():
        with open(industry_research_path, "r", encoding="utf-8") as f:
            industry_research = json.load(f)
            
    country_research = {}
    if country_research_path.exists():
        with open(country_research_path, "r", encoding="utf-8") as f:
            country_research = json.load(f)
    
    # 4. Load the Universal Validator Schema
    logger.info("[4/7] Loading the Universal Validator Schema")
    template_path = Path(__file__).parent / "schemas" / "universal_validator_schema.json"
    logger.info(f"Looking for validator schema at: {template_path}")

    if not template_path.exists():
        error_msg = f"universal_validator_schema.json not found at {template_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            master_template = json.load(f)
        logger.info("Successfully loaded Universal Validator Schema.")
    except Exception as e:
        logger.error(f"Error loading validator schema: {str(e)}")
        raise

    # 5. Prepare the LLM prompt for schema generation
    logger.info("[5/7] Preparing LLM prompt for schema generation")
    
    # Prepare industry and country research sections if data exists
    industry_research_section = "No industry research data available."
    if industry_research:
        industry_research_section = json.dumps(industry_research, indent=2)
    
    country_research_section = "No country research data available."
    if country_research:
        country_research_section = json.dumps(country_research, indent=2)
    
    prompt = f"""
    You are a Master Financial Analyst and Schema Architect. Your task is to analyze an issuer's historical financial data and generate a bespoke JSON validation schema for it. The schema you create must be a precise, logically consistent, and computationally sound description of the provided data for use by downstream deterministic engines.

    # CORE TASK:
    Your output will have two parts:
    1.  A "Chain-of-Thought Analysis" section in plain text where you outline your key findings and decisions.
    2.  The final, complete JSON schema object.

    # CHAIN-OF-THOUGHT ANALYSIS (First, provide this section)
    Before generating the JSON, provide a brief analysis covering these points:
    -   **Company Profile:** Based on the data, what is the most likely business model? (e.g., "This appears to be a lending institution based on...")
    -   **Key Hierarchy Decisions:** Justify one or two complex hierarchical (`subtotal_of`) decisions based on the "Formulaic Assertions" and "Hierarchy Logic" rules below.
    -   **Identified Inconsistencies:** Note any data inconsistencies you observed that your schema will help correct. (e.g., "The sign for 'interest_expense' was positive in one period. The schema will enforce the standard 'negative' sign.")

    # JSON SCHEMA GENERATION (Second, provide the complete JSON schema)

    ## CORE INSTRUCTIONS:
    1.  **Direct Mirroring:** For every unique line item key found in the issuer's data for a given statement, you MUST create a corresponding key in the output schema for that statement. Do not create entries for items that do not appear in the source data for a statement, with the sole exception of the mandatory CFS items listed below.
    2.  **Use the Master Template:** You MUST strictly adhere to the structure and allowed enumeration values defined in the `__metadata_definitions__` section of the provided "UNIVERSAL VALIDATOR SCHEMA".
    3.  **No Data Transformation:** You MUST NOT alter, rename, or map any of the original line item keys. Your output is a description of the data, not a transformation of it.

    ## HIERARCHY AND ARITHMETIC LOGIC (These are non-negotiable rules)
    1.  **"The Law of Absolute Totals":** Top-level `total` items, especially `total_assets`, `total_liabilities_and_equity`, and `net_profit`, represent the absolute end of a summation path. Their `subtotal_of` field **must always** be `null`. There are no exceptions to this rule.
    2.  **"Rule of Levels":** If a line item's name appears in another item's `subtotal_of` field, its own `level` **MUST** be `subtotal` or `total`. An item with `level: component` **CANNOT** be a parent to any other item.
    3.  **"Parental Exclusivity Rule":** If an item (e.g., `loans_receivable_gross`) is a component of a subtotal (e.g., `loans_receivable_net`), it **CANNOT** also be a direct component of that subtotal's parent (e.g., `total_current_assets`). The hierarchy must be strictly nested.
    4.  **Semantic Summation and Non-Duplication**: When defining `subtotal_of` relationships, infer the most accurate accounting hierarchy based on the provided `INDUSTRY RESEARCH` and the actual line item values. Ensure that components are not double-counted; if a subtotal implicitly includes certain items (e.g., `operating_income` including `net_interest_income` for financial institutions), those implicitly included items should not be added again as separate components to a higher-level subtotal (e.g., `profit_before_tax`). Prioritize a hierarchy that reflects standard accounting principles and avoids material discrepancies.
    5.  **Contra-Accounts:** Contra-accounts like `loan_loss_reserve` or `treasury_stock` MUST have `sign: "negative"`.

    ## ROLE ASSIGNMENT LOGIC
    1.  **Domain Scoping:** When assigning a `schema_role` to an item in a statement, you **MUST ONLY** use a role from that statement's corresponding list in the `__metadata_definitions__` (e.g., use `income_statement_roles` for income statement items).
    2.  **Unique Role Assignment:** Each `schema_role` value must be unique across all statements. If an item appears in multiple statements (like `depreciation_amortization`), only assign the role in the primary statement (usually the income statement) and set it to `null` in other statements.
    3.  **Cash Flow Statement Roles:** For the cash flow statement, only use roles with the `CFS_` prefix or `null`. Never use `IS_` prefixed roles in the cash flow statement.
    4.  **Sparsity:** Most line items will have `schema_role: null`. Only assign roles if the item's function is unambiguous and it is present in the source data.

    ## CASH FLOW STATEMENT: Special Structural Requirements
    -   The Cash Flow Statement requires a complete structure for validation. You **MUST** generate schema entries for `cfs_net_profit`, `beginning_cash_balance`, and `ending_cash_balance`, even if they are not in the source data. Assign their roles as defined in the `cash_flow_statement_roles`.

    # UNIVERSAL VALIDATOR SCHEMA (Your Guide)
    This template defines the required structure, domain-scoped roles, and all valid `enum` values.
    ```json
    {json.dumps(master_template, indent=2)}
    ```

    # ISSUER'S HISTORICAL FINANCIAL DATA (Your Primary Input)
    Analyze this data to generate the schema.
    ```json
    {json.dumps(historical_data, indent=2)}
    ```

    # INDUSTRY RESEARCH (Additional Context)
    Use this data to understand the business model and correctly interpret `cfs_classification`.
    ```json
    {industry_research_section}
    ```

    # COUNTRY RESEARCH (Additional Context)
    Use this data to understand any local accounting conventions.
    ```json
    {country_research_section}
    ```

    # FINAL SELF-CRITIQUE CHECKLIST (CRITICAL - Mentally review your generated JSON against these questions):
    -   [ ] Have I strictly enforced the "Law of Absolute Totals" ensuring `total_assets` and `net_profit` have `subtotal_of: null`?
    -   [ ] Does my entire hierarchy strictly obey the "Rule of Levels" and the "Parental Exclusivity Rule"?
    -   [ ] Is the structure for `net_profit`, `loans_receivable_net`, and `ending_cash_balance` consistent with the "Formulaic Assertions"?
    -   [ ] Did I use ONLY domain-scoped roles and only assign them to items that actually exist in the source data for that statement?
    -   [ ] Is my Cash Flow Statement schema structurally complete with ALL mandatory articulation and cash balance items?
    -   [ ] Is the `__metadata_definitions__` section EXCLUDED from my final JSON output?

    # REQUIRED OUTPUT FORMAT
    Return your "Chain-of-Thought Analysis" in plain text, followed by the complete JSON object representing the generated validation schema, enclosed in markdown code fences.
    ```json
    {{
      "income_statement": {{ ... }},
      "balance_sheet": {{ ... }},
      "cash_flow_statement": {{ ... }}
    }}
    ```
    """
    
    # 6. Initialize LLM and make the API call
    logger.info("[6/7] Initializing LLM and invoking schema generation...")
    import config # Assuming a config file for API keys
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-04-17",
            temperature=0.0,
            top_p=getattr(config, "LLM_TOP_P", 1.0),
            google_api_key=getattr(config, "GOOGLE_API_KEY", None),
        )
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        logger.info("LLM API call completed successfully.")
    except Exception as e:
        logger.error(f"LLM API call failed: {str(e)}")
        raise
    
    # Robustly parse the JSON from the LLM response
    logger.info("Processing LLM response to extract JSON schema")
    response_content = response.content
    generated_schema = {}
    
    pattern = r'```(?:json)?\s*({[\s\S]*?})\s*```'
    matches = re.findall(pattern, response_content)
    
    if matches:
        try:
            generated_schema = json.loads(matches[0])
            logger.info("Successfully extracted JSON schema from LLM response markdown block.")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            raise ValueError(f"Could not extract valid JSON from LLM response: {e}")
    else:
        try:
            generated_schema = json.loads(response_content)
            logger.info("Successfully parsed entire LLM response as JSON schema.")
        except json.JSONDecodeError as e:
            logger.error(f"LLM response did not contain valid JSON or a markdown block: {e}")
            raise ValueError(f"Could not extract valid JSON from LLM response: {e}")

    # 7. Save the generated schema to the jit_schemas folder
    logger.info("[7/7] Saving generated JIT validation schema to file")
    
    # Create jit_schemas subdirectory if it doesn't exist
    jit_schemas_folder = credit_analysis_folder / "jit_schemas"
    jit_schemas_folder.mkdir(exist_ok=True, parents=True)
    
    # Set output path
    output_path = jit_schemas_folder / "validation_schema.json"
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(generated_schema, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved JIT validation schema to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save validation schema: {str(e)}")
        raise
    
    logger.info("=== generate_jit_validation_schema completed successfully ===")
    return generated_schema


@mcp_app.tool()
async def generate_jit_cfs_derivation_schema(security_id: str) -> dict:
    """
    Analyzes validated financial data and generates a bespoke JIT CFS derivation schema.

    1. Loads the issuer's validated financial data, including all metadata from the Validator.
    2. Loads the universal CFS derivation schema that defines the rules for formula generation.
    3. Uses an LLM to analyze the data's quality issues (plugs, orphans) and generate a bespoke
       derivation schema with explicit `calculation_template` formulas.
    4. Saves the generated schema for use by the CFS Derivator Engine.

    Args:
        security_id: The security ID for which to generate a derivation schema.

    Returns:
        A dictionary containing the generated CFS derivation schema.
    """
    logger.info("=== Starting generate_jit_cfs_derivation_schema ===")
    logger.info(f"Starting JIT derivation schema generation for security_id: {security_id}")

    # 1. Validate security_id and get paths
    logger.info(f"[1/7] Validating security_id: {security_id}")
    if not security_id:
        raise ValueError("security_id argument must be provided")

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from financial_analyst.security_folder_utils import find_security_folder, get_security_file

    security_folder = find_security_folder(security_id)
    if not security_folder:
        raise ValueError(f"Security folder not found for {security_id}")
    
    credit_analysis_folder = security_folder / "credit_analysis"

    # 2. Load the VALIDATOR'S OUTPUT, which is the primary input for this engine
    logger.info("[2/7] Loading issuer's validated financial data (Validator output)")
    validator_output_path = credit_analysis_folder / "validator_checkpoints" / "validator_final_results.json"
    if not validator_output_path.exists():
        error_msg = f"validator_final_results.json not found for {security_id} at {validator_output_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        with open(validator_output_path, "r", encoding="utf-8") as f:
            validator_output = json.load(f)
        logger.info("Successfully loaded validator output data.")
    except Exception as e:
        logger.error(f"Error loading validator output data: {str(e)}")
        raise

    # 3. Load industry and country research data (remains the same)
    logger.info("[3/7] Loading industry and country research data")
    industry_research_path = Path("/home/me/CascadeProjects/Alpy/subagents/financial_analyst/external_reports/industry_research.json")
    country_research_path = Path("/home/me/CascadeProjects/Alpy/subagents/financial_analyst/external_reports/country_research.json")
    
    industry_research = {}
    if industry_research_path.exists():
        with open(industry_research_path, "r", encoding="utf-8") as f:
            industry_research = json.load(f)
            
    country_research = {}
    if country_research_path.exists():
        with open(country_research_path, "r", encoding="utf-8") as f:
            country_research = json.load(f)

    # 4. Load the Universal Derivation Schema
    logger.info("[4/7] Loading the Universal Derivation Schema")
    template_path = Path(__file__).parent / "schemas" / "universal_derivation_schema.json"
    logger.info(f"Looking for derivation schema template at: {template_path}")

    if not template_path.exists():
        error_msg = f"universal_derivation_schema.json not found at {template_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            master_template = json.load(f)
        logger.info("Successfully loaded Universal Derivation Schema.")
    except Exception as e:
        logger.error(f"Error loading derivation schema template: {str(e)}")
        raise

    # 5. Prepare the LLM prompt for schema generation
    logger.info("[5/7] Preparing LLM prompt for derivation schema generation")

    industry_research_section = json.dumps(industry_research, indent=2) if industry_research else "No industry research data available."
    country_research_section = json.dumps(country_research, indent=2) if country_research else "No country research data available."
    
    prompt = f"""
    You are a Master Financial Analyst and Derivation Logic Architect. Your task is to analyze the output from a data validation engine and generate a bespoke, machine-executable JSON schema to derive a Cash Flow Statement (CFS). The schema you create must be a precise, prescriptive set of calculation instructions.

    PRIME DIRECTIVE: GUARANTEE OF RECONCILIATION
    The single most important objective is to generate a schema that produces a Cash Flow Statement where the net_change_in_cash perfectly reconciles with the change in the balance sheet's cash account. Every rule that follows is in service of this prime directive. If a perfect reconciliation cannot be achieved using the standard line items, you MUST create a final balancing line item as described in the "Balancing & Final Reconciliation" section below.

    # CORE TASK:
    Your output will have two parts:
    1.  A "Chain-of-Thought Analysis" section in plain text.
    2.  The final, complete JSON schema object.

    # CHAIN-OF-THOUGHT ANALYSIS (First, provide this section)
    Before generating the JSON, provide a brief analysis covering these points:
    -   **Overall Derivation Strategy:** Briefly state the plan. (e.g., "The reported CFS is unreliable. A full indirect method derivation is required. The strategy will include neutralizing balance sheet plugs and handling orphaned totals by deriving cash flow from the parent-level balance.")
    -   **Plug Handling:** If plugs are present in the validator output, explicitly state the formula you will generate for the `cf_reversal_of_accounting_plugs` line item.
    -   **Orphan Handling:** If `ORPHANED_TOTAL_FLAG` warnings exist, identify one and explain how you will create a new line item to derive cash flow from the total's balance instead of its broken components.
    -   **Key-Matching Confirmation:** Explicitly confirm that you have identified all keys needed for the capex and dividend formulas and that you will use their exact names as they appear in the source data without modification.
    
    # JSON SCHEMA GENERATION (Second, provide the complete JSON schema)

    ## CORE INSTRUCTIONS (NON-NEGOTIABLE):
    1.  **ABSOLUTE REQUIREMENT: USE VERBATIM KEYS.** The single most critical rule is that every line item key you reference in a `calculation_template` (e.g., `[balance_sheet.some_item]`) or generate in the output schema **MUST EXIST VERBATIM** in the `VALIDATOR_OUTPUT_JSON` input.
        -   You **MUST NOT** alter, rename, map, standardize, or "correct" any line item keys.
        -   **FAILURE EXAMPLE TO AVOID:** If the input data contains `balance_sheet.fixed_assets`, your template MUST use `[balance_sheet.fixed_assets]`. It is a CRITICAL FAILURE to change this to `[balance_sheet.fixed_assets_net]`.
    2.  **Generate Calculation Templates:** Your primary goal is to create a valid `calculation_template` for every component-level line item in the CFS.
    3.  **Adhere to the Master Template:** You MUST strictly adhere to the structure and format defined in the `__metadata_definitions__` section of the provided "UNIVERSAL CFS DERIVATION SCHEMA".
    4.  **Analyze Validator Output:** Base your logic on the `VALIDATOR_OUTPUT_JSON`, paying close attention to `warnings` and `summation_plugs` to handle data quality issues.

    BALANCING & FINAL RECONCILIATION (NON-NEGOTIABLE RULE)
    After defining the calculation templates for all known operational, investing, and financing activities, you MUST ensure the final statement balances. To do this, you will create one final component line item within net_cash_from_operations named cfs_unexplained_cash_movement.
    This item MUST ALWAYS be generated.
    Its calculation_template MUST be:
    "([balance_sheet.cash_and_equivalents] - [balance_sheet.cash_and_equivalents_prior]) - ([sum_of_all_other_cfs_components])"
    The [sum_of_all_other_cfs_components] placeholder represents the symbolic sum of every other component-level CFS line item you have defined. Your role is to translate this conceptual formula into a concrete template that the deterministic engine can execute. For example: ... - (cfs_net_profit + cfs_depreciation_amortization_add_back + cf_from_change_in_other_assets + ...)

    ## DERIVATION LOGIC (These are non-negotiable rules for formulas)
    1.  **Balance Sheet Deltas:**
        -   For Assets: Cash flow is `[prior_period_value] - [current_period_value]`. Formula: `([balance_sheet.asset_item_prior] - [balance_sheet.asset_item])`.
        -   For Liabilities & Equity: Cash flow is `[current_period_value] - [prior_period_value]`. Formula: `([balance_sheet.liability_item] - [balance_sheet.liability_item_prior])`.
    2.  **Standard Formulas (Use these exact templates):**
        -   `cfs_net_profit`: `"[income_statement.net_profit]"`
        -   `cfs_depreciation_amortization_add_back`: `"-1 * [income_statement.is_depreciation_amortization]"`
        -   `cfs_derived_capex`: `"-1 * ([balance_sheet.fixed_assets_net] - [balance_sheet.fixed_assets_net_prior] - [income_statement.is_depreciation_amortization])"`
        -   `cfs_dividends_paid`: `"-1 * ([balance_sheet.retained_earnings_prior] + [income_statement.net_profit] - [balance_sheet.retained_earnings])"`
    3.  **Plug Neutralization (If `summation_plugs` or `__accounting_equation__` plugs exist):**
        -   You MUST create a single component line item named `cf_reversal_of_accounting_plugs`.
        -   Its formula MUST sum the deltas of ALL plugs. Example: `"-1 * (([balance_sheet.summation_plugs.total_assets] - [balance_sheet.summation_plugs.total_assets_prior]) + ([balance_sheet.__accounting_equation__] - [balance_sheet.__accounting_equation___prior]))"`
        -   This item should roll up into `net_cash_from_operations`.
    4.  **Orphaned Total Handling (If `ORPHANED_TOTAL_FLAG` warnings exist):**
        -   Identify the orphaned total (e.g., `loans_receivable_net`).
        -   DO NOT create derivation templates for its individual components (e.g., `loans_receivable_gross`).
        -   Instead, create ONE new component (e.g., `cf_from_change_in_orphaned_loans_receivable_net`).
        -   The `calculation_template` for this new item MUST derive from the total's balance (e.g., `"[balance_sheet.loans_receivable_net_prior] - [balance_sheet.loans_receivable_net]"`).
        -   Assign its `subtotal_of` based on the most likely CFS section (e.g., `net_cash_from_operations` for a lending asset).

    # UNIVERSAL CFS DERIVATION SCHEMA (Your Guide)
    This template defines the required structure and format for your output.
    ```json
    {json.dumps(master_template, indent=2)}
    ```

    # VALIDATOR_OUTPUT_JSON (Your Primary Input)
    Analyze this data, especially `warnings` and `summation_plugs`, to generate the schema.
    ```json
    {json.dumps(validator_output, indent=2)}
    ```

    # INDUSTRY/COUNTRY RESEARCH (Additional Context)
    Industry: {industry_research_section}
    Country: {country_research_section}

    # FINAL SELF-CRITIQUE CHECKLIST (CRITICAL - Mentally review your generated JSON):
    -   [ ] **KEY VERIFICATION:** Have I double-checked that every single key referenced in my `calculation_template` strings (e.g., `[statement.key]`) corresponds to an *exact, character-for-character* key in the input `VALIDATOR_OUTPUT_JSON`?
    -   [ ] Does every component-level item have a `calculation_template` string?
    -   [ ] Are the formulas for asset deltas and liability/equity deltas mathematically correct (i.e., signs inverted)?
    -   [ ] If plugs were present in the input, have I created a single `cf_reversal_of_accounting_plugs` item with the correct formula?
    -   [ ] If orphans were present, have I correctly implemented the "fallback to total" logic?
    -   [ ] Is the `__metadata_definitions__` section EXCLUDED from my final JSON output?

    # REQUIRED OUTPUT FORMAT
    Return your "Chain-of-Thought Analysis" in plain text, followed by the complete JSON object representing the generated derivation schema, enclosed in markdown code fences.
    ```json
    {{
      "cash_flow_statement": {{ ... }}
    }}
    ```
    """

    # 6. Initialize LLM and make the API call
    logger.info("[6/7] Initializing LLM and invoking derivation schema generation...")
    import config

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-04-17",
            temperature=0.0,
            top_p=getattr(config, "LLM_TOP_P", 1.0),
            google_api_key=getattr(config, "GOOGLE_API_KEY", None),
        )
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        logger.info("LLM API call completed successfully.")
    except Exception as e:
        logger.error(f"LLM API call failed: {str(e)}")
        raise

    # 7. Process response and save the generated schema
    logger.info("[7/7] Processing LLM response and saving generated JIT derivation schema")
    response_content = response.content
    generated_schema = {}
    
    pattern = r'```(?:json)?\s*({[\s\S]*?})\s*```'
    matches = re.findall(pattern, response_content)
    
    if matches:
        try:
            generated_schema = json.loads(matches[0])
            logger.info("Successfully extracted JSON schema from LLM response markdown block.")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            raise ValueError(f"Could not extract valid JSON from LLM response: {e}")
    else:
        logger.error(f"LLM response did not contain a valid JSON markdown block.")
        raise ValueError("Could not extract valid JSON from LLM response.")

    jit_schemas_folder = credit_analysis_folder / "jit_schemas"
    jit_schemas_folder.mkdir(exist_ok=True, parents=True)
    output_path = jit_schemas_folder / "cfs_derivation_schema.json"
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(generated_schema, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved JIT CFS derivation schema to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save derivation schema: {str(e)}")
        raise

    logger.info("=== generate_jit_cfs_derivation_schema completed successfully ===")
    return generated_schema


@mcp_app.tool()
async def generate_jit_projections_schema(security_id: str) -> dict:
    """
    Analyzes final derived financials and generates a bespoke JIT Projections Schema.

    1. Loads the issuer's final derived financial data (output of the CFS Derivator).
    2. Loads the universal projections schema that defines the rules and roles for forecasting.
    3. Uses an LLM to analyze the historical data, identify the business model, and generate
       a bespoke projections schema with formulas for all 3 statements.
    4. Saves the generated schema for use by the Projections Engine.

    Args:
        security_id: The security ID for which to generate a projections schema.

    Returns:
        A dictionary containing the generated projections schema.
    """
    logger.info("=== Starting generate_jit_projections_schema ===")
    logger.info(f"Starting JIT projections schema generation for security_id: {security_id}")

    # 1. Validate security_id and get paths (Pattern replicated)
    logger.info(f"[1/7] Validating security_id: {security_id}")
    if not security_id:
        raise ValueError("security_id argument must be provided")

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from financial_analyst.security_folder_utils import find_security_folder, get_security_file

    security_folder = find_security_folder(security_id)
    if not security_folder:
        raise ValueError(f"Security folder not found for {security_id}")
    
    credit_analysis_folder = security_folder / "credit_analysis"

    # 2. Load the DERIVED FINANCIALS, which is the primary input for this engine
    logger.info("[2/7] Loading issuer's final derived financial data")
    derived_data_path = credit_analysis_folder / "final_derived_financials.json"
    if not derived_data_path.exists():
        error_msg = f"final_derived_financials.json not found for {security_id} at {derived_data_path}. Run CFS Derivation first."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        with open(derived_data_path, "r", encoding="utf-8") as f:
            full_derived_data = json.load(f)
            # Extract historical data and metadata into separate variables
            derived_financials_data = full_derived_data.get('transformed_data', {})
            derived_metadata = full_derived_data.get('metadata', {})
            
            # Extract annual backbone periods from metadata if available
            historical_data = []
            if 'time_series_map' in derived_metadata and 'annual_backbone' in derived_metadata['time_series_map']:
                backbone_periods = derived_metadata['time_series_map']['annual_backbone']
                if 'mapped_historical_data' in derived_financials_data:
                    historical_data = [
                        derived_financials_data['mapped_historical_data'][period['data_index']] 
                        for period in backbone_periods
                        if period['data_index'] < len(derived_financials_data['mapped_historical_data'])
                    ]
            
            # Add historical data to the main data structure
            derived_financials_data['historical_data'] = historical_data
        logger.info("Successfully loaded and filtered final derived financial data.")
    except Exception as e:
        logger.error(f"Error loading final derived financial data: {str(e)}")
        raise

    # 3. Load industry and country research data (Pattern replicated)
    logger.info("[3/7] Loading industry and country research data")
    industry_research_path = Path("/home/me/CascadeProjects/Alpy/subagents/financial_analyst/external_reports/industry_research.json")
    country_research_path = Path("/home/me/CascadeProjects/Alpy/subagents/financial_analyst/external_reports/country_research.json")
    
    industry_research = {}
    if industry_research_path.exists():
        with open(industry_research_path, "r", encoding="utf-8") as f: industry_research = json.load(f)
            
    country_research = {}
    if country_research_path.exists():
        with open(country_research_path, "r", encoding="utf-8") as f: country_research = json.load(f)

    # 4. Load the Universal Projections Schema (Pattern replicated)
    logger.info("[4/7] Loading the Universal Projections Schema")
    template_path = Path(__file__).parent / "schemas" / "universal_projections_schema.json"
    logger.info(f"Looking for projections schema template at: {template_path}")

    if not template_path.exists():
        error_msg = f"universal_projections_schema.json not found at {template_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            master_projections_template = json.load(f)
        logger.info("Successfully loaded Universal Projections Schema.")
    except Exception as e:
        logger.error(f"Error loading projections schema template: {str(e)}")
        raise

    # 5. Load issuer business profile
    logger.info("[5/7] Loading issuer business profile")
    issuer_business_profile = {}
    try:
        issuer_profile_path = get_security_file(security_id, "data_extraction/issuer_business_profile.json")
        with open(issuer_profile_path, "r", encoding="utf-8") as f:
            issuer_business_profile = json.load(f)
        logger.info("Successfully loaded issuer business profile.")
    except Exception as e:
        logger.warning(f"Error loading issuer business profile: {str(e)}")

    # 6. Prepare the LLM prompt for projections schema generation
    logger.info("[6/7] Preparing LLM prompt for projections schema generation")

    industry_research_section = json.dumps(industry_research, indent=2) if industry_research else "No industry research data available."
    country_research_section = json.dumps(country_research, indent=2) if country_research else "No country research data available."
    issuer_profile_section = json.dumps(issuer_business_profile, indent=2, ensure_ascii=False) if issuer_business_profile else "No issuer business profile data available."
    
    prompt = f"""
    ### SECTION I: ROLE AND OBJECTIVE DEFINITION

    You are an expert financial modeler and a meticulous data-mapping specialist. Your single, non-negotiable task is to correctly populate the provided master_template.json with the financial data and context supplied. Your output must be a syntactically perfect and financially logical JSON object. All explanations must be confined to the justification field within each driver.

    ### SECTION II: STRUCTURED CONTEXTUAL DATA

    <HISTORICAL_DATA>
    {historical_data}
    </HISTORICAL_DATA>

    <METADATA>
    {derived_metadata}
    </METADATA>

    <BUSINESS_PROFILE>
    {issuer_profile_section}
    </BUSINESS_PROFILE>

    <MACRO_CONTEXT>
    "industry_research": {industry_research_section};
    "country_research": {country_research_section}
    </MACRO_CONTEXT>

    <SCHEMA_CONTRACT>
    {master_projections_template}
    </SCHEMA_CONTRACT>

    SECTION III: THE ALGORITHMIC MANDATE
    MASTER DIRECTIVE: PRECISION IN POPULATION
    Your primary directive is to populate the <SCHEMA_CONTRACT> with absolute precision. Every decision—from mapping a historical account to selecting a projection model—must be a direct and logical application of the rules below.
    Rule 0 (The Supreme Law of Precedence & Conflict Resolution): The rules are organized into a strict, non-negotiable hierarchy. A rule from a higher pillar always overrides a rule from a lower pillar. Within a pillar, a rule with a lower number takes absolute precedence over a rule with a higher number.
    Pillar A (Mapping & Structural Integrity) > Pillar B (Model Selection & Causality) > Pillar C (Driver Population & Analytics)
    A. Pillar I: Mapping & Structural Integrity
    This pillar governs the correct mapping of historical data onto the Master Template.
    Rule A.1 (The Law of Exhaustive Mapping): You MUST attempt to map every financial line item from <HISTORICAL_DATA> to its single most logical slot in the <SCHEMA_CONTRACT>. The target for the mapping is the historical_account_key property.
    Rule A.2 (The Law of Null Mapping): If a line item in the Master Template has no corresponding data in the historicals (e.g., inventory for a bank), its historical_account_key MUST be null. You are forbidden from mapping an unrelated historical account to fill a blank slot.
    Rule A.3 (The Law of Granularity for Lists): For sections in the Master Template that can accept multiple items (e.g., scheduled_long_term_debt), you are mandated to create a distinct entry for each corresponding debt instrument found in the historical data. You are strictly forbidden from pre-aggregating these items yourself.
    B. Pillar II: Model Selection & Causality
    This pillar governs the logical heart of the model: selecting the correct projection methodology.
    Rule B.1 (The Law of Model Selection): For every projectable line item in the Master Template, you MUST select the most appropriate projection_model from the list of available options defined in the <SCHEMA_CONTRACT>. This selection must be directly justified by the company's profile in <BUSINESS_PROFILE>.
    * Example: If <BUSINESS_PROFILE> states "The company is a commercial bank," you are mandated to select BS_Driven for the primary revenue line.
    Rule B.2 (The Law of The Inviolate Plug): The revolver in the debt section MUST have its projection_model set to Balancing_Plug_Debt. The cash item MUST be set to Derived_From_CFS. These are non-negotiable and override all other considerations.
    Rule B.3 (The Law of The Derived Cash Flow Statement): All line items within the cash_flow_statement section of the template are derived by the engine. You are forbidden from assigning any projection_model or drivers to them.
    C. Pillar III: Driver Population & Analytics
    This pillar governs the quality and auditability of the financial assumptions.
    Rule C.1 (The Law of Required Drivers): Once a projection_model is selected for a line item, you MUST populate its drivers object with all the required driver keys specified for that model in the <SCHEMA_CONTRACT>.
    Rule C.2 (The Law of Auditable Justification): The justification field for every single driver MUST contain a concise, logical reason for the chosen baseline value. This justification must explicitly reference <HISTORICAL_DATA> or <BUSINESS_PROFILE>.
    * Correct Example: "justification": "Baseline set to the T-1 historical calculated Gross Margin of 45.2% as per <HISTORICAL_DATA>."
    * Forbidden Example: "justification": "This is a reasonable assumption."
    SECTION IV: DEBUGGING AND REASONING LOG
    You WILL add a top-level key to the final JSON called __reasoning_log__. The value of this key will be an object containing your analysis of the modeling task, addressing the following points concisely:
    Difficult Mappings: "Which historical accounts were the most difficult to map to a slot in the Master Template, and why?"
    Key Model Selection: "What was the single most important projection_model selection you made (e.g., choosing BS_Driven for revenue), and what specific text in the <BUSINESS_PROFILE> justified it?"
    Most Significant Driver: "What is the single most significant driver baseline value you generated, and how confident are you in its calculation?"
    Confidence Score: "Provide a 'Confidence Score' from 0.0 to 1.0 that the populated schema is a financially logical representation of the issuer, and briefly justify it."
    SECTION V: FINAL OUTPUT INSTRUCTION
    Your final and only output MUST be a single JSON markdown block containing the fully populated Master Template. The output must be complete and syntactically perfect.
    """

    # 7. Initialize LLM and make the API call (Pattern replicated)
    logger.info("[7/7] Initializing LLM and invoking projections schema generation...")
    import config

    try:
        # NOTE: Model and parameters might be adjusted for this more complex task
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-04-17",
            temperature=0.0,
            top_p=getattr(config, "LLM_TOP_P", 1.0),
            google_api_key=getattr(config, "GOOGLE_API_KEY", None),
        )
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        logger.info("LLM API call completed successfully.")
    except Exception as e:
        logger.error(f"LLM API call failed: {str(e)}")
        raise

    # 8. Process response and save the generated schema (Pattern replicated)
    logger.info("[8/8] Processing LLM response and saving generated JIT projections schema")
    response_content = response.content
    generated_schema = {}
    
    pattern = r'```(?:json)?\s*({[\s\S]*?})\s*```'
    matches = re.findall(pattern, response_content)
    
    if matches:
        try:
            generated_schema = json.loads(matches[0])
            logger.info("Successfully extracted JSON schema from LLM response markdown block.")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            raise ValueError(f"Could not extract valid JSON from LLM response: {e}")
    else:
        logger.error(f"LLM response did not contain a valid JSON markdown block.")
        raise ValueError("Could not extract valid JSON from LLM response.")

    jit_schemas_folder = credit_analysis_folder / "jit_schemas"
    jit_schemas_folder.mkdir(exist_ok=True, parents=True)
    output_path = jit_schemas_folder / "projections_schema.json" # New output filename
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(generated_schema, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved JIT projections schema to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save projections schema: {str(e)}")
        raise

    logger.info("=== generate_jit_projections_schema completed successfully ===")
    return generated_schema


@mcp_app.tool()
async def generate_projection_drivers(security_id: str) -> dict:
    """
    Analyzes final derived financials and the projection schema to generate a
    complete set of bespoke, time-variant drivers for the projection model.

    1.  Loads the issuer's final derived financial data.
    2.  Loads the issuer's bespoke JIT projections schema to discover required drivers.
    3.  Loads the universal driver schema that defines the output format.
    4.  Uses an LLM to analyze all available context and generate baseline values
        and trends for every required driver.
    5.  Saves the generated drivers for use by the Projections Engine.

    Args:
        security_id: The security ID for which to generate projection drivers.

    Returns:
        A dictionary containing the generated projection drivers.
    """
    logger.info("=== Starting generate_projection_drivers ===")
    logger.info(f"Starting JIT projection driver generation for security_id: {security_id}")

    # 1. Validate security_id and get paths (Pattern replicated)
    logger.info(f"[1/8] Validating security_id: {security_id}")
    if not security_id:
        raise ValueError("security_id argument must be provided")

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from financial_analyst.security_folder_utils import find_security_folder, get_security_file

    security_folder = find_security_folder(security_id)
    if not security_folder:
        raise ValueError(f"Security folder not found for {security_id}")
    
    credit_analysis_folder = security_folder / "credit_analysis"

    # 2. Load the DERIVED FINANCIALS as the historical fact base
    logger.info("[2/8] Loading issuer's final derived financial data")
    derived_data_path = credit_analysis_folder / "final_derived_financials.json"
    if not derived_data_path.exists():
        error_msg = f"final_derived_financials.json not found for {security_id} at {derived_data_path}."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        with open(derived_data_path, "r", encoding="utf-8") as f:
            derived_financials_data = json.load(f)
        logger.info("Successfully loaded final derived financial data.")
    except Exception as e:
        logger.error(f"Error loading final derived financial data: {str(e)}")
        raise

    # 3. Load the PROJECTIONS SCHEMA to understand the model's logic
    logger.info("[3/8] Loading issuer's JIT projections schema")
    projections_schema_path = credit_analysis_folder / "jit_schemas" / "projections_schema.json"
    if not projections_schema_path.exists():
        error_msg = f"projections_schema.json not found for {security_id} at {projections_schema_path}."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        with open(projections_schema_path, "r", encoding="utf-8") as f:
            projections_schema = json.load(f)
        logger.info("Successfully loaded JIT projections schema.")
    except Exception as e:
        logger.error(f"Error loading JIT projections schema: {str(e)}")
        raise

    # 4. Discover all required drivers from the projections schema (Deterministic)
    logger.info("[4/8] Discovering required drivers from projections schema")
    schema_as_string = json.dumps(projections_schema)
    driver_pattern = r'\[driver:(\w+)\]'
    required_drivers_list = sorted(list(set(re.findall(driver_pattern, schema_as_string))))
    logger.info(f"Discovered {len(required_drivers_list)} required drivers: {required_drivers_list}")

    # 5. Load the Universal Driver Schema (Pattern replicated)
    logger.info("[5/8] Loading the Universal Driver Schema")
    template_path = Path(__file__).parent / "schemas" / "universal_driver_schema.json"
    if not template_path.exists():
        error_msg = f"universal_driver_schema.json not found at {template_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            master_driver_template = json.load(f)
        logger.info("Successfully loaded Universal Driver Schema.")
    except Exception as e:
        logger.error(f"Error loading driver schema template: {str(e)}")
        raise

    # 6. Load industry and country research data and issuer business profile (Pattern replicated)
    logger.info("[6/8] Loading industry and country research data and issuer profile")
    industry_research_path = Path("/home/me/CascadeProjects/Alpy/subagents/financial_analyst/external_reports/industry_research.json")
    country_research_path = Path("/home/me/CascadeProjects/Alpy/subagents/financial_analyst/external_reports/country_research.json")
    
    industry_research = {}
    if industry_research_path.exists():
        with open(industry_research_path, "r", encoding="utf-8") as f: industry_research = json.load(f)
            
    country_research = {}
    if country_research_path.exists():
        with open(country_research_path, "r", encoding="utf-8") as f: country_research = json.load(f)

    issuer_business_profile = {}
    try:
        issuer_profile_path = get_security_file(security_id, "data_extraction/issuer_business_profile.json")
        with open(issuer_profile_path, "r", encoding="utf-8") as f:
            issuer_business_profile = json.load(f)
        logger.info("Successfully loaded issuer business profile.")
    except Exception as e:
        logger.warning(f"Error loading issuer business profile: {str(e)}")

    # 7. Prepare the LLM prompt for projections schema generation
    logger.info("[7/8] Preparing LLM prompt for projections schema generation")

    industry_research_section = json.dumps(industry_research, indent=2) if industry_research else "No industry research data available."
    country_research_section = json.dumps(country_research, indent=2) if country_research else "No country research data available."
    issuer_profile_section = json.dumps(issuer_business_profile, indent=2, ensure_ascii=False) if issuer_business_profile else "No issuer business profile data available."


    # 8. Prepare the LLM prompt for driver generation
    logger.info("[8/8] Preparing LLM prompt for projection driver generation")

    # Note: Industry/Country research is omitted here to keep the context focused, but can be added back if needed.
    
    prompt = f"""
    You are a Senior Investment Analyst and Economic Forecaster. Your task is to generate a complete set of forward-looking assumptions (drivers) for a financial projection model. You must analyze all available historical data to inform your baseline and trends.

    # PRIME DIRECTIVE:
    Your goal is to generate a `drivers.json` object. For every single driver in the `REQUIRED_DRIVERS_LIST`, you must create an entry containing a `description`, a `baseline` value for Year 1, a `justification` for your choices, and trend values for the `short_term` (Years 2-3), `medium_term` (Years 4-9), and `terminal` (Years 10+).

    # CORE TASK:
    Your output will have two parts:
    1.  A "Chain-of-Thought Analysis" section in plain text.
    2.  The final, complete JSON object of all drivers.

    # CHAIN-OF-THOUGHT ANALYSIS (First, provide this section)
    -   **Key Driver Analysis:** Pick one key driver (e.g., `loans_receivable_growth_rate`). Explain your reasoning for the chosen `baseline` by referencing specific historical data points (annual or interim). Justify your `trends` by citing economic principles (e.g., maturation, competition, regression to the mean).

    # JSON DRIVER GENERATION (Second, provide the complete JSON object)

    ## CORE INSTRUCTIONS (NON-NEGOTIABLE):
    1.  **COMPLETE COVERAGE:** You MUST generate an entry for EVERY driver listed in the `REQUIRED_DRIVERS_LIST`. Do not omit any.
    2.  **DATA-DRIVEN BASELINE:** Your `baseline` value for each driver must be logically derived from the provided `DERIVED_FINANCIALS_JSON`. Analyze the historical data, including interim periods, to form a credible starting point for Year 1.
    3.  **LOGICAL TRENDS:** Your `trends` should tell a coherent story. Typically, high-growth drivers should fade towards a more conservative terminal value. Stable ratios or policies can be held constant.
    4.  **MANDATORY JUSTIFICATION:** The `justification` field is not optional. For each driver, you must provide a concise rationale for your chosen baseline and trend values. This is critical for auditability.
    5.  **STRICT SCHEMA COMPLIANCE:** Your final JSON output MUST be a dictionary where keys are the driver names and values are objects that perfectly conform to the structure defined in the `UNIVERSAL_DRIVER_SCHEMA`.

    # UNIVERSAL DRIVER SCHEMA (Your Guide)
    This template defines the required structure and format for your output.
    ```json
    {json.dumps(master_driver_template, indent=2)}
    ```

    # DERIVED_FINANCIALS_JSON (Your Primary Input for Historical Analysis)
    Analyze this complete, balanced historical data to generate the driver values.
    ```json
    {json.dumps(derived_financials_data, indent=2)}
    ```

    # PROJECTIONS_SCHEMA_JSON (Your Guide for Context)
    Use this schema to understand how each driver will be used in the model's formulas.
    ```json
    {json.dumps(projections_schema, indent=2)}
    ```
    
    # REQUIRED_DRIVERS_LIST (Your Checklist)
    You must provide a complete entry for every driver in this list.
    {str(required_drivers_list)}

    # INDUSTRY/COUNTRY RESEARCH (Additional Context)
    Industry: {industry_research_section}
    Country: {country_research_section}
    Issuer: {issuer_profile_section}

    # FINAL SELF-CRITIQUE CHECKLIST (CRITICAL):
    -   [ ] **Completeness:** Have I created a valid entry for EVERY driver in the `REQUIRED_DRIVERS_LIST`?
    -   [ ] **Schema Compliance:** Does the structure of my entire output perfectly match the `UNIVERSAL_DRIVER_SCHEMA` format?
    -   [ ] **Justification:** Does every single driver have a logical, non-generic `justification`?
    -   [ ] **Data Grounding:** Is my `baseline` for each driver plausibly linked to the historical data provided?

    # REQUIRED OUTPUT FORMAT
    Return your "Chain-of-Thought Analysis" in plain text, followed by the complete JSON object representing all generated drivers, enclosed in markdown code fences.
    ```json
    {{
      "driver_1": {{...}},
      "driver_2": {{...}}
    }}
    ```
    """

    # 8. Initialize LLM and make the API call (Pattern replicated)
    logger.info("[8/8] Initializing LLM and invoking projection driver generation...")
    import config

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-04-17",
            temperature=0.0,
            top_p=getattr(config, "LLM_TOP_P", 1.0),
            google_api_key=getattr(config, "GOOGLE_API_KEY", None),
        )
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        logger.info("LLM API call completed successfully.")
    except Exception as e:
        logger.error(f"LLM API call failed: {str(e)}")
        raise

    # 9. Process response and save the generated drivers (Pattern replicated)
    logger.info("[9/9] Processing LLM response and saving generated JIT projection drivers")
    response_content = response.content
    generated_drivers = {}
    
    pattern = r'```(?:json)?\s*({[\s\S]*?})\s*```'
    matches = re.findall(pattern, response_content)
    
    if matches:
        try:
            generated_drivers = json.loads(matches[0])
            logger.info("Successfully extracted JSON drivers from LLM response markdown block.")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            raise ValueError(f"Could not extract valid JSON from LLM response: {e}")
    else:
        logger.error(f"LLM response did not contain a valid JSON markdown block.")
        raise ValueError("Could not extract valid JSON from LLM response.")

    # Save to a new dedicated folder for assumptions
    jit_assumptions_folder = credit_analysis_folder / "projections"
    jit_assumptions_folder.mkdir(exist_ok=True, parents=True)
    output_path = jit_assumptions_folder / "baseline_projection_drivers.json"
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(generated_drivers, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved baseline projection drivers to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save projection drivers: {str(e)}")
        raise

    logger.info("=== generate_projection_drivers completed successfully ===")
    return generated_drivers




@mcp_app.tool()
async def validate_financial_statements(security_id: str) -> dict:
    """
    Deterministically validate and reconcile financial statements from mapped historical data.
    
    This tool performs the following:
    1. Loads the mapped historical data from the credit_analysis folder
    2. Applies deterministic validation to ensure three-statement linkage
    3. Skips periods with insufficient data
    4. Derives cash flow statements from balance sheet changes and income statements
    5. Uses plug variables to reconcile any inconsistencies between statements
    6. Saves the validated data as validated_historical_data.json
    """
    logger.info(f"Validating financial statements for security_id: {security_id}")
    
    try:
        # Get security folder
        security_folder = require_security_folder(security_id)
        if not security_folder:
            return {"error": f"Security folder not found for {security_id}"}
        
        # Find the security folder
        security_folder = require_security_folder(security_id)
        
        # Create the validator checkpoints directory
        checkpoints_dir = security_folder / "credit_analysis" / "validator_checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the mapped historical data
        # First try in data_extraction folder
        mapped_data_path = security_folder / "data_extraction" / "mapped_historical_data.json"
        
        # If not found there, try in credit_analysis folder
        if not mapped_data_path.exists():
            mapped_data_path = security_folder / "credit_analysis" / "mapped_historical_data.json"
        
        if not mapped_data_path.exists():
            return {"error": f"Mapped historical data not found in data_extraction or credit_analysis folders"}
        
        with open(mapped_data_path, "r", encoding="utf-8") as f:
            mapped_data = json.load(f)
            
        logger.info(f"Loaded mapped historical data from {mapped_data_path}")
        
        # Add more detailed error handling
        try:
            # Create credit_analysis_folder path
            credit_analysis_folder = security_folder / "credit_analysis"
            # Create checkpoints directory if it doesn't exist
            checkpoints_dir = credit_analysis_folder / "validator_checkpoints"
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            
            # Validate the financial statements
            # The validator will handle its own checkpoint system internally
            validated_data = validate_historical_data(
                mapped_data, 
                save_checkpoints=True,  # Enable checkpoints in validator
                filter_sensible_data=True,  # Filter periods with sensible data
                base_dir=str(checkpoints_dir),  # Pass checkpoint directory
                derive_cash_flow=True,  # Derive cash flow after filtering
                normalize_to_annual=False,  # Use semi-annual data directly instead of annualizing
                validate_inter_period=True  # Perform inter-period consistency validation
            )
            
            # The filtered data with cash flow is now in validated_data
            filtered_data = validated_data
            
            # Save filtered validated data (main output) before reclassification
            validated_output_path = credit_analysis_folder / "validated_historical_data.json"
            with open(validated_output_path, "w", encoding="utf-8") as f:
                json.dump(filtered_data, f, indent=2)
                
            logger.info(f"Saved validated historical data to {validated_output_path}")
            
            # Reclassify financial plugs for the entire dataset
            logger.info("Starting financial plug reclassification for all periods...")
            from financial_statement_validator import reclassify_financial_plugs
            
            # Create a deep copy of the filtered data for reclassification
            reclass_data = {
                "mapped_historical_data": copy.deepcopy(filtered_data["mapped_historical_data"]),
                "metadata": filtered_data.get("metadata", {})
            }
            
            # Perform the reclassification on the entire dataset
            reclassified_data = reclassify_financial_plugs(reclass_data)
            
            if not reclassified_data or not reclassified_data.get("mapped_historical_data"):
                logger.warning("Reclassification did not return valid data")
                reclassified_data = copy.deepcopy(filtered_data)
            
            # Save reclassified data to the checkpoints directory following the standard pattern
            checkpoint_path = os.path.join(str(checkpoints_dir), "validator_state_reclassified_data.json")
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(reclassified_data, f, indent=2)
            logger.info(f"Saved reclassified data to {checkpoint_path}")
            
            # Update the output path to point to the checkpoint location
            reclassified_output_path = Path(checkpoint_path)
            
            # Return both filtered and reclassified data
            return {
                "filtered_data": filtered_data,
                "reclassified_data": reclassified_data,
                "metadata": {
                    "complete_periods_count": len(validated_data["mapped_historical_data"]),
                    "filtered_periods_count": len(filtered_data["mapped_historical_data"]),
                    "filtered_periods_removed": len(validated_data["mapped_historical_data"]) - len(filtered_data["mapped_historical_data"]),
                    "reclassification_applied": True,
                    "reclassified_data_path": str(reclassified_output_path)
                }
            }
            
        except TypeError as te:
            # Handle specific TypeError (None operations)
            logger.error(f"TypeError in validation: {te}")
            return {"error": f"Data validation error: {str(te)}", 
                    "validation_metadata": {
                        "status": "failed",
                        "reason": f"Type error during validation: {str(te)}"
                    }}
        except Exception as inner_e:
            logger.error(f"Validation error: {inner_e}")
            return {"error": f"Validation processing error: {str(inner_e)}"}
    
    except Exception as e:
        logger.error(f"Error in validate_financial_statements: {e}")
        logger.error(traceback.format_exc())
        return {"error": f"Failed to validate financial statements: {str(e)}"}


@mcp_app.tool()
async def generate_baseline_projection_drivers(security_id: str) -> dict:
    """
    Generate baseline projection drivers for 3-year financial projections using historical data,
    industry research, and country research.
    
    This tool performs the following:
    1. Loads the cleaned historical financial data and determines the industry.
    2. Loads the correct industry-specific financial schema.
    3. Dynamically generates a tailored `drivers_schema` from the industry schema.
    4. Loads industry and country research data.
    5. Uses an LLM to generate baseline projection drivers based on the generated schema.
    6. Saves the baseline drivers to a new projections subfolder.
    
    Args:
        security_id: The security ID to generate baseline drivers for.
        
    Returns:
        Dictionary containing the generated baseline drivers.
    """
    try:
        # Validate security_id and get paths
        if not security_id:
            raise ValueError("security_id argument must be provided")
            
        logger.info(f"Generating baseline projection drivers for security_id={security_id}")
        
        # Import from the correct relative path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from financial_analyst.security_folder_utils import find_security_folder, get_subfolder
        
        # Find the security folder (UNCHANGED)
        security_folder = find_security_folder(security_id)
        if not security_folder:
            raise ValueError(f"Security folder not found for {security_id}")
        
        # Get paths to required files (UNCHANGED)
        credit_analysis_folder = security_folder / "credit_analysis"
        
        # Create projections folder if it doesn't exist (UNCHANGED)
        projections_folder = credit_analysis_folder / "projections"
        projections_folder.mkdir(exist_ok=True)
        
        # Load industry and country research data (UNCHANGED)
        industry_research_path = Path("/home/me/CascadeProjects/Alpy/subagents/financial_analyst/external_reports/industry_research.json")
        country_research_path = Path("/home/me/CascadeProjects/Alpy/subagents/financial_analyst/external_reports/country_research.json")
        
        industry_research = {}
        if industry_research_path.exists():
            with open(industry_research_path, "r", encoding="utf-8") as f:
                industry_research = json.load(f)
                
        country_research = {}
        if country_research_path.exists():
            with open(country_research_path, "r", encoding="utf-8") as f:
                country_research = json.load(f)
        
        # Load issuer business profile using security folder helper (UNCHANGED)
        issuer_profile_path = security_folder / "data_extraction" / "issuer_business_profile.json"
        issuer_business_profile = {}
        if issuer_profile_path.exists():
            with open(issuer_profile_path, "r", encoding="utf-8") as f:
                issuer_business_profile = json.load(f)
        
        # --- NEW LOGIC: DYNAMIC SCHEMA GENERATION ---
        
        # 1. Load bond metadata to determine the industry schema type
        from financial_analyst.security_folder_utils import get_security_file
        
        try:
            bond_metadata_path = get_security_file(security_id, "data_extraction/bond_metadata.json")
            with open(bond_metadata_path, "r", encoding="utf-8") as f:
                bond_metadata = json.load(f)
            
            # Access industry_type from the nested bond_metadata object
            industry_schema_type = bond_metadata.get('bond_metadata', {}).get('industry_type')
            if not industry_schema_type:
                raise ValueError("industry_type field not found in bond_metadata.json under 'bond_metadata' object")
                
        except Exception as e:
            raise ValueError(f"Could not determine industry schema type from bond_metadata.json: {str(e)}")
            
        # 2. Load the correct industry-specific schema
        industry_schema = _load_industry_schema(industry_schema_type)
        
        # 3. Dynamically generate the driver schema from the industry schema
        projection_drivers_schema = _generate_drivers_schema_from_industry_schema(industry_schema, industry_schema_type)
        logger.info(f"Dynamically generated drivers schema for industry: {industry_schema_type}")

        # Optional but recommended: Save the generated schema for auditing/debugging
        generated_schema_path = projections_folder / "generated_drivers_schema.json"
        with open(generated_schema_path, "w", encoding="utf-8") as f:
            json.dump(projection_drivers_schema, f, indent=2)

        # --- END OF NEW LOGIC ---
        
        # Load historical financial data
        historical_data_path = credit_analysis_folder / "final_derived_financials.json"
        if not historical_data_path.exists():
            raise FileNotFoundError(f"Historical financial data not found at {historical_data_path}")
            
        with open(historical_data_path, "r", encoding="utf-8") as f:
            historical_data = json.load(f)
        
        # Initialize LLM (UNCHANGED)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-04-17",
            temperature=0.0,
            top_p=getattr(config, "LLM_TOP_P", 1.0),
            google_api_key=getattr(config, "GOOGLE_API_KEY", None),
        )
        
        # Create prompt for the LLM using the NEWLY GENERATED schema
        prompt = _create_baseline_drivers_prompt(
            historical_data=historical_data,
            industry_research=industry_research,
            country_research=country_research,
            projection_drivers_schema=projection_drivers_schema, # Pass the generated schema
            security_id=security_id,
            issuer_business_profile=issuer_business_profile
        )
        
        # Call the LLM (UNCHANGED)
        logger.info("Calling LLM to generate baseline projection drivers")
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        
        # Parse the LLM response to extract the projection drivers (UNCHANGED)
        # Your existing robust JSON parsing logic remains here...
        try:
            # Try to extract JSON from the response
            response_text = response.content
            logger.info(f"Raw LLM response length: {len(response_text)}")
            # Write full response to a debug file
            debug_file = projections_folder / "llm_response_debug.txt"
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(response_text)
            logger.info(f"Wrote full LLM response to {debug_file}")
                
            # Find JSON content between triple backticks
            json_match = re.search(r'```json\n(.+?)\n```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                logger.info("Found JSON content between triple backticks")
            else:
                # Try to find any JSON-like content - match from first { to last }
                json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    logger.info("Found JSON content using regex pattern")
                else:
                    raise ValueError("Could not extract JSON from LLM response")
            
            # Write the extracted JSON to a debug file
            json_debug_file = projections_folder / "extracted_json_debug.txt"
            with open(json_debug_file, "w", encoding="utf-8") as f:
                f.write(json_str)
            logger.info(f"Wrote extracted JSON to {json_debug_file}")
            
            # Use a more robust approach to parse the JSON
            try:
                # Try to parse as is first
                baseline_drivers = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                # If that fails, try to clean it up
                logger.info("Initial JSON parsing failed, attempting to clean and fix the JSON")
                
                # Try a different approach - use ast.literal_eval to safely evaluate the string
                try:
                    import ast
                    # First, find the outermost valid JSON object
                    first_brace = json_str.find('{')
                    last_brace = json_str.rfind('}')
                    if first_brace != -1 and last_brace != -1:
                        json_str = json_str[first_brace:last_brace+1]
                        
                        # Try using the json5 library if available (handles more relaxed JSON)
                        try:
                            import json5
                            baseline_drivers = json5.loads(json_str)
                            logger.info("Successfully parsed with json5")
                        except (ImportError, ValueError) as e:
                            logger.info(f"json5 parsing failed: {e}, trying advanced cleanup")
                            
                            # Enhanced cleanup for common LLM JSON errors
                            # 1. Fix the specific error with unexpected comma at line 3, column 6
                            lines = json_str.split('\n')
                            if len(lines) > 2 and len(lines[2]) > 5:
                                # Check for unexpected comma and remove it
                                if lines[2][5] == ',':
                                    lines[2] = lines[2][:5] + lines[2][6:]
                                    logger.info("Fixed unexpected comma at line 3, column 6")
                            json_str = '\n'.join(lines)
                            
                            # 2. Additional common JSON fixes
                            json_str = re.sub(r'\n\s*\n', '\n', json_str)  # Remove empty lines
                            json_str = re.sub(r'([^\\])"([^"]*?)\n([^"]*?)"', r'\1"\2 \3"', json_str)  # Fix newlines in strings
                            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas in objects
                            json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                            json_str = re.sub(r'\s*\/\/.*?\n', '\n', json_str)  # Remove JS-style comments
                            
                            # Try standard json loads with enhanced cleanup
                            try:
                                baseline_drivers = json.loads(json_str)
                                logger.info("Successfully parsed with enhanced cleanup")
                            except json.JSONDecodeError as json_err:
                                logger.error(f"Enhanced cleanup failed: {json_err}")
                                logger.info("Attempting more aggressive JSON repair")
                                
                                try:
                                    # More aggressive approach - try to fix JSON structure issues
                                    # Save the problematic JSON for debugging
                                    debug_path = credit_analysis_folder / "projections" / "problematic_json_debug.txt"
                                    with open(debug_path, "w", encoding="utf-8") as f:
                                        f.write(json_str)
                                    logger.info(f"Saved problematic JSON to {debug_path}")
                                    
                                    # Create a minimal valid structure for universal driver schema
                                    baseline_drivers = {
                                        "company_info": {
                                            "security_id": security_id,
                                            "company_name": "",
                                            "industry_type": ""
                                        },
                                        "drivers": []
                                    }
                                    
                                    try:
                                        # Try to extract company_info
                                        company_info_match = re.search(r'"company_info"\s*:\s*({[^{}]*})', json_str, re.DOTALL)
                                        if company_info_match:
                                            try:
                                                company_info_str = company_info_match.group(1)
                                                # Use format() instead of f-string to avoid brace escaping issues
                                                company_info = json.loads('{' + company_info_str + '}')
                                                baseline_drivers["company_info"].update(company_info)
                                                logger.info("Successfully extracted company_info")
                                            except Exception as e:
                                                logger.error(f"Failed to parse company_info: {e}")
                                        
                                        # Try to extract drivers array
                                        drivers_match = re.search(r'"drivers"\s*:\s*(\[[\s\S]*?\])', json_str)
                                        if drivers_match:
                                            try:
                                                drivers_str = drivers_match.group(1)
                                                # Clean up the drivers string
                                                drivers_str = re.sub(r',\s*]', ']', drivers_str)  # Remove trailing commas
                                                drivers = json.loads(drivers_str)
                                                baseline_drivers["drivers"] = drivers
                                                logger.info(f"Successfully extracted {len(drivers)} drivers")
                                            except Exception as e:
                                                logger.error(f"Failed to parse drivers array: {e}")
                                    except Exception as e:
                                        logger.error(f"Error during manual extraction: {e}")
                                    
                                except Exception as inner_e:
                                    logger.error(f"Failed in manual JSON extraction: {inner_e}")
                                    # Create minimal valid structure for universal driver schema
                                    baseline_drivers = {
                                        "company_info": {
                                            "security_id": security_id,
                                            "company_name": "",
                                            "industry_type": ""
                                        },
                                        "drivers": []
                                    }
                    else:
                        raise ValueError("Could not find valid JSON object in response")
                except Exception as e:
                    logger.error(f"Failed to clean and parse JSON: {e}")
                    raise ValueError(f"Failed to parse LLM response after cleanup attempts: {e}")

        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.error(f"Raw response: {response.content}")
            raise ValueError(f"Failed to parse LLM response: {e}")

        # Save the baseline drivers to the projections folder (UNCHANGED)
        baseline_drivers_path = projections_folder / "baseline_projection_drivers.json"
                
        # Save the validated baseline drivers (UNCHANGED)
        with open(baseline_drivers_path, "w", encoding="utf-8") as f:
            json.dump(baseline_drivers, f, indent=2)
        
        logger.info(f"Successfully generated and saved baseline projection drivers to {baseline_drivers_path}")
        
        # Prepare the success response (UNCHANGED)
        result = {
            "baseline_drivers": baseline_drivers,
            "metadata": {
                "security_id": security_id,
                "status": "success",
                "source": "generate_baseline_projection_drivers"
            }
        }

    except Exception as e:
        error_msg = f"Error in generate_baseline_projection_drivers: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        result = {"error": error_msg}
    
    # Ensure we always return a dictionary (UNCHANGED)
    return result


def _create_baseline_drivers_prompt(
    historical_data: dict,
    industry_research: dict,
    country_research: dict,
    projection_drivers_schema: dict,
    security_id: str,
    issuer_business_profile: dict = None
) -> str:
    """
    Creates a clear, focused, and constrained prompt for the LLM to populate a
    dynamically generated projection drivers schema.
    """
    # 1. Prepare the data inputs for the prompt (UNCHANGED)
    historical_summary = json.dumps(historical_data.get('mapped_historical_data', []), indent=2)
    industry_schema_type = historical_data.get('metadata', {}).get('industry_schema_applied', 'N/A')
    projection_schema_str = json.dumps(projection_drivers_schema, indent=2)
    issuer_profile_summary = json.dumps(issuer_business_profile, indent=2) if issuer_business_profile else "No issuer business profile available."
    industry_summary = json.dumps(industry_research, indent=2) if industry_research else "No industry research data available."
    country_summary = json.dumps(country_research, indent=2) if country_research else "No country research data available."

    # 2. Construct the prompt
    prompt_sections = [
        f"# TASK: GENERATE BASELINE FINANCIAL PROJECTION DRIVERS FOR security_id: {security_id}",
        "\nYou are an expert senior financial analyst. Your task is to generate a complete and economically sensible set of drivers for a 3-year financial forecast.",

        "\n\n## INSTRUCTIONS",
        "1.  **Analyze Context:** Thoroughly review all provided historical data, business profile, and external research to understand the company's business model and operating environment.",
        "2.  **Adhere to the Schema:** Your sole output **MUST** be a single JSON object that strictly validates against the provided `Projection Drivers Schema`. The schema's structure is non-negotiable.",
        "3.  **Anchor to Interim Data (CRITICAL):** If historical data for an interim period within the first forecast year (e.g., Q1, Q2, or Q3 of Year 1) is available, your `base_value` for that driver **MUST** be a realistic, annualized figure that logically continues from those actual results. Do not simply extrapolate from prior full years if it contradicts the latest data.",
        "4.  **Define the Full Projection Lifecycle:** For each driver's `projection` object, you must define the complete forecast path:",
        "    -   **`base_value`:** The starting value for the driver in the first forecast year (Year 1), anchored to interim data if available.",
        "    -   **`trends`:** The multi-stage evolution of the driver, including `short_term` (Years 1-3), `medium_term` (Years 4-9), and a stable `long_term_terminal_value` (Year 10+).",
        "5.  **Provide Justification:** For each driver, provide a concise `justification` explaining your reasoning for the chosen `base_value` and `trends`.",
        "6.  **Use Positive Ratios:** All `base_value` and `trends` values for ratios **MUST BE POSITIVE**. The projection engine handles signs automatically.",

        "\n\n## PROVIDED DATA & SCHEMA",
        f"\n### Historical Financial Data (Industry: {industry_schema_type})",
        "```json",
        historical_summary,
        "```",

        "\n### Issuer Business Profile",
        "```json",
        issuer_profile_summary,
        "```",

        "\n### Industry Research",
        "```json",
        industry_summary,
        "```",

        "\n### Country Research",
        "```json",
        country_summary,
        "```",

        "\n### Projection Drivers Schema (Your Output Target)",
        "Your response MUST be a single JSON object that validates against this schema. The `description` for each driver provides critical guidance.",
        "```json",
        projection_schema_str,
        "```",

        "\n\n## FINAL OUTPUT REQUIREMENTS",
        "1.  Your response must be **ONLY** a single, well-formed JSON object.",
        "2.  Enclose the entire JSON object in triple backticks (```json ... ```).",
        "3.  Do not include any text or explanations outside of the JSON structure."
    ]

    return "\n".join(prompt_sections)


def _load_industry_schema(industry_type: str) -> dict:
    """Loads the specified industry schema file."""
    # This pathing assumes a specific project structure. Adjust if necessary.
    # It assumes the script is run from a location where this relative path is valid.
    project_root = Path(__file__).resolve().parents[4] 
    schema_path = project_root / 'subagents/financial_analyst/mcp_servers/mcp_data_extractor/industry_schemas' / f"{industry_type}_schema.json"
    
    if not schema_path.exists():
        raise FileNotFoundError(f"Industry schema file '{industry_type}_schema.json' not found at: {schema_path}")
        
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema_data = json.load(f)
        
    # The schema is expected to be nested under a key like 'finance_schema'
    key = f"{industry_type}_schema"
    if key not in schema_data:
        raise KeyError(f"Schema for type '{industry_type}' not found within the JSON file at key '{key}'.")
        
    return schema_data[key]


def _generate_drivers_schema_from_industry_schema(industry_schema: dict, industry_type: str) -> dict:
    """
    Parses an industry schema and generates a corresponding, explicit drivers_schema JSON.
    This new schema is structured as an object to enforce completeness and provide
    unambiguous, machine-readable links between drivers and financial statement items.
    """
    # Define the sophisticated "Projection" object with the multi-stage trend system.
    projection_definition = {
        "type": "object",
        "description": "Defines the full lifecycle projection for a driver metric.",
        "properties": {
            "base_value": {
                "type": "number",
                "description": "The starting value for the driver in the first forecast year (Year 1). MUST BE POSITIVE for ratios."
            },
            "trends": {
                "type": "object",
                "description": "Defines annual changes to be applied to the base value over time.",
                "properties": {
                    "short_term_annual_change": {
                        "type": "number",
                        "description": "The value to add/subtract each year for the first 3 years (e.g., -0.001 for -0.1% per year)."
                    },
                    "medium_term_annual_change": {
                        "type": "number",
                        "description": "The value to add/subtract each year for years 4-9."
                    },
                    "long_term_terminal_value": {
                        "type": "number",
                        "description": "The stable value the driver is assumed to hold in perpetuity from year 10 onwards."
                    }
                },
                "required": ["short_term_annual_change", "medium_term_annual_change", "long_term_terminal_value"]
            }
        },
        "required": ["base_value", "trends"]
    }

    # Define the schema for a single driver entry
    driver_entry_schema = {
        "type": "object",
        "properties": {
            "driver_tier": {
                "type": "string",
                "description": "The modeling tier (Tier 1 or Tier 2) this driver belongs to."
            },
            "metric": {
                "type": "string",
                "description": "The specific metric used for projection (e.g., 'growth_rate', '%_of_revenue')."
            },
            "justification": {
                "type": "string",
                "description": "Analyst's (your) concise reasoning for the chosen projection values and trends."
            },
            "projection": {
                "$ref": "#/definitions/Projection"
            }
        },
        "required": ["driver_tier", "metric", "justification", "projection"]
    }

    # -- Main Schema Construction --
    drivers_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": f"Projection Drivers for {industry_type.title()} Industry",
        "description": "A dynamically generated schema defining the required drivers for a multi-stage financial projection.",
        "type": "object",
        "definitions": {
            "Projection": projection_definition,
            "DriverEntry": driver_entry_schema
        },
        "properties": {
            "company_info": {
                "type": "object",
                "properties": {
                    "security_id": {"type": "string"},
                    "company_name": {"type": "string"},
                    "industry_type": {"type": "string", "enum": [industry_type]}
                },
                "required": ["security_id", "company_name", "industry_type"]
            },
            "drivers": {
                "type": "object",
                "description": "A dictionary of all required drivers, where the key is the exact financial statement line item being driven.",
                "properties": {}, # To be populated below
                "required": []    # To be populated below
            }
        },
        "required": ["company_info", "drivers"]
    }

    driver_properties = {}
    required_drivers = []

    # Iterate through IS, BS, and CFS to find all driver-based items
    for statement_type in ['income_statement', 'balance_sheet', 'cash_flow_statement']:
        statement_schema = industry_schema.get(statement_type, {})
        for item_name, properties in statement_schema.items():
            proj_logic = properties.get("projection_logic", {})

            if proj_logic.get("forecast_method") == "driver_based":
                driver_metric = proj_logic.get("default_driver_metric")
                if not driver_metric:
                    continue
                
                # Create the explicit, machine-readable key
                # e.g., "income_statement.loan_loss_provisions"
                driver_key = f"{statement_type}.{item_name}"
                
                # Add this driver to the schema's properties
                driver_properties[driver_key] = {
                    "$ref": "#/definitions/DriverEntry",
                    "description": f"Driver for the '{item_name}' line item. Default metric: '{driver_metric}'."
                }
                
                # Add this driver to the list of required keys
                required_drivers.append(driver_key)

    # Populate the "drivers" object in the main schema
    drivers_schema["properties"]["drivers"]["properties"] = driver_properties
    drivers_schema["properties"]["drivers"]["required"] = required_drivers
    
    return drivers_schema
    
    
async def run_stdio_server():
    await mcp_app.run_stdio_async()

def main():
    import asyncio
    asyncio.run(run_stdio_server())

@mcp_app.tool()
async def reconcile_debt_schedule(security_id: str) -> dict:
    """
    Reconciles the debt schedule with financial statements to ensure accurate debt classification.
    
    This tool:
    1. Loads the reclassified historical data and debt schedule
    2. Classifies each debt instrument as short-term or long-term based on maturity
    3. Reconciles debt amounts with financial statements
    4. Redistributes amounts from other liability accounts if needed
    5. Returns the updated financial data with reconciled debt amounts
    
    Args:
        security_id: The security ID to reconcile debt for
        
    Returns:
        Dictionary containing the updated financial data with reconciled debt
    """
    try:
        logger.info(f"Starting debt reconciliation for security: {security_id}")
        
        # Get security folder and required file paths
        try:
            security_folder = require_security_folder(security_id)
            logger.info(f"Security folder: {security_folder}")
            
            debt_schedule_path = get_security_file(security_id, "data_extraction/historical_debt_schedule.json")
            financial_data_path = get_security_file(security_id, "credit_analysis/validated_historical_data.json")
            logger.info(f"Debt schedule path: {debt_schedule_path}")
            logger.info(f"Financial data path: {financial_data_path}")
            
            # Verify files exist
            if not os.path.exists(debt_schedule_path):
                raise FileNotFoundError(f"Debt schedule file not found: {debt_schedule_path}")
            if not os.path.exists(financial_data_path):
                raise FileNotFoundError(f"Financial data file not found: {financial_data_path}")
            
        except Exception as e:
            logger.error(f"Error setting up file paths: {str(e)}", exc_info=True)
            raise
        
        # Load the data
        try:
            with open(debt_schedule_path, 'r') as f:
                debt_schedule = json.load(f).get('historical_debt_schedule', [])
                logger.info(f"Loaded {len(debt_schedule)} debt schedule entries")
            
            with open(financial_data_path, 'r') as f:
                financial_data = json.load(f)
                logger.info(f"Loaded financial data with {len(financial_data.get('mapped_historical_data', []))} periods")
                
        except json.JSONDecodeError as e:
            error_msg = f"Error parsing JSON file: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"status": "error", "message": error_msg}
        except Exception as e:
            error_msg = f"Error loading data files: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"status": "error", "message": error_msg}
        
        try:
            # Process each period in the financial data
            for period_idx, period_data in enumerate(financial_data.get('mapped_historical_data', [])):
                period_end = period_data.get('reporting_period_end_date')
                if not period_end:
                    logger.warning(f"Skipping period {period_idx} - missing reporting_period_end_date")
                    continue
                
                logger.info(f"Processing period {period_idx}: {period_end}")
                
                # Get the most recent debt schedule that's on or before the period end date
                try:
                    period_date = datetime.strptime(period_end, '%Y-%m-%d')
                    
                    # Find all debt schedules with dates <= current period
                    valid_debt_schedules = []
                    period_end_year = period_date.year
                    for schedule in debt_schedule:
                        try:
                            schedule_year = datetime.strptime(schedule.get('period_label', ''), '%Y-%m-%d').year
                            if schedule_year == period_end_year:
                                valid_debt_schedules.append(schedule)
                        except (ValueError, TypeError):
                            continue
                    
                    # If we found valid schedules, use them
                    if valid_debt_schedules:
                        # Use the most recent schedule
                        period_debt = valid_debt_schedules
                        logger.info(f"Using debt schedule from {valid_debt_schedules[0].get('period_label')} for period {period_end}")
                    else:
                        logger.info(f"No valid debt schedule found for period {period_end}")
                        continue
                        
                except Exception as e:
                    logger.error(f"Error processing debt schedules for period {period_end}: {str(e)}")
                    continue
                
                # Classify debt by maturity
                try:
                    period_date = datetime.strptime(period_end, '%Y-%m-%d')
                except ValueError as e:
                    logger.error(f"Invalid date format in period {period_end}: {str(e)}")
                    continue
                    
                current_liab = period_data.get('balance_sheet', {}).get('current_liabilities', {})
                non_current_liab = period_data.get('balance_sheet', {}).get('non_current_liabilities', {})
                
                # Initialize debt amounts
                short_term_debt = 0.0
                long_term_debt = 0.0
                
                logger.info(f"Processing {len(period_debt)} debt entries for period {period_end}")
                
                # Calculate total debt from schedule
                for debt in period_debt:
                    try:
                        maturity = datetime.strptime(debt['maturity_date'], '%Y-%m-%d')
                        amount = float(debt.get('principal_amount_outstanding_at_period_end', 0))
                        
                        logger.info(f"Processing debt: {debt.get('debt_instrument_name')}, "
                                    f"Amount: {amount}, Maturity: {debt['maturity_date']}")
                        
                        days_to_maturity = (maturity - period_date).days
                        logger.info(f"Days to maturity: {days_to_maturity}")
                        
                        if days_to_maturity <= 365:
                            short_term_debt += amount
                            logger.info(f"Classified as short-term debt. New ST total: {short_term_debt}")
                        else:
                            long_term_debt += amount
                            logger.info(f"Classified as long-term debt. New LT total: {long_term_debt}")
                            
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Error processing debt entry: {str(e)}")
                        logger.debug(f"Problematic debt entry: {debt}", exc_info=True)
                        continue
                
                logger.info(f"Period {period_end}: Calculated ST Debt: {short_term_debt}, LT Debt: {long_term_debt})")
                
                # Ensure balance_sheet, current_liabilities, and non_current_liabilities exist
                if 'balance_sheet' not in period_data:
                    period_data['balance_sheet'] = {}
                if 'current_liabilities' not in period_data['balance_sheet']:
                    period_data['balance_sheet']['current_liabilities'] = {}
                if 'non_current_liabilities' not in period_data['balance_sheet']:
                    period_data['balance_sheet']['non_current_liabilities'] = {}

                # Calculate the increase needed for short-term and long-term debt
                st_debt_increase = short_term_debt - (period_data['balance_sheet']['current_liabilities'].get('short_term_debt', 0) or 0)
                lt_debt_increase = long_term_debt - (period_data['balance_sheet']['non_current_liabilities'].get('long_term_debt', 0) or 0)

                # Redistribute for short-term debt if there's an increase
                if st_debt_increase > 0:
                    current_liab_accounts = period_data['balance_sheet'].get('current_liabilities', {})
                    if current_liab_accounts:
                        redistribute_from_largest(current_liab_accounts, 'short_term_debt', st_debt_increase, exclude=['short_term_debt'])

                # Redistribute for long-term debt if there's an increase
                if lt_debt_increase > 0:
                    non_current_liab_accounts = period_data['balance_sheet'].get('non_current_liabilities', {})
                    if non_current_liab_accounts:
                        redistribute_from_largest(non_current_liab_accounts, 'long_term_debt', lt_debt_increase, exclude=['long_term_debt'])

                # Update the period data directly in the financial_data structure
                if 'balance_sheet' not in period_data:
                    period_data['balance_sheet'] = {}
                if 'current_liabilities' not in period_data['balance_sheet']:
                    period_data['balance_sheet']['current_liabilities'] = {}
                if 'non_current_liabilities' not in period_data['balance_sheet']:
                    period_data['balance_sheet']['non_current_liabilities'] = {}
                    
                period_data['balance_sheet']['current_liabilities']['short_term_debt'] = short_term_debt
                period_data['balance_sheet']['non_current_liabilities']['long_term_debt'] = long_term_debt
                
                # Also update the top-level debt fields for backward compatibility
                period_data['balance_sheet']['short_term_debt'] = short_term_debt
                period_data['balance_sheet']['long_term_debt'] = long_term_debt

                # We only update the debt items and the largest liability account
                # The balance sheet's own logic will handle the totals
                logger.info("Only updated debt items and largest liability account. Balance sheet totals will be recalculated by the system.")
                
                logger.info(f"Period {period_end}: Updated ST Debt: {short_term_debt}, LT Debt: {long_term_debt}")
                logger.info(f"Updated period_data: {json.dumps(period_data, indent=2)}")
                
        except Exception as e:
            error_msg = f"Error processing financial data: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"status": "error", "message": error_msg}
        
        # Save the updated financial data
        try:
            output_path = get_security_file(security_id, "credit_analysis/reconciled_financial_data.json")
            with open(output_path, 'w') as f:
                json.dump(financial_data, f, indent=2)
                
            logger.info(f"Successfully saved reconciled financial data to {output_path}")
            
        except Exception as e:
            error_msg = f"Error saving reconciled financial data: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"status": "error", "message": error_msg}
        
        return {
            "status": "success", 
            "message": f"Debt schedule reconciled and saved to {output_path}",
            "financial_data": financial_data
        }
        
    except Exception as e:
        error_msg = f"Unexpected error in reconcile_debt_schedule: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"status": "error", "message": error_msg}
        
    except Exception as e:
        logger.error(f"Error reconciling debt schedule for {security_id}: {str(e)}", exc_info=True)
        return {"status": "error", "message": f"Failed to reconcile debt schedule: {str(e)}"}

def redistribute_from_largest(accounts: dict, target_account: str, amount: float, exclude: list = []) -> None:
    """
    Redistribute amount from the largest account to the target account.
    
    Args:
        accounts: Dictionary of account names to values
        target_account: The account to add the amount to
        amount: Amount to redistribute
        exclude: List of accounts to exclude from redistribution
    """
    if amount <= 0:
        return
    
    # Find the largest account that's not excluded
    candidates = [(k, abs(v)) for k, v in accounts.items() 
                 if k not in exclude and v is not None and k != target_account]
    
    if not candidates:
        return
        
    # Sort by absolute value in descending order
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Get the largest account
    largest_account, largest_amount = candidates[0]
    
    # Calculate how much we can take (don't make the account negative)
    available = min(amount, largest_amount)
    
    # Update the accounts
    accounts[largest_account] = largest_amount - available
    accounts[target_account] = accounts.get(target_account, 0) + available
    
    # Log the redistribution
    logger.info(f"Redistributed {available} from {largest_account} to {target_account}")


if __name__ == "__main__":
    main()
