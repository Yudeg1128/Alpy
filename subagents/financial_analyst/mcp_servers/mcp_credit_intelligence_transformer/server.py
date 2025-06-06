"""
MCP Server for Credit Intelligence Transformation Phase
"""
import asyncio
import json
import logging
import sys
from pathlib import Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "financial_analyst"))
from field_mapping import get_best_field, get_best_assumption
from security_folder_utils import require_security_folder, get_subfolder, get_security_file
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP
from mcp.types import Implementation

# Ensure config.py is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "financial_analyst"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MCPCreditIntelligenceTransformer")
logger.info("Starting MCP Credit Intelligence Transformer server")

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
    financial_model_output: dict
    qualitative_inputs: dict  # issuer_business_profile, industry_profile, collateral_and_protective_clauses

# --- Tool Actions ---
@mcp_app.tool()
async def generate_assumptions(security_id: str) -> dict:
    """
    Hybrid deterministic + LLM logic for financial projection assumptions.
    1. Deterministic: Compute historical averages, ratios, extrapolations, and fill as much of the schema as possible.
    2. LLM: Refine, contextualize, fill gaps, and provide commentary.
    Output: Fully populated FinancialProjectionAssumptionsSchema.
    """
    # --- Output path validation/creation (use argument, not data) ---
    if not security_id:
        raise ValueError("security_id argument must be provided to tool action.")
    import logging
    logger = logging.getLogger("MCPCreditIntelligenceTransformer")
    try:
        data_path = get_subfolder(security_id, "data_extraction") / "complete_data.json"
        if not data_path.exists():
            logger.error(f"complete_data.json not found for security_id {security_id} at {data_path}")
            raise FileNotFoundError(f"complete_data.json not found for security_id {security_id} at {data_path}")
        logger.info(f"Found complete_data.json at: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)
    except Exception as e:
        logger.error(f"Error locating complete_data.json for security_id {security_id}: {e}")
        raise


    logger.info("Starting deterministic pre-processing for assumptions...")
    import numpy as np
    from copy import deepcopy
    # Load output schema
    schema_path = Path(__file__).parent / "generate_assumptions_output_schema.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        output_schema = json.load(f)

    # --- Load country economic context ---
    country_report_path = Path(__file__).parent.parent.parent / "external_reports" / "country_research.json"
    with open(country_report_path, "r", encoding="utf-8") as f:
        country_report = json.load(f)
    logger.info(f"Loaded country economic report: {country_report}")
    # Add to output: instruct LLM to use this for inflation/fx/macro_risks
    country_economic_context = {
        "country": country_report.get("country", "Mongolia"),
        "report_date": country_report.get("report_date", "2025-01-01"),
        "inflation": country_report["inflation"],
        "fx": country_report["fx"],
        "macro_risks": country_report["macro_risks"],
        "sources": country_report["sources"]
    }

    bond_fin_hist = input_data.get("bond_financials_historical", {})
    hist_raw = bond_fin_hist.get("historical_financial_statements", [])
    
    # Check if we have truncated data and try to load complete historical data
    has_truncated_data = any(isinstance(item, str) and "truncated" in item for item in hist_raw)
    
    if has_truncated_data:
        logger.warning(f"Detected truncated historical data in complete_data.json. Attempting to load complete data from dedicated file...")
        try:
            # Try to load the complete historical data from the dedicated file
            hist_file_path = get_subfolder(security_id, "data_extraction") / "bond_financials_historical.json"
            if hist_file_path.exists():
                with open(hist_file_path, "r", encoding="utf-8") as f:
                    complete_hist_data = json.load(f)
                complete_hist = complete_hist_data.get("bond_financials_historical", {}).get("historical_financial_statements", [])
                logger.info(f"Successfully loaded complete historical data: {len(complete_hist)} periods from dedicated file")
                hist = [item for item in complete_hist if isinstance(item, dict)]
            else:
                logger.error(f"Dedicated historical file not found at {hist_file_path}")
                # Fallback to filtered data from complete_data.json
                hist = [item for item in hist_raw if isinstance(item, dict)]
        except Exception as e:
            logger.error(f"Failed to load complete historical data: {e}")
            # Fallback to filtered data from complete_data.json
            hist = [item for item in hist_raw if isinstance(item, dict)]
    else:
        # Filter out any non-dict items from the historical statements
        hist = []
        for i, item in enumerate(hist_raw):
            if isinstance(item, dict):
                hist.append(item)
            else:
                logger.warning(f"Skipping non-dict historical statement at index {i}: {type(item)} - {str(item)[:100]}...")
    
    logger.info(f"Loaded {len(hist)} valid historical statements for analysis")
    # Compute projection periods (5 future years, including current year)
    import datetime
    today = datetime.date.today()
    # Try to get the latest year from bond_metadata or financials
    maturity_year = None
    try:
        bond_metadata = input_data.get("bond_metadata", {})
        if isinstance(bond_metadata, dict):
            maturity_date = bond_metadata.get("maturity_date")
            if maturity_date:
                maturity_year = int(str(maturity_date)[:4])
    except Exception:
        pass
    # Use latest historical year or current year if not found
    latest_hist_year = None
    for h in reversed(hist):
        d = h.get("reporting_period_end_date") or h.get("period")
        if d:
            try:
                latest_hist_year = int(str(d)[:4])
                break
            except Exception:
                continue
    base_year = max([y for y in [maturity_year, latest_hist_year, today.year] if y is not None])
    projection_periods_list = [str(base_year + i) for i in range(5)]
    # Helper: Compute CAGR
    def compute_cagr(start, end, n):
        try:
            if (
                start is None or end is None or n is None or n <= 0 or start == 0
                or not isinstance(start, (int, float)) or not isinstance(end, (int, float))
            ):
                return None
            return (end / start) ** (1 / n) - 1
        except Exception:
            return None
    # Helper: Compute avg ratio
    def avg_ratio(numer, denom):
        arr = []
        for n, d in zip(numer, denom):
            try:
                if n is not None and d not in (None, 0) and isinstance(n, (int, float)) and isinstance(d, (int, float)):
                    arr.append(n/d)
            except Exception:
                continue
        return float(np.mean(arr)) if arr else None
    # Extract series for key metrics
    def extract_series(metric):
        return [safe_get(h, "financials", metric) for h in hist]
    # Revenue growth (CAGR basis is dynamic: len(vals)-1)
    revs = extract_series("total_revenue")
    revenue_growth = compute_cagr(revs[0], revs[-1], len(revs)-1) if len(revs) > 1 and revs[0] and revs[-1] else None
    # Net income growth
    netincomes = extract_series("net_profit")
    net_income_growth = compute_cagr(netincomes[0], netincomes[-1], len(netincomes)-1) if len(netincomes) > 1 and netincomes[0] and netincomes[-1] else None
    # Asset growth
    assets = extract_series("total_assets")
    asset_growth = compute_cagr(assets[0], assets[-1], len(assets)-1) if len(assets) > 1 and assets[0] and assets[-1] else None
    # Expense ratios
    interest_exp = extract_series("interest_expense")
    avg_interest_exp_pct_rev = avg_ratio(interest_exp, revs) if revs and interest_exp else None
    # Capex ratios
    capex = extract_series("capital_expenditures")
    avg_capex_pct_rev = avg_ratio(capex, revs) if revs and capex else None
    avg_capex_pct_assets = avg_ratio(capex, assets) if assets and capex else None
    # Tax rate
    pbt = extract_series("profit_before_tax")
    taxes = extract_series("income_tax")
    avg_tax_rate = avg_ratio(taxes, pbt) if pbt and taxes else None
    # Working capital (as % revenue)
    wc = extract_series("working_capital")
    avg_wc_pct_rev = avg_ratio(wc, revs) if wc and revs else None

    # Build deterministic assumptions (partial)
    det_assumptions = deepcopy(output_schema["properties"])
    det_assumptions["projection_horizon_years"] = len(projection_periods_list)
    bond_metadata = input_data.get("bond_metadata", {})
    det_assumptions["base_currency"] = safe_get(bond_metadata, "currency") if isinstance(bond_metadata, dict) else "MNT"
    det_assumptions["assumptions_by_category"] = {
        "revenue_assumptions": [],
        "expense_assumptions": [],
        "capital_expenditure_assumptions": [],
        "working_capital_assumptions": [],
        "debt_assumptions": [],
        "tax_assumptions": [],
        "other_operating_assumptions": []
    }
    # Extract loan growth rates for consistency with interest income
    loans = extract_series("net_loans")
    loan_growth = compute_cagr(loans[0], loans[-1], len(loans)-1) if len(loans) > 1 and loans[0] and loans[-1] else None
    
    # Populate deterministic revenue growth - for NBFIs, align with loan growth
    if revenue_growth is not None:
        # For NBFIs, interest income should grow in line with loan portfolio
        # Use a tapering approach for more realistic projections
        if loan_growth is not None:
            # Use loan growth as base with gradual tapering
            tapered_growth = []
            base_growth = min(loan_growth, revenue_growth)  # More conservative of the two
            for i in range(len(projection_periods_list)):
                # Taper down gradually over projection period
                taper_factor = max(0.7, 1.0 - (i * 0.075))  # Reduce by 7.5% each year, floor at 70%
                tapered_growth.append(base_growth * taper_factor)
            
            det_assumptions["assumptions_by_category"]["revenue_assumptions"].append({
                "metric_name": "total_revenue_growth",
                "assumption_type": "growth_rate",
                "values": tapered_growth,
                "projection_periods": projection_periods_list,
                "basis_of_assumption": "Derived from historical loan and revenue growth with realistic tapering to reflect maturing business",
                "unit_notes": ""
            })
        else:
            # Fallback to original approach if loan growth not available
            det_assumptions["assumptions_by_category"]["revenue_assumptions"].append({
                "metric_name": "total_revenue_growth",
                "assumption_type": "growth_rate",
                "values": [revenue_growth] * len(projection_periods_list),
                "projection_periods": projection_periods_list,
                "basis_of_assumption": "Derived programmatically from 3-year CAGR of total_revenue",
                "unit_notes": ""
            })
        
    # Populate deterministic net income growth
    if net_income_growth is not None:
        det_assumptions["assumptions_by_category"]["expense_assumptions"].append({
            "metric_name": "net_income_growth",
            "assumption_type": "growth_rate",
            "values": [net_income_growth] * len(projection_periods_list),
            "projection_periods": projection_periods_list,
            "basis_of_assumption": "Derived programmatically from 3-year CAGR of net_profit",
            "unit_notes": ""
        })
    # Asset growth as other operating assumption
    if asset_growth is not None:
        # For NBFIs, asset growth should be more aligned with loan growth since loans are the primary asset
        if loan_growth is not None:
            # Use similar tapering approach as revenue growth
            tapered_asset_growth = []
            base_asset_growth = min(loan_growth * 1.1, asset_growth)  # Slightly higher than loan growth but capped
            for i in range(len(projection_periods_list)):
                # Taper down gradually over projection period
                taper_factor = max(0.7, 1.0 - (i * 0.075))  # Reduce by 7.5% each year, floor at 70%
                tapered_asset_growth.append(base_asset_growth * taper_factor)
            
            det_assumptions["assumptions_by_category"]["other_operating_assumptions"].append({
                "metric_name": "asset_growth",
                "assumption_type": "growth_rate",
                "values": tapered_asset_growth,
                "projection_periods": projection_periods_list,
                "basis_of_assumption": "Derived from historical asset and loan growth with realistic tapering to reflect maturing business",
                "unit_notes": ""
            })
        else:
            # Fallback to original approach
            det_assumptions["assumptions_by_category"]["other_operating_assumptions"].append({
                "metric_name": "asset_growth",
                "assumption_type": "growth_rate",
                "values": [asset_growth] * len(projection_periods_list),
                "projection_periods": projection_periods_list,
                "basis_of_assumption": "Derived programmatically from 3-year CAGR of total_assets",
                "unit_notes": ""
            })
    # Populate deterministic expense ratios
    if avg_interest_exp_pct_rev is not None:
        det_assumptions["assumptions_by_category"]["expense_assumptions"].append({
            "metric_name": "interest_expense_as_pct_of_revenue",
            "assumption_type": "percentage_of_revenue",
            "values": [avg_interest_exp_pct_rev] * len(projection_periods_list),
            "projection_periods": projection_periods_list,
            "basis_of_assumption": "Derived programmatically from historical average",
            "unit_notes": ""
        })
    # Capex
    if avg_capex_pct_rev is not None:
        det_assumptions["assumptions_by_category"]["capital_expenditure_assumptions"].append({
            "metric_name": "capex_as_pct_of_revenue",
            "assumption_type": "percentage_of_revenue",
            "values": [avg_capex_pct_rev] * len(projection_periods_list),
            "projection_periods": projection_periods_list,
            "basis_of_assumption": "Derived programmatically from historical average",
            "unit_notes": ""
        })
    if avg_capex_pct_assets is not None:
        det_assumptions["assumptions_by_category"]["capital_expenditure_assumptions"].append({
            "metric_name": "capex_as_pct_of_assets",
            "assumption_type": "percentage_of_assets",
            "values": [avg_capex_pct_assets] * len(projection_periods_list),
            "projection_periods": projection_periods_list,
            "basis_of_assumption": "Derived programmatically from historical average",
            "unit_notes": ""
        })
    # Tax
    if avg_tax_rate is not None:
        det_assumptions["assumptions_by_category"]["tax_assumptions"].append({
            "metric_name": "tax_rate",
            "assumption_type": "percentage_of_revenue",
            "values": [avg_tax_rate] * len(projection_periods_list),
            "projection_periods": projection_periods_list,
            "basis_of_assumption": "Derived programmatically from historical average",
            "unit_notes": ""
        })
    # Working capital
    if avg_wc_pct_rev is not None:
        det_assumptions["assumptions_by_category"]["working_capital_assumptions"].append({
            "metric_name": "working_capital_as_pct_of_revenue",
            "assumption_type": "percentage_of_revenue",
            "values": [avg_wc_pct_rev] * len(projection_periods_list),
            "projection_periods": projection_periods_list,
            "basis_of_assumption": "Derived programmatically from historical average",
            "unit_notes": ""
        })
    # --- Ensure explicit placeholders for key CFS drivers ---
    # 1. Depreciation as % of PPE
    if not any(a.get("metric_name") == "depreciation_as_pct_of_ppe" for a in det_assumptions["assumptions_by_category"].get("other_operating_assumptions", [])):
        det_assumptions["assumptions_by_category"].setdefault("other_operating_assumptions", []).append({
            "metric_name": "depreciation_as_pct_of_ppe",
            "assumption_type": "percentage_of_assets",
            "values": [None] * len(projection_periods_list),
            "projection_periods": projection_periods_list,
            "basis_of_assumption": "Explicit placeholder: No historical data for depreciation or PPE. LLM must fill or justify.",
            "unit_notes": ""
        })
    # 2. Change in accounts receivable as % of revenue
    if not any(a.get("metric_name") == "change_in_accounts_receivable_as_pct_of_revenue" for a in det_assumptions["assumptions_by_category"].get("working_capital_assumptions", [])):
        det_assumptions["assumptions_by_category"].setdefault("working_capital_assumptions", []).append({
            "metric_name": "change_in_accounts_receivable_as_pct_of_revenue",
            "assumption_type": "percentage_of_revenue",
            "values": [None] * len(projection_periods_list),
            "projection_periods": projection_periods_list,
            "basis_of_assumption": "Explicit placeholder: No historical data for receivables. LLM must fill or justify.",
            "unit_notes": ""
        })
    # 3. Change in accounts payable as % of COGS
    if not any(a.get("metric_name") == "change_in_accounts_payable_as_pct_of_cogs" for a in det_assumptions["assumptions_by_category"].get("working_capital_assumptions", [])):
        det_assumptions["assumptions_by_category"].setdefault("working_capital_assumptions", []).append({
            "metric_name": "change_in_accounts_payable_as_pct_of_cogs",
            "assumption_type": "percentage_of_cogs",
            "values": [None] * len(projection_periods_list),
            "projection_periods": projection_periods_list,
            "basis_of_assumption": "Explicit placeholder: No historical data for payables or COGS. LLM must fill or justify.",
            "unit_notes": ""
        })
    # 4. Change in inventory as % of COGS
    if not any(a.get("metric_name") == "change_in_inventory_as_pct_of_cogs" for a in det_assumptions["assumptions_by_category"].get("working_capital_assumptions", [])):
        det_assumptions["assumptions_by_category"].setdefault("working_capital_assumptions", []).append({
            "metric_name": "change_in_inventory_as_pct_of_cogs",
            "assumption_type": "percentage_of_cogs",
            "values": [None] * len(projection_periods_list),
            "projection_periods": projection_periods_list,
            "basis_of_assumption": "Explicit placeholder: No historical data for inventory or COGS. LLM must fill or justify.",
            "unit_notes": ""
        })
    # --- LLM refinement layer ---
    logger.info("Calling LLM for refinement and completion of assumptions...")
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        import config
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-04-17",
            temperature=0.0,
            top_p=getattr(config, "LLM_TOP_P", 1.0),
            google_api_key=getattr(config, "GOOGLE_API_KEY", None),
        )
        prompt = f"""
You are a financial modeling expert. You are provided with:
1. A partially pre-filled FinancialProjectionAssumptionsSchema (JSON below), with deterministic assumptions derived from historical data.
2. The full original extracted data for a single security (JSON below).
3. COUNTRY_ECONOMIC_CONTEXT (JSON below), containing inflation, FX, macro risks, and sources for the relevant country.

IMPORTANT INSTRUCTIONS:
- ALWAYS use actual historical values when available. NEVER state "no specific historical data is available" when historical data exists.
- For loan_loss_provision_as_pct_of_loans, use the historical actual rate of 4.39% from 2024 Q3 data (9.16B MNT / 208.59B MNT).
- For operating_expenses, use the historical management_and_operating_expenses data to calculate appropriate ratios.
- When historical data exists, explicitly cite the specific historical values and calculations in your basis_of_assumption.

Your tasks:
- Review and validate each pre-filled assumption, adjusting if qualitative information suggests a revision.
- For any missing assumptions, generate reasonable values based on all available data.
- For every assumption, fill the basis_of_assumption field, citing specific data points or qualitative insights.
- Populate llm_comments and input_quality_notes with your process, data limitations, and major judgment calls.
- Use the COUNTRY_ECONOMIC_CONTEXT data to fill and explain the corresponding section in the output schema.
- Output ONLY a valid JSON object matching the provided schema, and nothing else.

SCHEMA:
{json.dumps(output_schema, indent=2)}

PARTIAL_ASSUMPTIONS_OBJECT:
{json.dumps(det_assumptions, indent=2)}

COUNTRY_ECONOMIC_CONTEXT:
{json.dumps(country_economic_context, indent=2)}

ORIGINAL_INPUT_DATA:
{json.dumps(input_data, indent=2)}

Output:
"""
        result = await llm.ainvoke(prompt)
        logger.info(f"Raw LLM response: {result.content}")
        content_str = result.content.strip()
        if content_str.startswith('```'):
            content_str = content_str.split('\n', 1)[-1]
            if content_str.endswith('```'):
                content_str = content_str[:-3].strip()
        try:
            assumptions_json = json.loads(content_str)
        except Exception as e:
            logger.error(f"Failed to parse LLM output: {e}")
            assumptions_json = det_assumptions
        # Save output to credit_analysis folder in security's folder (use validated path)
        analysis_dir = get_subfolder(security_id, "credit_analysis")
        analysis_dir.mkdir(parents=True, exist_ok=True)
        output_path = analysis_dir / "assumptions.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"assumptions": assumptions_json, "_schema": str(schema_path)}, f, indent=2, ensure_ascii=False)
        return {"assumptions": assumptions_json, "_schema": str(schema_path)}
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        # Save deterministic fallback as well
        # security_id is already available as a parameter, no need to extract it again
        if security_id:
            try:
                analysis_dir = get_subfolder(security_id, "credit_analysis")
                analysis_dir.mkdir(parents=True, exist_ok=True)
                out_path = analysis_dir / "assumptions.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump({"assumptions": det_assumptions, "_schema": str(schema_path)}, f, indent=2, ensure_ascii=False)
            except Exception as ex:
                logger.error(f"Failed to save fallback assumptions for {security_id}: {ex}")
        return {"assumptions": det_assumptions, "_schema": str(schema_path)}

from deterministic_financial_model import run_financial_model as _run_financial_model

@mcp_app.tool()
async def run_financial_model(security_id: str) -> dict:
    """
    Deterministic financial model for a bond security. Applies conservative logic and explicit inflation/FX adjustments using real extracted and assumptions data.
    Returns a partial output with the macroeconomic_adjustments section fully populated.
    """
    return await _run_financial_model(security_id)


@mcp_app.tool()
async def summarize_credit_risk(input_data: CreditSummaryInput) -> dict:
    """
    LLM interprets model output, produces narrative summary and risk assessment.
    """
    logger.info("Generating credit risk summary using LLM...")
    # TODO: Implement LLM call, enforce output schema
    return {"credit_summary": {}, "_schema": CREDIT_SUMMARY_OUTPUT_SCHEMA_PATH}

async def run_stdio_server():
    await mcp_app.run_stdio_async()

def main():
    import asyncio
    asyncio.run(run_stdio_server())

if __name__ == "__main__":
    main()
