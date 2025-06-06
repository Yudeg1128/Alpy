"""
MCP Server for Quality Checking Finalized Bond Data JSON
"""
import asyncio
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP
from mcp.types import Implementation

# Ensure config.py is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "financial_analyst"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MCPQualityChecker")
logger.info("Starting MCP Quality Checker server")

mcp_app = FastMCP(
    name="MCPQualityCheckerServer",
    version="0.1.0",
    description="MCP server for quality checking finalized bond data JSON"
)

class QualityCheckInput(BaseModel):
    finalized_data_path: str = Field(..., description="Path to the finalized data JSON file produced by the data extractor.")
    schema_path: Optional[str] = Field(None, description="Optional path to the schema JSON file with comments for LLM context.")

class QualityCheckResult(BaseModel):
    status: str
    message: str
    deterministic_checks: dict
    llm_assessment: Optional[dict] = None

class JsonRepairInput(BaseModel):
    finalized_data_path: str = Field(..., description="Path to the finalized data JSON file to repair.")
    json_patches: list = Field(..., description="List of JSON patch operations to apply.")

class JsonRepairResult(BaseModel):
    status: str
    message: str
    repaired_data_path: Optional[str] = None

# --- Deterministic Quality Checks ---
def run_deterministic_checks(data: dict, schema: dict) -> dict:
    # Type checking
    if not isinstance(data, dict):
        logger.error(f"Expected data to be dict, got {type(data)}")
        return {"passed": False, "problems": [{"type": "DataTypeError", "message": f"Expected data to be dict, got {type(data)}"}], "warnings": []}
    
    if not isinstance(schema, dict):
        logger.warning(f"Expected schema to be dict, got {type(schema)}, treating as empty dict")
        schema = {}
    
    # Initialize results structure
    results = {
        "passed": True,  # Will be set to False if problems are found
        "problems": [],
        "warnings": []  # Separate list for non-critical issues
    }
    # Create references to lists for nested functions
    problems = results["problems"]
    warnings = results["warnings"]
    # 1. Currency Code Consistency
    try:
        currency = data.get("bond_metadata", {}).get("currency")
    except Exception as e:
        logger.error(f"Error accessing currency in bond_metadata: {e}")
        logger.error(f"bond_metadata type: {type(data.get('bond_metadata'))}")
        problems.append({
            "type": "DataAccessError",
            "message": f"Error accessing currency: {str(e)}"
        })
        currency = None
    if not (isinstance(currency, str) and len(currency) == 3 and currency.isupper()):
        problems.append({
            "section": "bond_metadata",
            "field": "currency",
            "period": None,
            "problem_type": "UnitInconsistency",
            "description": "bond_metadata.currency is not a valid ISO 4217 code.",
            "value_found": currency,
            "expected_value": "Valid ISO 4217 currency code (e.g., 'USD', 'MNT')"
        })
    # LLM comments: check for unit inconsistency in comments
    try:
        bond_financials_hist = data.get("bond_financials_historical", {})
        logger.info(f"bond_financials_historical type: {type(bond_financials_hist)}")
        comments_hist = bond_financials_hist.get("comments", "")
    except Exception as e:
        logger.error(f"Error accessing comments in bond_financials_historical: {e}")
        comments_hist = ""
    if any(kw in comments_hist.lower() for kw in ["inconsistent units", "conversion discrepancies", "conflicting units"]):
        problems.append({
            "type": "UnitInconsistency",
            "message": "LLM comments indicate unit inconsistency in bond_financials_historical.",
            "suggested_action": "Review for unit conversion errors."
        })
    # 2. Order of Magnitude Plausibility
    try:
        hist = data.get("bond_financials_historical", {}).get("historical_financial_statements", [])
        # Filter out any non-dict items (like truncation markers)
        hist = [item for item in hist if isinstance(item, dict)]
        logger.info(f"historical_financial_statements type: {type(hist)}, length: {len(hist) if isinstance(hist, list) else 'N/A'}")
    except Exception as e:
        logger.error(f"Error accessing historical_financial_statements: {e}")
        hist = []
    for metric in ["total_assets", "total_revenue", "net_profit"]:
        prev = None
        for period in hist:
            val = period.get("financials", {}).get(metric)
            if isinstance(val, (int, float)) and prev is not None and isinstance(prev, (int, float)):
                if prev != 0 and (val/prev > 1000 or val/prev < 0.001):
                    if not ("explained" in period.get("comments", "").lower()):
                        problems.append({
                            "type": "MagnitudeAnomaly",
                            "message": f"{metric} changes by >1000x between periods.",
                            "suggested_action": "Review for unit/decimal error."
                        })
                prev = val
            elif isinstance(val, (int, float)):
                prev = val
    # 3. Balance Sheet: Assets = Liabilities + Equity
    # Balance sheet equation tolerance (1% is more appropriate for financial data with potential rounding)
    tolerance = 0.01
    for period in hist:
        fin = period.get("financials", {})
        ta = fin.get("total_assets")
        tl = fin.get("total_liabilities")
        te = fin.get("total_equity")
        if all(isinstance(x, (int, float)) for x in [ta, tl, te]):
            if abs(ta - (tl + te)) > tolerance * abs(ta):
                problems.append({
                    "type": "BalanceSheetEquationMismatch",
                    "message": "total_assets != total_liabilities + total_equity (beyond tolerance)",
                    "suggested_action": "Re-extract balance sheet for this period."
                })
    # 4. All Numerical Fields are Numbers (from schema)
    def check_numbers(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    check_numbers(v, path + "." + k if path else k)
                else:
                    if schema and k in schema.get("properties", {}) and schema["properties"][k].get("type") == "number":
                        if not isinstance(v, (int, float)):
                            problems.append({
                                "type": "IncorrectDataType",
                                "message": f"{path+'.'+k}: Not a number.",
                                "suggested_action": "Re-extract field as number."
                            })
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                check_numbers(item, f"{path}[{i}]")
    check_numbers(data)
    # 5. Date Format Validity & Order
    import re
    date_re = re.compile(r"^\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2})?$")
    issue_date = data.get("bond_metadata", {}).get("issue_date")
    maturity_date = data.get("bond_metadata", {}).get("maturity_date")
    if issue_date and not date_re.match(str(issue_date)):
        warnings.append({"type": "InvalidDateFormat", "message": "issue_date invalid format.", "suggested_action": "Review date extraction."})
    if maturity_date and not date_re.match(str(maturity_date)):
        warnings.append({"type": "InvalidDateFormat", "message": "maturity_date invalid format.", "suggested_action": "Review date extraction."})
    if issue_date and maturity_date and issue_date > maturity_date:
        warnings.append({"type": "ChronologicalError", "message": "issue_date after maturity_date.", "suggested_action": "Review date order."})
    # reporting_period_end_date order
    prev_date = None
    for period in hist:
        d = period.get("reporting_period_end_date")
        if d and not date_re.match(str(d)):
            warnings.append({"type": "InvalidDateFormat", "message": "reporting_period_end_date invalid format.", "suggested_action": "Review date extraction."})
        if d and prev_date and d < prev_date:
            warnings.append({"type": "ChronologicalError", "message": "reporting_period_end_date not increasing.", "suggested_action": "Review period order."})
        if d:
            prev_date = d
    # 6. Non-Negative Core Values
    nonneg_fields = ["principal_amount_issued", "total_assets", "total_revenue", "net_interest_income", "total_equity"]
    for field in nonneg_fields:
        val = data.get("bond_metadata", {}).get(field) or data.get("bond_financials_historical", {}).get(field)
        if val is not None and isinstance(val, (int, float)) and val < -tolerance:
            problems.append({"type": "NegativeValueAnomaly", "message": f"{field} is negative.", "suggested_action": "Review for sign error."})
    # 7. Coupon Rate Plausibility
    cr = data.get("bond_metadata", {}).get("coupon_rate")
    if cr is not None and (cr < 0.005 or cr > 0.5):
        problems.append({"type": "ImplausibleCouponRate", "message": "coupon_rate out of range (0.5%-50%).", "suggested_action": "Review coupon rate extraction."})
        
    # 8. Unit Conversion Notes Presence Check
    financial_sections = [
        "bond_financials_historical", 
        "bond_financials_projections", 
        "bond_metadata"
    ]
    # Unit conversion notes check removed as per user request
            
    # 9. Order of Magnitude Consistency Across Sections
    # Compare principal amount between bond_metadata and financials
    principal_metadata = data.get("bond_metadata", {}).get("principal_amount_issued")
    
    # Find bond payable or similar in most recent historical period
    bond_payable = None
    hist_statements = data.get("bond_financials_historical", {}).get("historical_financial_statements", [])
    # Filter out any non-dict items (like truncation markers)
    hist_statements = [item for item in hist_statements if isinstance(item, dict)]
    if hist_statements:
        # Sort by date to get most recent
        sorted_statements = sorted(hist_statements, 
                                  key=lambda x: x.get("reporting_period_end_date", ""), 
                                  reverse=True)
        if sorted_statements:
            financials = sorted_statements[0].get("financials", {})
            # Try different possible field names for bond payable
            for field in ["bond_payable", "bonds_payable", "debt_securities_issued"]:
                if field in financials:
                    bond_payable = financials[field]
                    break
    
    # Compare if both values exist
    if principal_metadata is not None and bond_payable is not None:
        # Allow some difference due to partial repayment, but check order of magnitude
        if principal_metadata != 0 and bond_payable != 0:
            ratio = max(principal_metadata, bond_payable) / min(principal_metadata, bond_payable)
            if ratio > 100:  # Order of magnitude difference
                problems.append({
                    "section": "bond_metadata, bond_financials_historical",
                    "field": "principal_amount_issued, bond_payable",
                    "period": None,
                    "problem_type": "CrossSectionUnitInconsistency",
                    "description": f"Large discrepancy between principal_amount in metadata ({principal_metadata}) and bond_payable in financials ({bond_payable}).",
                    "value_found": f"Ratio: {ratio:.2f}",
                    "expected_value": "Consistent units across sections"
                })
    
    # 10. Financial Ratio Plausibility Checks
    for period in hist_statements:
        fin = period.get("financials", {})
        period_date = period.get("reporting_period_end_date", "Unknown")
        
    # 11. Unit Keyword Pattern Matching
    import re
    unit_keywords = [
        "million", "billion", "trillion", "mn", "bn", "tn",
        "сая", "тэрбум", "төгрөг", "доллар",  # Mongolian terms
        r"$", "USD", "MNT", "₮"  # Currency symbols
    ]
    
    # Create regex pattern for unit keywords
    pattern = r'\b(' + '|'.join(unit_keywords) + r')\b'
    unit_regex = re.compile(pattern, re.IGNORECASE)
    
    # Fields that should not contain unit keywords (except in comments/notes fields)
    financial_fields = [
        "total_assets", "total_liabilities", "total_equity", "total_revenue", 
        "net_profit", "principal_amount_issued", "bond_payable", "current_assets",
        "current_liabilities", "cash_and_cash_equivalents"
    ]
    
    # Check for unit keywords in string representations of financial values
    for section_name, section_data in data.items():
        if not isinstance(section_data, dict):
            continue
            
        # Skip checking comment/notes fields
        skip_fields = ["comments", "notes", "unit_conversion_notes", "description"]
        
        # Recursive function to check all nested fields
        def check_unit_keywords(obj, path):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in skip_fields:
                        continue
                    new_path = f"{path}.{k}" if path else k
                    check_unit_keywords(v, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_unit_keywords(item, f"{path}[{i}]")
            elif isinstance(obj, str) and any(f in path.lower() for f in financial_fields):
                # Check if string contains unit keywords
                matches = unit_regex.findall(obj)
                if matches:
                    problems.append({
                        "section": section_name,
                        "field": path,
                        "period": None,
                        "problem_type": "UnitKeywordInValue",
                        "description": f"Found unit keyword(s) {matches} in financial value field {path}.",
                        "value_found": obj,
                        "expected_value": "Pure numeric value without unit text"
                    })
        
        check_unit_keywords(section_data, section_name)
        
    # 12. Trailing Zeros Analysis
    # This helps detect scale issues (e.g., values in millions vs. billions)
    trailing_zero_counts = {}
    section_zero_patterns = {}
    
    def count_trailing_zeros(num):
        """Count trailing zeros in a number."""
        if not isinstance(num, (int, float)):
            return 0
        if num == 0:
            return 0
            
        # Convert to string and remove decimal point
        str_num = str(abs(num)).replace(".", "")
        # Count trailing zeros
        count = 0
        for c in reversed(str_num):
            if c == "0":
                count += 1
            else:
                break
        return count
    
    def check_trailing_zeros(obj, section):
        """Recursively check for trailing zeros in numeric values."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (int, float)) and abs(v) >= 1000:  # Only check significant values
                    field_path = f"{section}.{k}"
                    zeros = count_trailing_zeros(v)
                    if zeros >= 3:  # Only record if at least 3 trailing zeros
                        if field_path not in trailing_zero_counts:
                            trailing_zero_counts[field_path] = []
                        trailing_zero_counts[field_path].append(zeros)
                        
                        # Track zeros by section for cross-section comparison
                        if section not in section_zero_patterns:
                            section_zero_patterns[section] = []
                        section_zero_patterns[section].append(zeros)
                elif isinstance(v, (dict, list)):
                    check_trailing_zeros(v, f"{section}.{k}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                check_trailing_zeros(item, section)
    
    # Check each major section
    for section_name, section_data in data.items():
        if isinstance(section_data, dict):
            check_trailing_zeros(section_data, section_name)
    
    # Analyze results for suspicious patterns within fields
    for field, zeros_list in trailing_zero_counts.items():
        if len(zeros_list) >= 3:  # Need multiple values to detect a pattern
            # If all values have exactly 6 trailing zeros, likely in millions
            if all(z == 6 for z in zeros_list):
                problems.append({
                    "section": field.split(".")[0],
                    "field": field,
                    "period": None,
                    "problem_type": "SuspiciousTrailingZeros",
                    "description": f"All values in {field} have exactly 6 trailing zeros, suggesting values may be in millions rather than base units.",
                    "value_found": f"Consistent {zeros_list[0]} trailing zeros",
                    "expected_value": "Varied trailing zeros pattern for properly normalized values"
                })
            # If all values have exactly 9 trailing zeros, likely in billions
            elif all(z == 9 for z in zeros_list):
                problems.append({
                    "section": field.split(".")[0],
                    "field": field,
                    "period": None,
                    "problem_type": "SuspiciousTrailingZeros",
                    "description": f"All values in {field} have exactly 9 trailing zeros, suggesting values may be in billions rather than base units.",
                    "value_found": f"Consistent {zeros_list[0]} trailing zeros",
                    "expected_value": "Varied trailing zeros pattern for properly normalized values"
                })
    
    # Compare trailing zeros patterns across sections to detect unit inconsistencies
    hist_zeros = section_zero_patterns.get("bond_financials_historical", [])
    proj_zeros = section_zero_patterns.get("bond_financials_projections", [])
    
    if hist_zeros and proj_zeros:
        # Get the most common trailing zero count in each section
        from collections import Counter
        hist_most_common = Counter(hist_zeros).most_common(1)[0][0] if hist_zeros else 0
        proj_most_common = Counter(proj_zeros).most_common(1)[0][0] if proj_zeros else 0
        
        # If there's a significant difference in trailing zeros between sections (3 = 1000x difference)
        if abs(hist_most_common - proj_most_common) >= 3:
            problems.append({
                "section": "bond_financials_historical, bond_financials_projections",
                "field": "multiple fields",
                "period": None,
                "problem_type": "CrossSectionUnitInconsistency",
                "description": f"Unit inconsistency detected: historical data typically has {hist_most_common} trailing zeros while projections have {proj_most_common} trailing zeros.",
                "value_found": f"Difference of {abs(hist_most_common - proj_most_common)} orders of magnitude",
                "expected_value": "Consistent unit scale across all sections"
            })
                
    # 13. Cross-Statement Consistency Checks
    # Check for consistency between financial statements for the same period
    for period in hist_statements:
        period_date = period.get("reporting_period_end_date", "Unknown")
        fin = period.get("financials", {})
        
        # Check net profit consistency between income statement and balance sheet
        net_profit = fin.get("net_profit")
        retained_earnings_change = None
        
        # Find previous period to calculate retained earnings change
        prev_period = None
        for p in hist_statements:
            p_date = p.get("reporting_period_end_date", "")
            if p_date < period_date:
                if prev_period is None or p_date > prev_period.get("reporting_period_end_date", ""):
                    prev_period = p
        
        if prev_period:
            current_retained = fin.get("retained_earnings")
            prev_retained = prev_period.get("financials", {}).get("retained_earnings")
            
            if isinstance(current_retained, (int, float)) and isinstance(prev_retained, (int, float)):
                # Account for dividends as a simple approximation
                retained_earnings_change = current_retained - prev_retained
        
        # Compare net profit with change in retained earnings
        if isinstance(net_profit, (int, float)) and isinstance(retained_earnings_change, (int, float)) and abs(net_profit) > 0:
            # Allow for some difference due to dividends, but check order of magnitude
            ratio = max(abs(net_profit), abs(retained_earnings_change)) / min(abs(net_profit), abs(retained_earnings_change))
            if ratio > 10 and abs(net_profit - retained_earnings_change) > 1000000:  # Significant difference
                problems.append({
                    "section": "bond_financials_historical",
                    "field": "net_profit, retained_earnings",
                    "period": period_date,
                    "problem_type": "CrossStatementInconsistency",
                    "description": f"Large discrepancy between net_profit ({net_profit}) and change in retained_earnings ({retained_earnings_change}) for period {period_date}.",
                    "value_found": f"Ratio: {ratio:.2f}",
                    "expected_value": "Consistent values across financial statements"
                })
        
        # Debt-to-Equity ratio check
        total_liabilities = fin.get("total_liabilities")
        total_equity = fin.get("total_equity")
        if isinstance(total_liabilities, (int, float)) and isinstance(total_equity, (int, float)) and total_equity != 0:
            debt_equity_ratio = total_liabilities / total_equity
            # For financial institutions, D/E can be higher, but still has reasonable limits
            if debt_equity_ratio > 50 or debt_equity_ratio < 0.01:
                problems.append({
                    "section": "bond_financials_historical",
                    "field": "total_liabilities, total_equity",
                    "period": period_date,
                    "problem_type": "ImplausibleFinancialRatio",
                    "description": f"Debt-to-Equity ratio of {debt_equity_ratio:.2f} is outside plausible range for period {period_date}.",
                    "value_found": f"{debt_equity_ratio:.2f}",
                    "expected_value": "0.01-50 (typical range for financial institutions)"
                })
        
        # Current Ratio check (if applicable)
        current_assets = fin.get("current_assets")
        current_liabilities = fin.get("current_liabilities")
        if isinstance(current_assets, (int, float)) and isinstance(current_liabilities, (int, float)) and current_liabilities != 0:
            current_ratio = current_assets / current_liabilities
            if current_ratio > 20 or current_ratio < 0.1:
                problems.append({
                    "section": "bond_financials_historical",
                    "field": "current_assets, current_liabilities",
                    "period": period_date,
                    "problem_type": "ImplausibleFinancialRatio",
                    "description": f"Current ratio of {current_ratio:.2f} is outside plausible range for period {period_date}.",
                    "value_found": f"{current_ratio:.2f}",
                    "expected_value": "0.1-20 (typical range)"
                })
    
    # 14. Historical vs. Projected Unit Consistency
    hist_data = data.get("bond_financials_historical", {})
    proj_data = data.get("bond_financials_projections", {})
    # (rest of function continues...)
    # ...
    # Get the most recent historical period
    latest_hist_period = None
    latest_hist_date = ""
    hist_statements = hist_data.get("historical_financial_statements", [])
    # Filter out any non-dict items (like truncation markers)
    hist_statements = [item for item in hist_statements if isinstance(item, dict)]
    
    if hist_statements:
        for period in hist_statements:
            period_date = period.get("reporting_period_end_date", "")
            if period_date > latest_hist_date:
                latest_hist_date = period_date
                latest_hist_period = period
    
    # Handle both traditional projected_financial_statements and issuer_projections.projected_metrics
    # First check for projected_financial_statements
    proj_statements = proj_data.get("projected_financial_statements", [])
    # Filter out any non-dict items (like truncation markers)
    proj_statements = [item for item in proj_statements if isinstance(item, dict)]
    issuer_proj_metrics = proj_data.get("issuer_projections", {}).get("projected_metrics", {})
    
    # Compare key metrics between latest historical and projections
    if latest_hist_period:
        hist_fin = latest_hist_period.get("financials", {})
        
        # Traditional projected financial statements comparison
        if proj_statements:
            earliest_proj_period = None
            earliest_proj_date = "9999-12-31"
            
            for period in proj_statements:
                period_date = period.get("reporting_period_end_date", "")
                if period_date < earliest_proj_date and period_date > latest_hist_date:
                    earliest_proj_date = period_date
                    earliest_proj_period = period
            
            if earliest_proj_period:
                proj_fin = earliest_proj_period.get("financials", {})
                
                # Key metrics to compare
                key_metrics = [
                    "total_assets", "total_liabilities", "total_equity", 
                    "total_revenue", "net_profit", "net_interest_income"
                ]
                
                for metric in key_metrics:
                    hist_val = hist_fin.get(metric)
                    proj_val = proj_fin.get(metric)
                    
                    if isinstance(hist_val, (int, float)) and isinstance(proj_val, (int, float)) and hist_val != 0 and proj_val != 0:
                        # Calculate growth rate between periods
                        growth_rate = (proj_val / hist_val) - 1
                        
                        # Flag suspicious jumps that might indicate unit inconsistency
                        # Allow for reasonable growth but flag extreme changes
                        if abs(growth_rate) > 5:  # 500% growth or -80% decline is suspicious
                            problems.append({
                                "section": "bond_financials_historical, bond_financials_projections",
                                "field": metric,
                                "period": f"{latest_hist_date} to {earliest_proj_date}",
                                "problem_type": "HistoricalProjectedUnitInconsistency",
                                "description": f"Suspicious jump in {metric} from {hist_val} (historical) to {proj_val} (projected), growth rate: {growth_rate:.2f}.",
                                "value_found": f"Growth rate: {growth_rate:.2f}",
                                "expected_value": "Reasonable growth rate between -0.8 and 5.0"
                            })
        
        # Handle issuer_projections.projected_metrics structure
        if issuer_proj_metrics:
            # Map historical metrics to their projected counterparts
            metric_mappings = {
                "interest_income": [k for k in issuer_proj_metrics.keys() if "projected_interest_income" in k],
                "net_interest_income": [k for k in issuer_proj_metrics.keys() if "projected_net_interest_income" in k],
                "operating_income": [k for k in issuer_proj_metrics.keys() if "projected_operating_income" in k],
                "net_profit": [k for k in issuer_proj_metrics.keys() if "projected_net_profit" in k]
            }
            
            # Compare each historical metric with its projected counterparts
            for hist_metric, proj_metric_keys in metric_mappings.items():
                hist_val = hist_fin.get(hist_metric)
                
                if isinstance(hist_val, (int, float)) and hist_val != 0 and proj_metric_keys:
                    # Get the average of projected values for comparison
                    proj_vals = [issuer_proj_metrics.get(k) for k in proj_metric_keys if isinstance(issuer_proj_metrics.get(k), (int, float))]
                    
                    if proj_vals:
                        avg_proj_val = sum(proj_vals) / len(proj_vals)
                        
                        # Calculate growth rate
                        growth_rate = (avg_proj_val / hist_val) - 1
                        
                        # Flag suspicious jumps that might indicate unit inconsistency
                        if abs(growth_rate) > 5:  # 500% growth or -80% decline is suspicious
                            problems.append({
                                "section": "bond_financials_historical, bond_financials_projections.issuer_projections",
                                "field": f"{hist_metric}, {proj_metric_keys[0]} (and similar)",
                                "period": f"From {latest_hist_date} to projection period",
                                "problem_type": "HistoricalProjectedUnitInconsistency",
                                "description": f"Suspicious jump in {hist_metric} from {hist_val} (historical) to {avg_proj_val:.2f} (avg projected), growth rate: {growth_rate:.2f}.",
                                "value_found": f"Growth rate: {growth_rate:.2f}",
                                "expected_value": "Reasonable growth rate between -0.8 and 5.0"
                            })
    
    # 15. Principal Amount Reasonableness Check
    principal = data.get("bond_metadata", {}).get("principal_amount_issued")
    total_assets = None
    total_revenue = None
    
    # Get the most recent total assets and revenue
    if hist_statements:
        for period in sorted(hist_statements, key=lambda x: x.get("reporting_period_end_date", ""), reverse=True):
            fin = period.get("financials", {})
            if "total_assets" in fin and total_assets is None:
                total_assets = fin["total_assets"]
            if "total_revenue" in fin and total_revenue is None:
                total_revenue = fin["total_revenue"]
            if total_assets is not None and total_revenue is not None:
                break
    
    # Check if principal amount is reasonable compared to total assets
    if isinstance(principal, (int, float)) and isinstance(total_assets, (int, float)) and total_assets > 0:
        principal_to_assets_ratio = principal / total_assets
        
        # Flag if bond is implausibly large compared to issuer size
        if principal_to_assets_ratio > 1.0:  # Bond larger than total assets is suspicious
            problems.append({
                "section": "bond_metadata, bond_financials_historical",
                "field": "principal_amount_issued, total_assets",
                "period": None,
                "problem_type": "ImplausiblePrincipalAmount",
                "description": f"Bond principal ({principal}) exceeds issuer's total assets ({total_assets}), ratio: {principal_to_assets_ratio:.2f}.",
                "value_found": f"Principal/Assets ratio: {principal_to_assets_ratio:.2f}",
                "expected_value": "Ratio < 1.0 for most issuers"
            })
        # Flag if bond is implausibly small compared to issuer size
        elif principal_to_assets_ratio < 0.0001:  # Bond less than 0.01% of assets is suspicious
            problems.append({
                "section": "bond_metadata, bond_financials_historical",
                "field": "principal_amount_issued, total_assets",
                "period": None,
                "problem_type": "ImplausiblePrincipalAmount",
                "description": f"Bond principal ({principal}) is unusually small compared to issuer's total assets ({total_assets}), ratio: {principal_to_assets_ratio:.6f}.",
                "value_found": f"Principal/Assets ratio: {principal_to_assets_ratio:.6f}",
                "expected_value": "Ratio > 0.0001 for most issuers"
            })
    
    # 16. Compare bond principal with issuer projections
    # This check specifically targets the common issue of unit mismatch between bond metadata and projections
    issuer_proj_metrics = proj_data.get("issuer_projections", {}).get("projected_metrics", {})
    if isinstance(principal, (int, float)) and principal > 0 and issuer_proj_metrics:
        # Get average projected values
        proj_values = [v for k, v in issuer_proj_metrics.items() if isinstance(v, (int, float))]
        if proj_values:
            avg_proj_value = sum(proj_values) / len(proj_values)
            principal_to_proj_ratio = principal / avg_proj_value
            
            # If bond principal is 3+ orders of magnitude smaller than average projection values
            # This suggests a unit mismatch (e.g., principal in millions, projections in billions)
            if principal_to_proj_ratio < 0.001:
                problems.append({
                    "section": "bond_metadata, bond_financials_projections",
                    "field": "principal_amount_issued, issuer_projections.projected_metrics",
                    "period": None,
                    "problem_type": "UnitMismatchPrincipalVsProjections",
                    "description": f"Possible unit mismatch: Bond principal ({principal}) is 1000+ times smaller than average projection values ({avg_proj_value:.2f}).",
                    "value_found": f"Principal/Avg Projection ratio: {principal_to_proj_ratio:.6f}",
                    "expected_value": "Ratio > 0.001 for consistent units"
                })
            # If bond principal is 3+ orders of magnitude larger than average projection values
            elif principal_to_proj_ratio > 1000:
                problems.append({
                    "section": "bond_metadata, bond_financials_projections",
                    "field": "principal_amount_issued, issuer_projections.projected_metrics",
                    "period": None,
                    "problem_type": "UnitMismatchPrincipalVsProjections",
                    "description": f"Possible unit mismatch: Bond principal ({principal}) is 1000+ times larger than average projection values ({avg_proj_value:.2f}).",
                    "value_found": f"Principal/Avg Projection ratio: {principal_to_proj_ratio:.6f}",
                    "expected_value": "Ratio < 1000 for consistent units"
                })
    
    results["problems"] = problems
    results["warnings"] = warnings
    results["passed"] = len(problems) == 0  # Only critical issues affect pass/fail status
    # Set final pass/fail status based on problems list
    results["passed"] = len(problems) == 0
    return results

# --- LLM Assessment ---
async def llm_assess_quality(data: dict, schema: dict, deterministic_checks: dict) -> dict:
    """
    Compose a prompt for the LLM that:
    - Includes the QC schema (with comments)
    - Includes deterministic QC issues (in schema format)
    - Includes the extracted data (summarized if too large)
    - Instructs the LLM: output ONLY a JSON object matching the QC schema exactly, and nothing else.
    """
    
    # Type checking and validation
    if not isinstance(data, dict):
        logger.error(f"Expected data to be dict, got {type(data)}")
        return {"qc_status": "Fail", "deterministic_qc_issues": [], "value_corrections": {}, "reasoning": f"Data type error: expected dict, got {type(data)}"}
    
    if not isinstance(deterministic_checks, dict):
        logger.error(f"Expected deterministic_checks to be dict, got {type(deterministic_checks)}")
        return {"qc_status": "Fail", "deterministic_qc_issues": [], "value_corrections": {}, "reasoning": f"Deterministic checks type error: expected dict, got {type(deterministic_checks)}"}

    # Load qc_schema_with_corrections.json for the LLM
    qc_schema_path = Path(__file__).parent / "qc_schema_with_corrections.json"
    with open(qc_schema_path, "r", encoding="utf-8") as f:
        qc_schema = json.load(f)
    # Summarize data if too large - use a more granular approach to prevent excessive truncation
    # CRITICAL: Use deep copy to avoid modifying the original data structure
    data_for_llm = copy.deepcopy(data)
    data_str = json.dumps(data_for_llm)
    if len(data_str) > 12000:
        # More intelligent truncation that preserves structure
        for k, v in data_for_llm.items():
            if isinstance(v, dict):
                # For nested dictionaries, keep important fields and summarize the rest
                for sub_k, sub_v in list(v.items()):
                    if isinstance(sub_v, (list, dict)) and len(json.dumps(sub_v)) > 1000:
                        if isinstance(sub_v, list) and len(sub_v) > 3:
                            # Keep first 3 items for lists
                            # Note: This modifies data_for_llm only (deep copy), not the original data
                            v[sub_k] = sub_v[:3] + [f"<{len(sub_v)-3} more items truncated>"]
                        elif isinstance(sub_v, dict):
                            # Keep only essential fields for dictionaries
                            # Note: This modifies data_for_llm only (deep copy), not the original data
                            essential_keys = ["currency", "principal_amount_issued", "denomination", "unit_conversion_notes"]
                            v[sub_k] = {sk: sub_v.get(sk) for sk in essential_keys if sk in sub_v}
                            v[sub_k]["<truncated>"] = f"{len(sub_v) - len(v[sub_k])} fields truncated"
            elif isinstance(v, list) and len(v) > 5:
                # For top-level lists, keep first 5 items
                # Note: This modifies data_for_llm only (deep copy), not the original data
                data_for_llm[k] = v[:5] + [f"<{len(v)-5} more items truncated>"]
    # Filter out warnings and date-related issues from problems for LLM assessment
    critical_issues = []
    date_related_problem_types = ["InvalidDateFormat", "ChronologicalError"]
    
    for issue in deterministic_checks.get('problems', []):
        problem_type = issue.get("problem_type", "")
        if problem_type not in date_related_problem_types:
            critical_issues.append(issue)
    
    # Compose prompt
    prompt = f"""You are a financial data quality control expert. You will analyze bond data and provide a quality assessment.

Your output must be a JSON object matching this schema:
```json
{json.dumps(qc_schema, indent=2)}
```

Deterministic quality checks have already been run with these results:
```json
{json.dumps(critical_issues, indent=2)}
```

Here is a summary of the data to assess:
```json
{json.dumps(data_for_llm, indent=2)}
```

Your task is to analyze the data and deterministic check results, then output a JSON object that:
1. Sets qc_status to "Pass" if no critical issues are found, or "Fail" if any critical issues are found
2. Includes the deterministic_qc_issues array from the deterministic checks
3. Provides a value_corrections object with exact JSON paths to values that need correction
4. For complex issues that cannot be fixed with simple value corrections, provide repair_instructions
5. IMPORTANT: Add a new field called "reasoning" that explains your analysis and why you chose specific corrections

For the value_corrections object:
- Each key should be a dot-notation path to the value that needs correction (e.g., "bond_metadata.principal_amount_issued")
- Each value should be an expression that can be evaluated with the current value (e.g., "VALUE_FOUND * 1000000")
- IMPORTANT: principal_amount_issued from bond_metadata is in BASE UNITS (not millions or billions)
- All financial values should be normalized to BASE UNITS to match principal_amount_issued
- If a value appears too small compared to principal_amount_issued, it likely needs to be MULTIPLIED by 1,000,000 or 1,000,000,000
- If a value appears too large compared to principal_amount_issued, it might need to be DIVIDED
- You MUST address ALL unit inconsistencies and magnitude anomalies, not just one example
- Focus on unit normalization issues (millions vs. billions vs. base units)
- Ensure the corrected values are of the appropriate type (number, string, boolean, etc.)
- For array elements, use index notation (e.g., "bond_financials_historical.historical_financial_statements[0].financials.total_assets")

For the repair_instructions array:
- Include section_to_re_extract (e.g., "bond_financials_historical")
- Include llm_repair_prompt_directive that follows this DETAILED structure:
  1. Start with "This is a repair extraction. A previous extraction attempt had quality issues that need to be addressed."
  2. Provide a COMPREHENSIVE explanation of what was extracted previously and the specific issues identified
  3. Explain in DETAIL why these issues matter (e.g., impact on financial analysis, consistency with other data, implications for projections)
  4. Describe the RELATIONSHIPS between the problematic data and other sections (e.g., how unit inconsistencies affect comparability with bond principal)
  5. Provide CONTEXTUAL guidance on what to pay special attention to, including:
     - Document structure and how information is presented
     - Typical financial reporting conventions for this type of issuer
     - Unit conversion considerations specific to this data
     - Relationships between financial statement items that should be maintained
  6. STRONGLY encourage a fresh, complete extraction of the entire section rather than targeted fixes
  7. Include SPECIFIC examples of the problematic data to illustrate the issues (e.g., "total assets reported as X trillion when it should be X billion")
- Your repair_prompt_directive should be at least 150 words to provide sufficient context
- Focus on providing rich context rather than specific correction instructions
- Emphasize that the LLM should use its own judgment to extract all data correctly
- Include information about the document structure, data relationships, and expected formats
- IGNORE any date format or chronological issues as these are handled deterministically

IMPORTANT: DO NOT include any repair instructions or value corrections for:
- issue_date format issues
- reporting_period_end_date chronological issues
- Any other date-related problems
These are handled by deterministic checks and do not need LLM intervention.

Output ONLY the JSON object, no additional text.
"""
    # Use Gemini LLM (hardcoded model), API key from config.py
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        import config
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-04-17",  # Hardcoded model
            temperature=0.0,
            top_p=getattr(config, "LLM_TOP_P", 1.0),
            google_api_key=getattr(config, "GOOGLE_API_KEY", None),
        )
        result = await llm.ainvoke(prompt)
        logger.info(f"Raw LLM response: {result.content}")
        content_str = result.content.strip()
        # Remove markdown code block if present
        if content_str.startswith('```'):
            content_str = content_str.split('\n', 1)[-1]  # Remove first line (```json or ```)
            if content_str.endswith('```'):
                content_str = content_str[:-3].strip()
        try:
            qc_json = json.loads(content_str)
            # Ensure required fields exist
            if "value_corrections" not in qc_json:
                qc_json["value_corrections"] = {}
            if "reasoning" not in qc_json:
                qc_json["reasoning"] = "No reasoning provided"
        except Exception as e:
            logger.error(f"JSON parsing failed: {e}")
            qc_json = {"qc_status": "Fail", "deterministic_qc_issues": deterministic_checks.get('problems', []), "value_corrections": {}}
        return qc_json
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return {
            "qc_status": "Fail",
            "deterministic_qc_issues": deterministic_checks.get('problems', []),
            "value_corrections": {},
            "reasoning": f"LLM assessment failed: {str(e)}"
        }


# Import the value corrector module
from value_corrector import apply_corrections, get_value_at_path

@mcp_app.tool()
async def quality_check_final_data(input_data: QualityCheckInput) -> QualityCheckResult:
    # Load data and schema
    try:
        logger.info(f"Loading data from: {input_data.finalized_data_path}")
        with open(input_data.finalized_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded data type: {type(data)}")
        if not isinstance(data, dict):
            raise ValueError(f"Expected data to be a dict, got {type(data)}")
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to load data: {str(e)}")
        return QualityCheckResult(
            status="error",
            message=f"Failed to load finalized data JSON: {str(e)}",
            deterministic_checks={"passed": False, "problems": [], "warnings": []},
            llm_assessment=None
        )
    except Exception as e:
        logger.error(f"Unexpected error loading data: {str(e)}")
        return QualityCheckResult(
            status="error",
            message=f"Unexpected error loading data: {str(e)}",
            deterministic_checks={"passed": False, "problems": [], "warnings": []},
            llm_assessment=None
        )

    # Load schema if provided
    schema = None
    if input_data.schema_path:
        try:
            with open(input_data.schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
            if not isinstance(schema, dict):
                raise ValueError(f"Expected schema to be a dict, got {type(schema)}")
        except (json.JSONDecodeError, ValueError) as e:
            return QualityCheckResult(
                status="error",
                message=f"Failed to load schema JSON: {str(e)}",
                deterministic_checks={"passed": False, "problems": [], "warnings": []},
                llm_assessment=None
            )
    
    # Run deterministic checks
    try:
        logger.info("Running deterministic checks")
        det_checks = run_deterministic_checks(data, schema or {})
        logger.info(f"Deterministic checks completed, type: {type(det_checks)}")
    except Exception as e:
        logger.error(f"Error in deterministic checks: {str(e)}")
        return QualityCheckResult(
            status="error",
            message=f"Error in deterministic checks: {str(e)}",
            deterministic_checks={"passed": False, "problems": [], "warnings": []},
            llm_assessment=None
        )
    
    # Get LLM assessment with value corrections
    try:
        logger.info("Starting LLM assessment")
        llm_result = await llm_assess_quality(data, schema or {}, det_checks)
        logger.info(f"LLM assessment completed, type: {type(llm_result)}")
        if not isinstance(llm_result, dict):
            logger.error(f"LLM assessment returned non-dict: {llm_result}")
            return QualityCheckResult(
                status="error",
                message=f"LLM assessment returned non-dict result: {llm_result}",
                deterministic_checks=det_checks,
                llm_assessment=None
            )
    except Exception as e:
        logger.error(f"LLM assessment failed: {str(e)}")
        return QualityCheckResult(
            status="error",
            message=f"LLM assessment failed: {str(e)}",
            deterministic_checks=det_checks,
            llm_assessment=None
        )
    
    # Apply corrections if any
    corrections_applied = False
    if isinstance(llm_result, dict) and llm_result.get("qc_status") == "Fail" and llm_result.get("value_corrections"):
        value_corrections = llm_result.get("value_corrections", {})
        if value_corrections:
            logger.info(f"Applying {len(value_corrections)} value corrections")
            
            # Evaluate expressions in value_corrections
            evaluated_corrections = {}
            for path, expr in value_corrections.items():
                try:
                    value_found = get_value_at_path(data, path)
                    if value_found is not None:
                        # Use a safe way to evaluate the expression
                        # For now, assuming simple arithmetic, but eval() can be dangerous.
                        # In a production system, consider a more robust expression evaluator.
                        evaluated_value = eval(expr, {"VALUE_FOUND": value_found})
                        evaluated_corrections[path] = evaluated_value
                    else:
                        logger.warning(f"Could not find value at path '{path}' for evaluation.")
                except Exception as e:
                    logger.error(f"Error evaluating correction expression for path '{path}': {e}")

            # Apply corrections to the data
            corrected_data = apply_corrections(data, evaluated_corrections)
            
            # Save the corrected data back to the file
            with open(input_data.finalized_data_path, "w", encoding="utf-8") as f:
                json.dump(corrected_data, f, indent=2, ensure_ascii=False)
            
            # Save a backup of the original data
            backup_path = input_data.finalized_data_path + ".backup"
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Applied corrections and saved to {input_data.finalized_data_path}")
            logger.info(f"Original data backed up to {backup_path}")
            
            # Update the data for the result
            data = corrected_data
            corrections_applied = True
            
            # Re-run deterministic checks on corrected data
            det_checks = run_deterministic_checks(data, schema or {})
    
    # Create a summary of corrections if applied
    corrections_summary = ""
    if corrections_applied:
        corrections_summary = "\nAUTOMATIC CORRECTIONS APPLIED:\n"
        for path, value in llm_result.get("value_corrections", {}).items():
            corrections_summary += f"- {path}: {value}\n"
    
    # Create a simplified result for the agent - use LLM's assessment as the final authority
    simplified_result = {
        "qc_status": llm_result.get("qc_status", "Fail"),  # Trust LLM's assessment
        "message": f"Quality check complete.{corrections_summary}",
        "corrections_applied": llm_result.get("value_corrections", {}) if corrections_applied else {},
        "fix_successful": corrections_applied and det_checks["passed"],
        "reasoning": llm_result.get("reasoning", "No reasoning provided")
    }
    
    # Add warnings if present (these don't affect pass/fail status)
    if det_checks.get("warnings"):
        simplified_result["warnings"] = det_checks["warnings"]
    
    # Add repair instructions for complex issues if present
    if llm_result.get("repair_instructions") and not det_checks["passed"]:
        simplified_result["repair_instructions"] = llm_result.get("repair_instructions")
    
    return QualityCheckResult(
        status="success" if det_checks["passed"] else "failed",
        message=f"Quality check complete.{corrections_summary}",
        deterministic_checks=det_checks,
        llm_assessment=simplified_result
    )

async def run_stdio_server():
    await mcp_app.run_stdio_async()

def main():
    import asyncio
    asyncio.run(run_stdio_server())

if __name__ == "__main__":
    main()
