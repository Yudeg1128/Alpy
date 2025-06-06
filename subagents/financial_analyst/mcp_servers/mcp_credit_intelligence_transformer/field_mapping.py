"""
Field mapping utilities for robust financial modeling across diverse bond data structures.
"""

# Maps canonical model field names to possible alternatives in extracted data
FIELD_ALIASES = {
    # Lending/Asset fields
    "net_loans": ["net_loans", "loans_net", "gross_loans", "loan_portfolio", "loan_book"],
    "total_revenue": ["total_revenue", "interest_income", "operating_income", "revenue"],
    "operating_expenses": ["operating_expenses", "management_and_operating_expenses", "operating_costs"],
    # Debt/Liabilities
    "total_debt": [
        "total_debt", "bond_payable", "loan_payable", "trust_payable", "asset_backed_securities", "other_payables"
    ],
    "accounts_receivable": ["accounts_receivable", "receivables"],
    "accounts_payable": ["accounts_payable", "payables", "other_payables"],
    "total_liabilities": ["total_liabilities", "liabilities", "total_liabilities_and_equity"],
    # Working capital
    "inventory": ["inventory", "inventories"],
    # Cash
    "cash_and_equivalents": ["cash_and_equivalents", "cash", "cash_balance"],
    # PPE
    "property_plant_and_equipment": ["property_plant_and_equipment", "ppe"],
    "accumulated_depreciation": ["accumulated_depreciation", "depreciation"],
    # Equity
    "total_equity": ["total_equity", "equity", "shareholders_equity"],
    # Income/Expense Ratios
    "interest_income_as_pct_of_average_net_loans": [
        "interest_income_as_pct_of_average_net_loans",
        "interest_yield_on_gross_loans",
        "interest_yield_on_loans",
        "interest_income_ratio"
    ],
    "operating_expenses_as_pct_of_interest_income": [
        "operating_expenses_as_pct_of_interest_income",
        "operating_expenses_as_pct_of_net_interest_income",
        "operating_expenses_as_pct_of_revenue"
    ],
    # Loan loss
    "loan_loss_provision_as_pct_of_ending_net_loans": [
        "loan_loss_provision_as_pct_of_ending_net_loans",
        "loan_loss_provision_as_pct_of_gross_loans"
    ],
    # Growth
    "net_loan_growth": ["net_loan_growth", "gross_loan_growth", "loan_portfolio_growth"],
    # Tax
    "income_tax_rate": ["income_tax_rate", "tax_rate"],
    # Capex
    "capex_absolute_value": ["capex_absolute_value", "capex"],
    # Depreciation
    "depreciation_absolute_value": ["depreciation_absolute_value", "depreciation"],
    # Add more mappings as needed
}


def get_best_field(data: dict, canonical: str, default=0):
    """
    Return the first found value from FIELD_ALIASES[canonical] in data, or default if none found.
    """
    for key in FIELD_ALIASES.get(canonical, [canonical]):
        if key in data and data[key] is not None:
            return data[key]
    return default


def get_best_assumption(assumptions_by_cat: dict, canonical: str, period: str, default=None):
    """
    Return the first found value for canonical metric in assumptions, considering FIELD_ALIASES.
    """
    for metric in FIELD_ALIASES.get(canonical, [canonical]):
        for cat in assumptions_by_cat.values():
            for a in cat:
                if a.get("metric_name") == metric and period in a.get("projection_periods", []):
                    idx = a["projection_periods"].index(period)
                    return a["values"][idx]
    return default
