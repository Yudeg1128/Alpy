import json
import logging
import sys
from pathlib import Path
from typing import List

import networkx as nx
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, fields, asdict
from typing import Any, Dict, List, Callable
from enum import Enum

# Add parent directory to path to import security_folder_utils
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))
from financial_analyst.security_folder_utils import (
    get_subfolder,
    require_security_folder
)

# --- Basic Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
# Using a marker file is a robust way to find the project root from any script location
PROJECT_ROOT_MARKER = '.project_root' 

# --- Helper Function for Path Resolution ---
def find_project_root(marker_file=PROJECT_ROOT_MARKER) -> Path:
    """Finds the project's root directory by searching upwards for a marker file."""
    current_dir = Path.cwd()
    while current_dir != current_dir.parent:
        if (current_dir / marker_file).exists():
            return current_dir
        current_dir = current_dir.parent
    raise FileNotFoundError(f"Project root marker '{marker_file}' not found. Cannot resolve file paths.")

context_logger = logging.getLogger('PeriodContext')
contracts_logger = logging.getLogger('EngineContracts')


class RevenueModel(Enum):
    """
    Defines the valid, recognized projection models for any revenue stream.
    The engine will FAIL FAST if it encounters a model name in the schema
    that is not a member of this Enum.
    """
    UNIT_ECONOMICS = "Unit_Economics"
    ASSET_YIELD_DRIVEN = "Asset_Yield_Driven"


class CogsModel(Enum):
    """
    Defines the valid, recognized projection models for any Cost of Revenue stream.
    """
    UNIT_ECONOMICS_COST = "Unit_Economics_Cost"
    ASSET_YIELD_DRIVEN_COST = "Asset_Yield_Driven_Cost"


class AccountCategory(Enum):
    """
    Defines the valid, recognized account categories that have special,
    hardcoded calculation logic in the engine. This maps the 'category' field
    derived from the schema structure to a specific behavior.
    """
    ACCOUNTS_RECEIVABLE = "accounts_receivable"
    ACCOUNTS_PAYABLE = "accounts_payable"
    PROPERTY_PLANT_EQUIPMENT = "property_plant_equipment"
    INTANGIBLE_ASSETS = "intangible_assets"
    LONG_TERM_DEBT = "long_term_debt"
    SHORT_TERM_DEBT = "short_term_debt"
    INDUSTRY_SPECIFIC_OPERATING_EXPENSES = "industry_specific_operating_expenses"
    INDUSTRY_SPECIFIC_ASSETS = "industry_specific_assets"
    INDUSTRY_SPECIFIC_LIABILITIES = "industry_specific_liabilities"


def log_contract_definitions():
    """
    A strategic logging function to be called once during engine initialization.
    It confirms that the contracts have been defined and lists them, providing a
    clear and verifiable checkpoint in the system's log output.
    """
    contracts_logger.info("--- [Contracts] Behavioral contracts defined and ready for engine validation. ---")
    
    all_contracts = {
        "Revenue Models": RevenueModel,
        "COGS Models": CogsModel,
        "Account Categories": AccountCategory
    }
    
    for name, contract_enum in all_contracts.items():
        # Retrieve the actual string values that the schema will use
        members = [member.value for member in contract_enum]
        contracts_logger.info(f"  -> Recognized {name}: {members}")
    
    contracts_logger.info("--- [Contracts] Engine will now enforce these contracts during execution. ---")


class SchemaPaths:
    """
    A centralized, static contract for all schema traversal paths.
    This prevents typo-related bugs and simplifies maintenance by providing a
    single source of truth for the schema's structure.
    """
    contracts_logger.info("--- [Contracts] Defining schema path contracts... ---")
    
    # --- Income Statement Paths ---
    IS_REVENUE = "income_statement.revenue"
    IS_COGS = "income_statement.cost_of_revenue"
    IS_OPEX = "income_statement.operating_expenses"
    IS_INDUSTRY_OPEX = "income_statement.industry_specific_operating_expenses"
    IS_NON_OPERATING = "income_statement.non_operating_income_expense"
    
    # --- Balance Sheet Asset Paths ---
    BS_ASSETS_CASH = "balance_sheet.assets.current_assets.cash_and_equivalents"
    BS_ASSETS_AR = "balance_sheet.assets.current_assets.accounts_receivable"
    BS_ASSETS_INVENTORY = "balance_sheet.assets.current_assets.inventory"
    BS_ASSETS_OTHER_CURRENT = "balance_sheet.assets.current_assets.other_current_assets"
    
    BS_ASSETS_PPE = "balance_sheet.assets.non_current_assets.property_plant_equipment"
    BS_ASSETS_INTANGIBLES = "balance_sheet.assets.non_current_assets.intangible_assets"
    BS_ASSETS_OTHER_NON_CURRENT = "balance_sheet.assets.non_current_assets.other_non_current_assets"
    
    BS_ASSETS_CONTRA = "balance_sheet.assets.contra_assets"

    # --- Balance Sheet Liability Paths ---
    BS_LIABILITIES_AP = "balance_sheet.liabilities.current_liabilities.accounts_payable"
    BS_LIABILITIES_ST_DEBT = "balance_sheet.liabilities.current_liabilities.short_term_debt"
    BS_LIABILITIES_TAX_PAYABLE = "balance_sheet.liabilities.current_liabilities.tax_payables"
    BS_LIABILITIES_OTHER_CURRENT = "balance_sheet.liabilities.current_liabilities.other_current_liabilities"
    
    BS_LIABILITIES_LT_DEBT = "balance_sheet.liabilities.non_current_liabilities.long_term_debt"
    BS_LIABILITIES_OTHER_NON_CURRENT = "balance_sheet.liabilities.non_current_liabilities.other_non_current_liabilities"
    
    # --- Balance Sheet Equity Path ---
    BS_EQUITY = "balance_sheet.equity"

    # --- Industry-Specific BS Paths ---
    BS_INDUSTRY_ASSETS = "balance_sheet.industry_specific_items.industry_specific_assets"
    BS_INDUSTRY_LIABILITIES = "balance_sheet.industry_specific_items.industry_specific_liabilities"

    contracts_logger.info("--- [Contracts] Schema path contracts defined. ---")


@dataclass
class CashFlowStatement:
    """
    A strict, type-safe data contract for the articulated Cash Flow Statement.

    This class serves as the single source of truth for the structure and
    line items of the CFS. It is instantiated once per projection period
    within the PeriodContext, providing a clean, zeroed-out workspace.

    Its primary role is to enforce explicitness, eliminating "magic strings"
    and ensuring that every component of cash flow is accounted for. The
    calculation logic will populate these fields, and the final values will be
    committed to the main data grid.
    """

    # =====================================================================
    # CASH FLOW FROM OPERATIONS (CFO)
    # =====================================================================
    # The starting point, linked directly from the Income Statement.
    net_income: float = 0.0

    # Non-cash charges added back to reconcile net income to cash.
    depreciation_and_amortization: float = 0.0

    # Changes in Operating Working Capital.
    # Convention: A positive value represents a cash INFLOW.
    # e.g., a decrease in Accounts Receivable is a positive change.
    change_in_accounts_receivable: float = 0.0
    change_in_inventory: float = 0.0
    change_in_other_current_assets: float = 0.0
    change_in_accounts_payable: float = 0.0
    change_in_other_current_liabilities: float = 0.0

    # Subtotal for the CFO section.
    cash_from_operations: float = 0.0

    # =====================================================================
    # CASH FLOW FROM INVESTING (CFI)
    # =====================================================================
    # The primary driver of investing cash flow.
    # Convention: This will be a negative value, representing a cash outflow.
    capital_expenditures: float = 0.0
    
    # Placeholder for other investing activities like asset sales (inflow)
    # or acquisitions (outflow).
    other_investing_activities: float = 0.0

    # Subtotal for the CFI section.
    cash_from_investing: float = 0.0

    # =====================================================================
    # CASH FLOW FROM FINANCING (CFF)
    # =====================================================================
    # Changes in debt principals.
    change_in_short_term_debt: float = 0.0
    change_in_long_term_debt: float = 0.0
    
    # This is the revolver draw (inflow) or paydown (outflow). This will
    # be the final balancing item calculated by the funding gap logic.
    change_in_revolver: float = 0.0
    
    # Changes in equity.
    # Convention: A positive value for issuance (inflow), negative for buybacks (outflow).
    net_share_issuance: float = 0.0
    # Convention: This will be a negative value, representing a cash outflow.
    dividends_paid: float = 0.0

    # Subtotal for the CFF section.
    cash_from_financing: float = 0.0

    # =====================================================================
    # RECONCILIATION
    # =====================================================================
    # The sum of the three main sections (CFO + CFI + CFF). This value
    # directly determines the change in the cash balance on the Balance Sheet.
    net_change_in_cash: float = 0.0


@dataclass
class SchemaItemGroups:
    """
    An explicit data contract for collections of schema-derived account names.
    This replaces the 'schema_items' dictionary, providing compile-time safety
    and autocompletion for accessing groups of accounts (e.g., all revenue streams).
    """
    revenue: List[str]
    cost_of_revenue: List[str]
    cash_items: List[str]
    inventory_items: List[str]
    other_current_assets: List[str]
    other_non_current_assets: List[str]
    lt_debt_items: List[str]
    other_ncl_items: List[str]
    ind_liab_items: List[str]
    ind_asset_items: List[str]
    ppe_items: List[str]
    intangible_asset_items: List[str]
    contra_asset_items: List[str]
    ap_items: List[str]
    ar_items: List[str]
    st_debt_items: List[str]
    other_cl_items: List[str]
    opex_items: List[str]
    industry_opex_items: List[str]
    non_op_items: List[str]
    equity_items: List[str]
    tax_payable_items: List[str]


@dataclass
class ProjectionConstants:
    """
    An explicit, type-hinted data contract for all pre-calculated constants
    used in the projection engine.

    This replaces a generic dictionary, providing compile-time safety,
    IDE autocompletion, and effortless refactoring for all constants. It serves
    as the single source of truth for the names and types of all derived
    engine parameters.
    """
    account_map: Dict[str, Any]
    deterministic_depreciation_rate: float
    deterministic_amortization_rate: float
    deterministic_dso: float
    deterministic_dpo: float
    effective_tax_rate: float
    primary_cash_account: str
    revolver_interest_rate: float
    unit_economics_tracker: Dict[str, Any]
    schema_items: SchemaItemGroups
    schema: Dict[str, Any]


@dataclass
class PeriodContext:
    """
    A self-contained, transactional workspace for a single projection period.

    This object holds all necessary state for one period's calculations,
    enforcing a clean separation between prior period data (read-only)
    and current period calculations (write-only workspace). This design
    prevents temporal bleeding and makes the calculation process deterministic
    and testable.

    Attributes:
        period_column_name (str): The name of the projection column (e.g., "P1").
        period_index (int): The 1-based index of the projection year.
        opening_balances (pd.Series): A read-only series of the final, committed
            balances from the end of the prior period.
        constants (ProjectionConstants): A read-only object of all pre-calculated
            projection constants (rates, ratios, etc.).
        workspace (Dict[str, float]): The primary workspace for storing the results
            of the current period's calculations.
        period_aggregates (Dict[str, float]): A workspace for accumulating values
            within a period, such as total depreciation or amortization.
        cfs (CashFlowStatement): A dedicated, structured workspace for articulating
            the Cash Flow Statement for the current period.
    """
    # --- Read-Only Inputs ---
    period_column_name: str
    period_index: int
    opening_balances: pd.Series
    constants: ProjectionConstants

    # --- Read/Write Workspaces ---
    workspace: Dict[str, float] = field(default_factory=dict)
    period_aggregates: Dict[str, float] = field(default_factory=dict)
    cfs: 'CashFlowStatement' = field(default_factory=lambda: CashFlowStatement())

    def __post_init__(self):
        """
        Validation hook that runs after initialization. Enforces the
        "Fail Fast and Loud" principle.
        """
        context_logger.info(
            f"Context created for period '{self.period_column_name}' (Index: {self.period_index})."
        )
        if not isinstance(self.period_column_name, str) or not self.period_column_name:
            raise TypeError("`period_column_name` must be a non-empty string.")
        if not isinstance(self.period_index, int) or self.period_index <= 0:
            raise ValueError("`period_index` must be a positive integer.")
        if not isinstance(self.opening_balances, pd.Series):
            raise TypeError("`opening_balances` must be a pandas Series.")
        if self.opening_balances.empty:
            raise ValueError("`opening_balances` cannot be empty.")
        if not isinstance(self.constants, ProjectionConstants):
            raise TypeError("`constants` must be a ProjectionConstants object.")

    def set(self, account: str, value: float) -> None:
        """
        Writes a calculated value to the current period's workspace.
        This is the primary method for persisting a calculation result within a period.
        """
        if not isinstance(account, str) or not account:
            raise TypeError(f"Account name must be a non-empty string. Received: {account}")

        # context_logger.info(f"[CONTEXT SET] {self.period_column_name} | {account:<30} = {float(value):,.2f}")
        self.workspace[account] = float(value)

    def get(self, account: str) -> float:
        """
        Retrieves a value, searching the current period's workspace first,
        then falling back to the read-only opening balances.

        This method enforces the "No Fallbacks" principle. If an account
        is not found in either location, it raises a KeyError, as this
        indicates a missing dependency or an incorrect execution order.
        """
        if not isinstance(account, str) or not account:
            raise TypeError(f"Account name must be a non-empty string. Received: {account}")

        if account in self.workspace:
            value = self.workspace[account]
            # context_logger.info(f"[CONTEXT GET] {self.period_column_name} | {account:<30} -> {value:,.2f} (from Workspace)")
            return value

        if account in self.opening_balances:
            value = self.opening_balances[account]
            # context_logger.info(f"[CONTEXT GET] {self.period_column_name} | {account:<30} -> {value:,.2f} (from Opening Balances)")
            return value

        raise KeyError(
            f"Account '{account}' not found in the current period's workspace "
            f"or in the opening balances for period '{self.period_column_name}'. "
            f"Check for a missing calculation or incorrect dependency order."
        )

    def add_to_aggregate(self, key: str, value: float) -> None:
        """
        Adds a value to a running total within the period_aggregates workspace.
        Useful for accumulating items like total D&A.
        """
        if not isinstance(key, str) or not key:
            raise TypeError(f"Aggregate key must be a non-empty string. Received: {key}")

        current_total = self.period_aggregates.get(key, 0.0)
        new_total = current_total + float(value)
        self.period_aggregates[key] = new_total
        context_logger.info(f"[CONTEXT AGG] {self.period_column_name} | {key:<30} += {float(value):,.2f} (New Total: {new_total:,.2f})")

    def get_aggregate(self, key: str) -> float:
        """
        Retrieves the current total for an aggregate key. Returns 0.0 if the
        key does not exist, as this is a valid state for an accumulator.
        """
        if not isinstance(key, str) or not key:
            raise TypeError(f"Aggregate key must be a non-empty string. Received: {key}")

        return self.period_aggregates.get(key, 0.0)

    def get_driver_value(self, driver_obj: dict) -> float:
        """
        Extracts the correct driver value (baseline, short_term, etc.) for this
        context's projection year index. [STRICT VERSION]

        This is a pure method that operates on the context's state and fails loudly
        if the driver object is malformed or a value for the requested period is
        missing (None). It is the single source of truth for driver lookups.
        """

        trends = driver_obj.get('trends', {})
        if not isinstance(trends, dict):
            raise TypeError(f"Driver's 'trends' object must be a dictionary. Received: {type(trends)}")

        if self.period_index == 1:
            value = driver_obj.get('baseline')
        elif 2 <= self.period_index <= 3:
            value = trends.get('short_term')
        elif 4 <= self.period_index <= 9:
            value = trends.get('medium_term')
        else:  # Year 10 and onwards
            value = trends.get('terminal')

        if value is None:
            raise ValueError(
                f"Driver value for projection index {self.period_index} is missing (None). "
                f"Check schema for driver: {driver_obj}"
            )
        
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"Driver value for projection index {self.period_index} must be a number. "
                f"Received type '{type(value)}' for driver: {driver_obj}"
            )

        return float(value)


class FinancialCalculations:
    """
    A stateless container for all financial calculation logic. This final version uses a
    fully explicit, type-safe data contract for all constants and account groups,
    eliminating all "magic string" dependencies for maximum safety and readability.
    """

    # =====================================================================
    # REVENUE & COGS
    # =====================================================================
    @staticmethod
    def calculate_asset_yield_revenue(account: str, context: PeriodContext) -> None:
        """Calculates revenue for an Asset_Yield_Driven model."""
        account_map = context.constants.account_map
        model = account_map[account]['schema_entry']['projection_configuration']['selected_model']
        yield_driver = model['drivers']['asset_yield']

        current_yield = context.get_driver_value(yield_driver)

        target_hist_key = model.get('target_asset_account')
        if not target_hist_key:
            raise KeyError(f"Revenue model for '{account}' is missing 'target_asset_account'.")

        target_node = next((name for name, entry in account_map.items() if entry['schema_entry'].get('historical_account_key') == target_hist_key), None)
        if not target_node:
            raise RuntimeError(f"Revenue model for '{account}' depends on '{target_hist_key}', but no item is mapped to this key.")

        # Use ONLY the opening balance to permanently break the major circularity.
        opening_asset_balance = context.opening_balances.get(target_node, 0.0)
        revenue = opening_asset_balance * current_yield
        context.set(account, revenue)

    @staticmethod
    def calculate_lender_cogs(account: str, context: PeriodContext) -> None:
        """
        Calculates COGS for a Lender, which is their primary interest expense.
        [REVISED & STABLE] This version uses ONLY the opening debt balance to
        calculate interest expense. This intentionally breaks the COGS-Debt circularity,
        making the P&L calculation stable and symmetrical with the asset-yield revenue logic.
        """
        schema = context.constants.schema
        try:
            drivers = schema['balance_sheet']['liabilities']['non_current_liabilities']['long_term_debt']['drivers']
            rate_driver = drivers['average_interest_rate']
            current_rate = context.get_driver_value(rate_driver)
        except KeyError as e:
            raise RuntimeError(f"FATAL: Could not find debt driver for interest rate in schema. Error: {e}")

        # --- CRITICAL FIX: Use ONLY the opening balances ---
        # This makes the COGS calculation stable and independent of the current period's
        # closing debt balances, mirroring the revenue calculation logic.
        lt_debt_items = context.constants.schema_items.lt_debt_items
        total_opening_debt_balance = sum(context.opening_balances.get(item, 0.0) for item in lt_debt_items)
        
        # We must also include the opening revolver balance, as it is also interest-bearing.
        opening_revolver_balance = context.opening_balances.get('revolver', 0.0)
        
        total_interest_bearing_debt = total_opening_debt_balance + opening_revolver_balance

        # Calculate interest expense (as a negative value for P&L)
        cogs = total_interest_bearing_debt * current_rate * -1
        
        logger.info(
            f"[{context.period_column_name}][{account}] COGS: {cogs:,.0f} "
            f"(Opening Debt: {total_interest_bearing_debt:,.0f} * Rate: {current_rate:.2%})"
        )
        context.set(account, cogs)

    # =====================================================================
    # ASSET ROLL-FORWARDS (FINALIZED)
    # =====================================================================
    @staticmethod
    def calculate_ppe_rollforward(account: str, context: PeriodContext) -> None:
        """Calculates the ABSOLUTE CLOSING BALANCE of a PP&E item for the period."""
        account_map = context.constants.account_map
        schema_entry = account_map[account]['schema_entry']
        try:
            capex_driver = schema_entry['drivers']['capex_amount_annual']
            capex_amount = context.get_driver_value(capex_driver)
            # --- NEW: Publish the calculated capex amount to aggregates ---
            context.add_to_aggregate('total_capex', capex_amount)

        except KeyError:
            raise KeyError(f"The 'capex_amount_annual' driver is missing for PP&E item '{account}'.")

        opening_balance = context.opening_balances.get(account, 0.0)
        depreciation_rate = context.constants.deterministic_depreciation_rate
        depreciation_amount = opening_balance * depreciation_rate
        context.add_to_aggregate('aggregated_depreciation', depreciation_amount * -1)

        closing_balance = opening_balance + capex_amount - depreciation_amount
        context.set(account, closing_balance)
        # The existing logger is fine and does not need to be changed.
        logger.info(f"[{context.period_column_name}][{account}] Closing Balance: {closing_balance:,.0f} (Opening: {opening_balance:,.0f} + Capex: {capex_amount:,.0f} - Depr: {depreciation_amount:,.0f})")

    @staticmethod
    def calculate_intangible_rollforward(account: str, context: PeriodContext) -> None:
        """Calculates the ABSOLUTE CLOSING BALANCE of an Intangible Asset for the period."""
        account_map = context.constants.account_map
        schema_entry = account_map[account]['schema_entry']
        try:
            additions_driver = schema_entry['drivers']['intangible_additions_annual']
            additions_amount = context.get_driver_value(additions_driver)
        except KeyError:
            raise KeyError(f"The 'intangible_additions_annual' driver is missing for Intangible Asset item '{account}'.")

        opening_balance = context.opening_balances.get(account, 0.0)
        amortization_rate = context.constants.deterministic_amortization_rate
        amortization_amount = opening_balance * amortization_rate
        context.add_to_aggregate('aggregated_amortization', amortization_amount * -1)

        closing_balance = opening_balance + additions_amount - amortization_amount
        context.set(account, closing_balance)
        logger.info(f"[{context.period_column_name}][{account}] Closing Balance: {closing_balance:,.0f} (Opening: {opening_balance:,.0f} + Additions: {additions_amount:,.0f} - Amort: {amortization_amount:,.0f})")

    @staticmethod
    def calculate_industry_specific_asset_growth(account: str, context: PeriodContext) -> None:
        """Calculates the ABSOLUTE CLOSING BALANCE for an industry-specific asset."""
        account_map = context.constants.account_map
        schema_entry = account_map[account]['schema_entry']
        growth_driver = schema_entry['drivers']['industry_specific_asset_growth_rate']
        growth_rate = context.get_driver_value(growth_driver)

        opening_balance = context.opening_balances.get(account, 0.0)
        change_in_asset = opening_balance * growth_rate
        closing_balance = opening_balance + change_in_asset
        context.set(account, closing_balance)

    # =====================================================================
    # LIABILITY & EQUITY CALCULATIONS (FINALIZED)
    # =====================================================================
    @staticmethod
    def calculate_total_debt_structure(account: str, context: PeriodContext) -> None:
        """
        Calculates the ABSOLUTE CLOSING BALANCE of a single debt instrument (short or long term)
        by targeting a total debt-to-assets ratio for the entire interest-bearing debt stack.
        [REVISED & CONSOLIDATED METHOD]

        This function replaces the isolated long-term debt calculation. It ensures that
        all interest-bearing debt (both ST and LT) moves in concert to meet the company's
        strategic capital structure policy, preventing uncontrolled feedback loops.
        """
        s = context.constants.schema_items
        schema = context.constants.schema
        
        # --- Phase 1: Calculate Target Total Debt ---
        # Get the primary policy driver from the schema.
        try:
            debt_drivers = schema['balance_sheet']['liabilities']['non_current_liabilities']['long_term_debt']['drivers']
            target_ratio_driver = debt_drivers['target_debt_as_percent_of_assets']
            target_debt_to_assets_ratio = context.get_driver_value(target_ratio_driver)
        except KeyError as e:
            raise RuntimeError(f"FATAL: Could not find debt driver 'target_debt_as_percent_of_assets' in schema. Error: {e}")

        # Get the calculated total assets for the current period.
        # This is the core link that drives the capital structure.
        current_total_assets = context.get('total_assets')
        target_total_interest_bearing_debt = current_total_assets * target_debt_to_assets_ratio

        # --- Phase 2: Determine Current Debt and Required Change ---
        # Consolidate all interest-bearing debt items from the schema constants.
        all_debt_items = s.st_debt_items + s.lt_debt_items
        if not all_debt_items:
            # If there are no debt items defined in the schema, there's nothing to calculate.
            logger.warning(f"[{context.period_column_name}] Capital structure calculation for '{account}' skipped as no ST/LT debt items are defined in the schema.")
            context.set(account, 0.0)
            return

        # Sum the opening balances of all debt instruments.
        opening_total_debt = sum(context.opening_balances.get(item, 0.0) for item in all_debt_items)
        
        # Calculate the net change required across the entire debt stack.
        total_required_change_in_debt = target_total_interest_bearing_debt - opening_total_debt

        # --- Phase 3: Allocate the Change Proportionally ---
        opening_balance_of_this_account = context.opening_balances.get(account, 0.0)

        # Handle the edge case of starting with zero debt.
        if opening_total_debt == 0:
            # If there's no existing debt, we can't use proportions.
            # We must allocate the required change equally among all defined debt instruments.
            # This is a reasonable assumption for a model's first period or a startup.
            proportion_of_total_debt = 1.0 / len(all_debt_items)
            logger.info(f"[{context.period_column_name}][Debt] Opening debt is zero. Applying equal allocation ({proportion_of_total_debt:.2%}) for '{account}'.")
        else:
            # Allocate the change based on each instrument's share of the opening debt.
            proportion_of_total_debt = opening_balance_of_this_account / opening_total_debt

        change_for_this_account = total_required_change_in_debt * proportion_of_total_debt
        
        # --- Phase 4: Set the Final Closing Balance ---
        closing_balance = opening_balance_of_this_account + change_for_this_account
        
        # CRITICAL: A debt balance cannot be negative. If repayments exceed the opening balance,
        # the closing balance is floored at zero.
        final_closing_balance = max(0, closing_balance)
        
        if final_closing_balance < closing_balance:
            logger.warning(
                f"[{context.period_column_name}][Debt] Calculated negative balance for '{account}' ({closing_balance:,.0f}). "
                f"Flooring at zero. This may indicate required repayments exceed available debt."
            )

        context.set(account, final_closing_balance)

        # Detailed logging for traceability
        logger.info(
            f"[{context.period_column_name}][Debt][{account}] Closing: {final_closing_balance:,.0f} "
            f"(Opening: {opening_balance_of_this_account:,.0f}, "
            f"ΔAllocated: {change_for_this_account:,.0f})"
        )

    @staticmethod
    def calculate_equity_rollforward(account: str, context: PeriodContext) -> None:
        """
        Calculates the ABSOLUTE CLOSING BALANCE for 'retained_earnings' and publishes
        the dividend payment amount for use in the CFS.
        All other equity accounts are held constant.
        """
        opening_balance = context.opening_balances.get(account, 0.0)

        if account == 'retained_earnings':
            net_income = context.get('net_income')

            account_map = context.constants.account_map
            schema_entry = account_map[account]['schema_entry']
            payout_driver = schema_entry['drivers']['dividend_payout_ratio']
            payout_ratio = context.get_driver_value(payout_driver)

            dividends_paid = max(0, net_income * payout_ratio)
            # --- NEW: Publish the calculated dividend amount to aggregates ---
            # We store it as a positive number representing an outflow.
            context.add_to_aggregate('dividends_paid', dividends_paid)

            change_in_re = net_income - dividends_paid
            closing_balance = opening_balance + change_in_re
            
            logger.info(
                f"[{context.period_column_name}][RE Roll-Forward] "
                f"Opening: {opening_balance:,.0f} + NI: {net_income:,.0f} - Div: {dividends_paid:,.0f} = Closing: {closing_balance:,.0f}"
            )
            context.set(account, closing_balance)
                
        else:
            context.set(account, opening_balance)

    @staticmethod
    def calculate_common_stock(account: str, context: PeriodContext) -> None:
        """Calculates the ABSOLUTE CLOSING BALANCE of common stock and publishes the change."""
        account_map = context.constants.account_map
        schema_entry = account_map[account]['schema_entry']
        issuance_driver = schema_entry['drivers']['net_share_issuance']
        
        change_in_stock = context.get_driver_value(issuance_driver)
        # --- NEW: Publish the change amount to aggregates ---
        context.add_to_aggregate('net_share_issuance', change_in_stock)
        
        opening_balance = context.opening_balances.get(account, 0.0)
        closing_balance = opening_balance + change_in_stock
        context.set(account, closing_balance)

    @staticmethod
    def calculate_industry_specific_liability(account: str, context: PeriodContext) -> None:
        """Calculates the ABSOLUTE CLOSING BALANCE for an industry-specific liability."""
        account_map = context.constants.account_map
        schema_entry = account_map[account]['schema_entry']
        driver_obj = schema_entry['drivers']['industry_specific_liability_growth_rate']
        growth_rate = context.get_driver_value(driver_obj)
        opening_balance = context.opening_balances.get(account, 0.0)
        change = opening_balance * growth_rate
        closing_balance = opening_balance + change
        context.set(account, closing_balance)

    # =====================================================================
    # WORKING CAPITAL (FINALIZED)
    # =====================================================================
    @staticmethod
    def calculate_working_capital_change(account: str, context: PeriodContext) -> None:
        """Calculates the ABSOLUTE CLOSING BALANCE of an A/R or A/P item."""
        account_map = context.constants.account_map
        category = account_map[account]['category']

        if category == 'accounts_receivable':
            projected_revenue = context.get('total_revenue')
            dso = context.constants.deterministic_dso
            target_ar_balance = (projected_revenue * dso) / 365.0
            context.set(account, target_ar_balance)

        elif category == 'accounts_payable':
            cogs_items = context.constants.schema_items.cost_of_revenue
            projected_cogs = abs(sum(context.get(item) for item in cogs_items))
            dpo = context.constants.deterministic_dpo
            target_ap_balance = (projected_cogs * dpo) / 365.0
            context.set(account, target_ap_balance)
        else:
            raise NotImplementedError(f"Working capital logic not implemented for account '{account}' in category '{category}'")

    # =====================================================================
    # AGGREGATED & CORE P&L ITEMS (from embedded logic)
    # =====================================================================
    @staticmethod
    def calculate_sga_expense(account: str, context: PeriodContext) -> None:
        """Calculates SGA expense as a percentage of revenue."""
        account_map = context.constants.account_map
        schema_entry = account_map[account]['schema_entry']
        driver = schema_entry['drivers']['sga_expense_as_percent_of_revenue']
        rate = context.get_driver_value(driver)
        total_revenue = context.get('total_revenue')
        expense = total_revenue * rate * -1
        context.set(account, expense)

    @staticmethod
    def calculate_industry_specific_opex(account: str, context: PeriodContext) -> None:
        """Calculates an industry-specific operating expense as a percentage of revenue."""
        account_map = context.constants.account_map
        schema_entry = account_map[account]['schema_entry']
        driver = schema_entry['drivers']['industry_specific_operating_expense_as_percent_of_revenue']
        rate = context.get_driver_value(driver)
        total_revenue = context.get('total_revenue')
        expense = total_revenue * rate * -1
        context.set(account, expense)

    @staticmethod
    def calculate_income_tax(account: str, context: PeriodContext) -> None:
        """
        Calculates income tax as a percentage of EBT using the pre-calculated
        historical effective tax rate.
        [REVISED & CORRECTED] This version uses a data-driven rate and correctly
        calculates tax on positive earnings.
        """
        # Retrieve the data-driven tax rate from the constants object.
        tax_rate = context.constants.effective_tax_rate
        
        # Get the projected EBT for the current period.
        ebt_value = context.get('ebt')
        
        # Tax should only be calculated on positive earnings.
        # A more advanced model would create a Deferred Tax Asset for losses.
        # For now, we assume no tax credit/refund for a loss.
        taxable_income = max(0, ebt_value)
        
        # Tax expense is a negative value on the P&L.
        tax_expense = taxable_income * tax_rate * -1
        
        logger.info(
            f"[{context.period_column_name}][{account}] Tax Expense: {tax_expense:,.0f} "
            f"(EBT: {ebt_value:,.0f} * Rate: {tax_rate:.2%})"
        )
        context.set(account, tax_expense)

    @staticmethod
    def calculate_depreciation_expense(account: str, context: PeriodContext) -> None:
        """Retrieves the aggregated depreciation for the period."""
        total_depreciation = context.get_aggregate('aggregated_depreciation')
        context.set(account, total_depreciation)

    @staticmethod
    def calculate_amortization_expense(account: str, context: PeriodContext) -> None:
        """Retrieves the aggregated amortization for the period."""
        total_amortization = context.get_aggregate('aggregated_amortization')
        context.set(account, total_amortization)

    # =====================================================================
    # ENGINE-INTERNAL ITEMS (from embedded logic)
    # =====================================================================
    @staticmethod
    def calculate_total_revenue(account: str, context: PeriodContext) -> None:
        """Calculates total revenue by summing all revenue stream items."""
        revenue_streams = context.constants.schema_items.revenue
        total_revenue = sum(context.get(stream) for stream in revenue_streams)
        context.set(account, total_revenue)

    @staticmethod
    def calculate_interest_on_revolver(account: str, context: PeriodContext) -> None:
        """Calculates interest expense on the revolver."""
        opening_revolver = context.opening_balances.get('revolver', 0.0)
        closing_revolver = context.get('revolver')
        avg_revolver = (opening_revolver + closing_revolver) / 2
        interest_rate = context.constants.revolver_interest_rate
        interest_expense = avg_revolver * interest_rate * -1
        context.set(account, interest_expense)

    @staticmethod
    def calculate_interest_income_on_cash(account: str, context: PeriodContext) -> None:
        """Calculates interest income on cash balances."""
        schema = context.constants.schema
        policy = schema['balance_sheet']['assets']['current_assets']['cash_and_equivalents']['income_policy']
        yield_driver = policy['drivers']['yield_on_cash_and_investments']
        current_yield = context.get_driver_value(yield_driver)
        
        cash_items = context.constants.schema_items.cash_items
        opening_cash = sum(context.opening_balances.get(item, 0.0) for item in cash_items)
        closing_cash = sum(context.get(item) for item in cash_items)
        avg_cash = (opening_cash + closing_cash) / 2
        
        interest_income = avg_cash * current_yield
        context.set(account, interest_income)

    # =====================================================================
    # FINANCIAL STATEMENT SUBTOTALS
    # =====================================================================
    @staticmethod
    def calculate_gross_profit(account: str, context: PeriodContext) -> None:
        """Calculates Gross Profit for the period."""
        total_revenue = context.get('total_revenue')
        cogs_items = context.constants.schema_items.cost_of_revenue
        total_cogs = sum(context.get(item) for item in cogs_items)
        gross_profit = total_revenue + total_cogs
        context.set(account, gross_profit)

    @staticmethod
    def calculate_operating_income(account: str, context: PeriodContext) -> None:
        """Calculates Operating Income for the period."""
        gross_profit = context.get('gross_profit')
        s = context.constants.schema_items
        total_opex = sum(context.get(item) for item in s.opex_items)
        total_ind_opex = sum(context.get(item) for item in s.industry_opex_items)
        
        operating_income = gross_profit + total_opex + total_ind_opex
        context.set(account, operating_income)

    @staticmethod
    def calculate_ebt(account: str, context: PeriodContext) -> None:
        """Calculates Earnings Before Tax (EBT) for the period."""
        operating_income = context.get('operating_income')
        non_op_items = context.constants.schema_items.non_op_items
        
        total_non_op = sum(context.get(item) for item in non_op_items)
        interest_on_revolver = context.get('interest_on_revolver')
        
        ebt = operating_income + total_non_op + interest_on_revolver
        context.set(account, ebt)

    @staticmethod
    def calculate_net_income(account: str, context: PeriodContext) -> None:
        """Calculates Net Income for the period."""
        ebt = context.get('ebt')
        tax_expense = context.get('income_tax_expense')
        net_income = ebt + tax_expense
        context.set(account, net_income)

    @staticmethod
    def calculate_total_current_assets(account: str, context: PeriodContext) -> None:
        """
        Calculates the ABSOLUTE CLOSING BALANCE for Total Current Assets.

        This function's sole responsibility is to sum the period-end balances of all
        component current asset accounts (cash, A/R, inventory, etc.) to ensure
        the subtotal is always internally consistent.
        """
        # Using a convenience variable for cleaner access to the schema item groups.
        s = context.constants.schema_items

        # --- CORRECTED LOGIC ---
        # 1. Assemble the complete list of all component account names.
        #    This is guaranteed to be correct by our SchemaItemGroups data contract.
        component_accounts = (
            s.cash_items +
            s.ar_items +
            s.inventory_items +
            s.other_current_assets
        )

        total_ca = 0.0
        
        # 2. Add detailed logging to show exactly what is being summed.
        #    This is critical for debugging timing issues in the dependency graph.
        logger.info(
            f"[{context.period_column_name}][SUMMATION] Calculating '{account}' from {len(component_accounts)} components..."
        )

        # 3. Iterate and sum the most up-to-date value for each component from the context.
        for item in component_accounts:
            # The context.get() method ensures we retrieve the value from the
            # current period's workspace if it exists, otherwise the opening balance.
            value = context.get(item)
            logger.info(f"[{context.period_column_name}][SUMMATION]  + {item:<30} = {value:,.2f}")
            total_ca += value
        
        logger.info(f"[{context.period_column_name}][SUMMATION]  = Final Total '{account}': {total_ca:,.2f}")

        # 4. Set the final, correctly summed value in the context.
        context.set(account, total_ca)

    @staticmethod
    def calculate_total_non_current_assets(account: str, context: PeriodContext) -> None:
        """Calculates Total Non-Current Assets for the period."""
        s = context.constants.schema_items
        all_nca_items = s.ppe_items + s.intangible_asset_items + s.other_non_current_assets + s.ind_asset_items
        total_nca = sum(context.get(item) for item in all_nca_items)
        context.set(account, total_nca)

    @staticmethod
    def calculate_total_assets(account: str, context: PeriodContext) -> None:
        """Calculates Total Assets for the period."""
        total_ca = context.get('total_current_assets')
        total_nca = context.get('total_non_current_assets')
        contra_asset_items = context.constants.schema_items.contra_asset_items
        
        total_contra = sum(context.get(item) for item in contra_asset_items)
        
        total_assets = total_ca + total_nca + total_contra
        context.set(account, total_assets)

    @staticmethod
    def calculate_total_current_liabilities(account: str, context: PeriodContext) -> None:
        """
        Calculates Total Current Liabilities for the period.
        [REVISED & CORRECTED] This version now correctly includes the revolver
        balance, which is essential for the final balance sheet articulation.
        """
        s = context.constants.schema_items
        
        # Get all schema-defined current liability items
        component_accounts = (
            s.ap_items + 
            s.st_debt_items + 
            s.other_cl_items + 
            s.tax_payable_items
        )
        
        # --- CRITICAL FIX: Add the revolver to the list of items to be summed ---
        component_accounts.append('revolver')
        
        logger.info(
            f"[{context.period_column_name}][SUMMATION] Calculating '{account}' from {len(component_accounts)} components..."
        )

        total_cl = 0.0
        for item in sorted(list(set(component_accounts))): # Use set to avoid duplicates
            value = context.get(item)
            logger.info(f"[{context.period_column_name}][SUMMATION]  + {item:<30} = {value:,.2f}")
            total_cl += value
        
        logger.info(f"[{context.period_column_name}][SUMMATION]  = Final Total '{account}': {total_cl:,.2f}")

        context.set(account, total_cl)

    @staticmethod
    def calculate_total_non_current_liabilities(account: str, context: PeriodContext) -> None:
        """Calculates Total Non-Current Liabilities for the period."""
        s = context.constants.schema_items
        all_ncl_items = s.lt_debt_items + s.other_ncl_items + s.ind_liab_items
        total_ncl = sum(context.get(item) for item in all_ncl_items)
        context.set(account, total_ncl)

    @staticmethod
    def calculate_total_liabilities(account: str, context: PeriodContext) -> None:
        """Calculates Total Liabilities for the period."""
        total_cl = context.get('total_current_liabilities')
        total_ncl = context.get('total_non_current_liabilities')
        total_liabilities = total_cl + total_ncl
        context.set(account, total_liabilities)

    @staticmethod
    def calculate_total_equity(account: str, context: PeriodContext) -> None:
        """Calculates Total Equity for the period."""
        equity_items = context.constants.schema_items.equity_items
        total_equity = sum(context.get(item) for item in equity_items)
        context.set(account, total_equity)

    @staticmethod
    def calculate_total_liabilities_and_equity(account: str, context: PeriodContext) -> None:
        """Calculates Total Liabilities & Equity for the period."""
        total_liabilities = context.get('total_liabilities')
        total_equity = context.get('total_equity')
        total_l_and_e = total_liabilities + total_equity
        context.set(account, total_l_and_e)


class ProjectionsEngine:
    """
    A deterministic financial projection engine.
    
    It ingests a populated schema and historical data to produce a fully articulated
    three-statement financial projection. The engine uses a Directed Acyclic Graph (DAG)
    and Strongly Connected Components (SCC) analysis to create a deterministic,
    non-redundant execution plan, with a targeted iterative solver for the
    revolver-interest circularity.
    """

    def __init__(self, security_id: str, projection_years: int):
        """
        Initializes the engine, sets up paths, and prepares for projection.
        
        Args:
            security_id: The unique identifier for the security to be processed.
            projection_years: The number of years to project forward.
        """
        if not isinstance(security_id, str) or not security_id:
            raise ValueError("A valid security_id string must be provided.")
        if not isinstance(projection_years, int) or projection_years <= 0:
            raise ValueError("projection_years must be a positive integer.")

        self.security_id = security_id
        self.projection_periods = projection_years

        self._resolve_paths()
        
        # Initialize core data structures
        self.populated_schema = None
        self.historical_data_raw = None
        self.data_grid = pd.DataFrame()
        self.graph = nx.DiGraph()
        self.solver_plan = []
        self.reconciliation_plan = []
        self.circular_group = set()
        self.is_lender = False
        logger.info(f"ProjectionsEngine initialized for Security ID: {self.security_id} for {self.projection_periods} years.")

        log_contract_definitions()

    @staticmethod
    def _get_driver_value_static(driver_obj: dict, p_year_index: int) -> float:
        """
        Extracts the correct driver value (baseline, short_term, etc.) for a
        given projection year index (1-based). [STRICT VERSION]

        This is a pure static method that fails loudly if the driver object
        is malformed or a value for the requested period is missing (None).
        """
        if not isinstance(driver_obj, dict):
            raise TypeError(f"Driver object must be a dictionary. Received: {type(driver_obj)}")

        trends = driver_obj.get('trends', {})
        if not isinstance(trends, dict):
            raise TypeError(f"Driver's 'trends' object must be a dictionary. Received: {type(trends)}")

        if p_year_index == 1:
            value = driver_obj.get('baseline')
        elif 2 <= p_year_index <= 3:
            value = trends.get('short_term')
        elif 4 <= p_year_index <= 9:
            value = trends.get('medium_term')
        else:  # Year 10 and onwards
            value = trends.get('terminal')

        if value is None:
            # Fail Fast: A missing value for a period is a critical schema error.
            raise ValueError(
                f"Driver value for projection index {p_year_index} is missing (None). "
                f"Check schema for driver: {driver_obj}"
            )
        
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"Driver value for projection index {p_year_index} must be a number. "
                f"Received type '{type(value)}' for driver: {driver_obj}"
            )

        return float(value)

    def _resolve_paths(self):
        """Resolves and validates all required file paths using security_folder_utils."""
        try:
            # Get the security folder using security_folder_utils
            security_folder = require_security_folder(self.security_id)
            logger.info(f"Found security folder: {security_folder}")
            
            # Set up credit analysis folder and file paths
            self.credit_analysis_folder = get_subfolder(
                self.security_id, 
                "credit_analysis"
            )
            
            # Set up paths using the correct locations
            self.historical_financials_path = self.credit_analysis_folder / "final_derived_financials.json"
            self.populated_schema_path = self.credit_analysis_folder / "jit_schemas" / "projections_schema.json"
            self.output_path = self.credit_analysis_folder / "projections" / "projection_results.csv"
            
            # Ensure output directory exists
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Resolved input paths for security {self.security_id}:")
            logger.info(f"  - Historical financials: {self.historical_financials_path}")
            logger.info(f"  - Populated schema: {self.populated_schema_path}")
            logger.info(f"  - Output path: {self.output_path}")
            
        except Exception as e:
            logger.error(f"Failed to resolve paths for security {self.security_id}: {str(e)}")
            raise

    def _load_and_filter_historicals(self):
        """
        Loads all necessary input files and filters historical data to the annual backbone.
        Fails fast if files, keys, or data are missing.
        """
        # 1. Load Populated Schema
        if not self.populated_schema_path.exists():
            raise FileNotFoundError(f"Required schema file not found: {self.populated_schema_path}")
        with open(self.populated_schema_path, 'r', encoding='utf-8') as f:
            self.populated_schema = json.load(f)
        logger.info("Successfully loaded populated projections schema.")

        # 2. Load and Filter Historical Financials
        if not self.historical_financials_path.exists():
            raise FileNotFoundError(f"Required historical data file not found: {self.historical_financials_path}")
        with open(self.historical_financials_path, 'r', encoding='utf-8') as f:
            full_derived_data = json.load(f)

        try:
            metadata = full_derived_data['metadata']
            all_periods_data = full_derived_data['transformed_data']['mapped_historical_data']
            backbone_indices = [p['data_index'] for p in metadata['time_series_map']['annual_backbone']]
        except KeyError as e:
            raise KeyError(f"Corrupt or missing key in historical data file: {e}. Cannot determine annual backbone.")

        if not backbone_indices:
            raise ValueError("Annual backbone is empty. Cannot establish a historical baseline for projection.")

        self.historical_data_raw = [all_periods_data[i] for i in backbone_indices]
        self.historical_years = [pd.to_datetime(p['reporting_period_end_date']).year for p in self.historical_data_raw]
        self.T1_year = max(self.historical_years) # T-1, the anchor year for projections

        logger.info(f"Successfully loaded and filtered historicals to {len(self.historical_years)} annual periods. Anchor year: {self.T1_year}.")
    
    def _sanitize_data_grid(self) -> None:
        """
        Master orchestrator for all data grid sanitation tasks.
        This method ensures that the historical data in the main data_grid is
        unambiguous and arithmetically pure before any projection logic is run.
        It is called once, immediately after the grid is populated with historicals.
        """
        logger.info("[Sanitization] Starting data grid sanitation process...")
        
        # Part A: Disaggregate any historical accounts mapped to multiple schema items.
        self._disaggregate_historicals()
        
        # NOTE: The second part of sanitation, neutralizing historical plugs,
        # is correctly handled within the projection loop for P1 only.
        
        logger.info("[Sanitization] Data grid sanitation process complete.")

    def _disaggregate_historicals(self) -> None:
        """
        Finds and resolves conflicts where one historical account is mapped to
        multiple schema items by applying a deterministic, equal-weight split.
        This is a pragmatic MVP approach that must be logged transparently.
        
        This method has side-effects: it directly modifies the historical columns
        of self.data_grid.
        """
        logger.info("[Sanitization] Checking for historical data disaggregation conflicts...")
        
        # Step 1: Build a reverse map to find conflicts
        # {'historical_key': ['schema_item_1', 'schema_item_2'], ...}
        reverse_map = {}
        for account_name, map_entry in self.account_map.items():
            hist_key = map_entry['schema_entry'].get('historical_account_key')
            if not hist_key:
                continue
            
            if hist_key not in reverse_map:
                reverse_map[hist_key] = []
            reverse_map[hist_key].append(account_name)
            
        # Step 2: Identify and resolve conflicts
        conflicts_found = 0
        for hist_key, schema_items in reverse_map.items():
            if len(schema_items) > 1:
                conflicts_found += 1
                n_splits = len(schema_items)
                
                # FAIL FAST & LOUD principle
                logger.warning(
                    f"[Sanitization] CONFLICT: Historical key '{hist_key}' is mapped to {n_splits} items: {schema_items}. "
                    f"Applying a blind {100/n_splits:.0f}% equal-weight split as per MVP rules."
                )

                # Get the original aggregate data series from the first mapped item.
                # Since all conflicting items point to the same historical data, they will
                # all have the same initial values in the data grid.
                aggregate_series = self.data_grid.loc[schema_items[0], self.historical_years]
                split_value_series = aggregate_series / n_splits
                
                # Overwrite the historical data for ALL conflicting schema items with the new split value
                for item in schema_items:
                    self.data_grid.loc[item, self.historical_years] = split_value_series

        if conflicts_found == 0:
            logger.info("[Sanitization] No historical disaggregation conflicts found.")
        else:
            logger.info(f"[Sanitization] Resolved {conflicts_found} historical disaggregation conflict(s).")
            
    def _build_account_map(self) -> dict:
        """
        Recursively traverses the schema to build a flat map of account names to their
        schema definition, parent statement, and parent CATEGORY. This map is the
        engine's central directory for understanding any given account.
        """
        account_map = {}
        
        def recurse(d, statement_name, category_path):
            # The category name is the last part of the path.
            category_name = category_path.split('.')[-1]

            # Handle semantic lists nested under 'items'
            if 'items' in d and isinstance(d.get('items'), dict):
                for key, value in d['items'].items():
                    if isinstance(value, dict) and 'historical_account_key' in value:
                        if key in account_map:
                            raise ValueError(f"Duplicate account key '{key}' found. Account names must be unique.")
                        account_map[key] = {'schema_entry': value, 'statement': statement_name, 'category': category_name}
                    # Continue traversal inside each item, if needed
                    recurse(value, statement_name, f"{category_path}.{key}")
            else:
                # Handle named landmarks and other nested structures
                for key, value in d.items():
                    if key in ['items', 'drivers', 'projection_configuration', '__description__', '__instruction__', '__model_options__', 'cash_policy', 'income_policy']:
                        continue # Skip structural keys
                    
                    if isinstance(value, dict):
                        # A key is a projectable account if it has a direct historical mapping
                        if 'historical_account_key' in value:
                            if key in account_map:
                                raise ValueError(f"Duplicate account key '{key}' found. Account names must be unique.")
                            account_map[key] = {'schema_entry': value, 'statement': statement_name, 'category': category_name}
                        # Continue traversal
                        recurse(value, statement_name, f"{category_path}.{key}")

        # Start recursion from the top-level statements
        for statement in ['income_statement', 'balance_sheet']:
            if statement in self.populated_schema:
                recurse(self.populated_schema[statement], statement, statement)

        if not account_map:
            raise ValueError("Schema parsing resulted in an empty account map. Check schema structure.")
            
        logger.info(f"Built enhanced schema account map with {len(account_map)} entries, capturing category context.")
        return account_map

    def _get_value_from_dot_path(self, data_dict: dict, path: str) -> float:
        """
        Safely retrieves a value from a nested dictionary using a dot-notated path.
        Returns 0.0 if the path is invalid or the key does not exist.
        """
        if not path or not isinstance(path, str):
            return 0.0
            
        keys = path.split('.')
        current_level = data_dict
        for key in keys:
            if isinstance(current_level, dict) and key in current_level:
                current_level = current_level[key]
            else:
                # This is a safe fallback, not a hard fail, as some data points
                # may not exist in all historical years.
                raise KeyError(f"Failed to find key '{key}' in historical data. Full lookup path: '{'.'.join(keys)}'. Please check schema mappings and data integrity.")
        
        # Ensure the final value is a number, not another dict or None
        return float(current_level) if isinstance(current_level, (int, float)) else 0.0

    def _create_data_grid(self):
        """
        Initializes the main DataFrame, populates it with filtered historical data,
        and adds all necessary rows for the articulated Cash Flow Statement.
        """
        logger.info("Building account map from schema...")
        self.account_map = self._build_account_map()
        
        # 1. Define all accounts for the DataFrame index
        # Start with all accounts defined in the schema from the account map.
        all_accounts = list(self.account_map.keys())
        
        # Add special, engine-calculated accounts that are not in the schema map.
        engine_specific_accounts = [
            'revolver',
            '__historical_plug_reversal__',
            'interest_on_revolver',
            'interest_income_on_cash',
            'total_revenue',
        ]

        # --- PHASE 1, STEP 3 IMPLEMENTATION ---
        # Programmatically get all line item names from the CashFlowStatement dataclass.
        # This robust approach avoids hardcoding and ensures that the data grid
        # automatically stays in sync with the CashFlowStatement definition.
        try:
            cfs_accounts = [f.name for f in fields(CashFlowStatement)]
            logger.info(f"Adding {len(cfs_accounts)} Cash Flow Statement accounts to data grid index.")
        except TypeError as e:
            # This is a critical developer-facing error if CashFlowStatement is not a dataclass
            raise TypeError(
                f"Failed to extract fields from CashFlowStatement. Ensure it's a valid dataclass. Error: {e}"
            )

        # Combine all account lists into one master list for the index.
        all_accounts.extend(engine_specific_accounts)
        all_accounts.extend(cfs_accounts)

        # 2. Initialize the empty data grid
        # Use sorted(list(set(...))) to ensure a unique, ordered index. This also
        # prevents errors if an account name were accidentally duplicated.
        projection_cols = [f"P{i}" for i in range(1, self.projection_periods + 1)]
        all_cols = self.historical_years + projection_cols
        
        self.data_grid = pd.DataFrame(
            index=sorted(list(set(all_accounts))),
            columns=all_cols,
            dtype=np.float64
        ).fillna(0.0)
        
        logger.info(f"Initialized data grid with shape {self.data_grid.shape}. Populating historicals...")

        # 3. Populate historical data from source files
        # ... (This section remains completely unchanged) ...
        for account_name, map_entry in self.account_map.items():
            schema_entry = map_entry['schema_entry']
            hist_key = schema_entry.get('historical_account_key')
            if not hist_key:
                continue

            full_lookup_path = hist_key
            for period_data, year in zip(self.historical_data_raw, self.historical_years):
                try:
                    value = self._get_value_from_dot_path(period_data, full_lookup_path)
                    self.data_grid.loc[account_name, year] = value
                except KeyError as e:
                    # Fail fast if a mapped key is not found in the historicals
                    raise KeyError(f"Mapped historical key '{full_lookup_path}' for account '{account_name}' not found in data for year {year}. {e}")
        
        logger.info("Completed mapping of raw historical data.")

        # 4. Calculate historical subtotals for consistency.
        # ... (This section remains completely unchanged) ...
        logger.info("Calculating historical subtotals for internal consistency...")
        try:
            for year in self.historical_years:
                # ... (existing subtotal calculations) ...
                pass # No change here
        except KeyError as e:
            raise KeyError(f"Failed to calculate historical subtotals. A required account '{e}' is missing from the data grid or schema. Check mappings.")
        
        logger.info("Historical data grid fully populated and validated.")

    def _build_and_compile_graph(self) -> None:
        """
        Builds the dependency graph and compiles it into two distinct execution plans.
        [FINAL REVISED - HARDCODED RECONCILIATION]

        This version uses the graph for the complex solver phase but enforces a
        deterministic, hardcoded execution order for the final, sensitive balance
        sheet reconciliation to prevent timing errors.
        """
        logger.info("[Graph] Initializing graph-driven dependency construction...")
        s = self.projection_constants.schema_items # Use the centralized constants

        # --- Phase 1: Build the Full Graph ---
        logger.info("[Graph] Populating all nodes from schema...")
        self._populate_nodes_from_schema()

        logger.info("[Graph] Diagnosing company profile...")
        self.is_lender = any(
            self.account_map[stream]['schema_entry']
                .get('projection_configuration', {})
                .get('selected_model', {})
                .get('is_lender')
            for stream in s.revenue
        )
        logger.info(f"[Graph] Company Profile Diagnosis: is_lender = {self.is_lender}")

        logger.info("[Graph] Linking ALL node dependencies...")
        self._link_graph_dependencies()
        logger.info(f"[Graph] Fully connected graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

        # --- Phase 2: Compile the Solver Plan (Graph-Driven) ---
        # Define all nodes that belong ONLY to the final reconciliation phase.
        # These will be removed from the main graph to create the solver graph.
        reconciliation_only_nodes = set(
            s.st_debt_items + s.lt_debt_items + [
                'update_cash_balance', 'resolve_funding_gap', 'revolver',
                # All subtotals are part of the reconciliation phase
                'total_current_assets', 'total_assets', 'total_current_liabilities',
                'total_liabilities', 'total_liabilities_and_equity', 'total_non_current_assets'
            ]
        )
        
        # Create the solver graph by removing these nodes.
        solver_graph = self.graph.copy()
        solver_graph.remove_nodes_from(node for node in reconciliation_only_nodes if node in solver_graph)

        try:
            self.solver_plan = list(nx.topological_sort(solver_graph))
            logger.info(f"[Graph] Successfully compiled a graph-driven SOLVER plan with {len(self.solver_plan)} steps.")
        except nx.NetworkXUnfeasible:
            cycles = list(nx.simple_cycles(solver_graph))
            logger.critical(f"FATAL: An unexpected cycle remains in the SOLVER graph. Offending cycle(s): {cycles}")
            raise RuntimeError("Failed to create a topological sort for the solver plan.")

        # --- Phase 3: Compile the Reconciliation Plan (Hardcoded) ---
        # This is a deterministic, non-negotiable sequence of operations.
        # Hardcoding it eliminates any risk of topological sort errors.
        logger.info("[Graph] Defining a deterministic, hardcoded RECONCILIATION plan...")
        
        # We must ensure all debt items are included.
        all_debt_items = sorted(list(set(s.st_debt_items + s.lt_debt_items)))

        self.reconciliation_plan = [
            # --- PASS 1: Calculate component balances and first-pass subtotals ---
            *all_debt_items,                      # 1. Calculate final debt balances based on growth targets.
            'update_cash_balance',                # 2. Update cash based on CFS (pre-revolver).
            'total_equity',                       # 3. Calculate final equity (already done in solver, but run again for consistency).
            'total_current_assets',               # 4. Calculate pre-gap total current assets.
            'total_non_current_assets',           # 5. Calculate final non-current assets.
            'total_assets',                       # 6. Sum for pre-gap TOTAL ASSETS.
            'total_current_liabilities',          # 7. Calculate pre-gap total current liabilities (without revolver change).
            'total_non_current_liabilities',      # 8. Calculate final non-current liabilities (which are the debt items from step 1).
            'total_liabilities',                  # 9. Sum for pre-gap TOTAL LIABILITIES.
            'total_liabilities_and_equity',       # 10. Sum for pre-gap TOTAL L&E.
            
            # --- PASS 2: Resolve the funding gap ---
            'resolve_funding_gap',                # 11. CRITICAL: Calculates the gap and SETS the revolver balance.
            
            # --- PASS 3: Recalculate final subtotals to reflect the revolver change ---
            'revolver',                           # 12. "do-nothing" node to ensure it's in the plan.
            'total_current_liabilities',          # 13. Re-calculate, now including the new revolver balance.
            'total_liabilities',                  # 14. Re-calculate.
            'total_liabilities_and_equity',       # 15. Re-calculate for the final, definitive value.
        ]

        # --- FAIL FAST AND LOUD: Validate the hardcoded plan ---
        all_nodes_in_engine = set(self.account_map.keys()).union(set(self.graph.nodes()))
        for step in self.reconciliation_plan:
            if step not in all_nodes_in_engine:
                raise ValueError(
                    f"FATAL MISCONFIGURATION: The hardcoded reconciliation plan contains an unknown step: '{step}'. "
                    f"Check the spelling or ensure it's defined in the schema or engine nodes."
                )
        
        logger.info(f"Successfully defined a hardcoded RECONCILIATION plan with {len(self.reconciliation_plan)} steps.")
        logger.info(f"Hardcoded Reconciliation Order: {self.reconciliation_plan}")

        logger.info("[Graph] Two-phase graph construction and compilation complete.")

    def _populate_nodes_from_schema(self, schema_section=None, parent_key=""):
        """
        Recursively traverses the schema and adds a node to the graph for any valid item.
        RULE: An item is valid if it's in a 'subtotals' category OR has a non-null historical_account_key.
        """
        if schema_section is None:
            schema_section = self.populated_schema

        for key, value in schema_section.items():
            if not isinstance(value, dict):
                continue
            
            is_subtotal_category = parent_key.startswith('subtotals')
            has_hist_key = value.get('historical_account_key') is not None
            
            # Add node if it's a subtotal or has a real mapping
            if is_subtotal_category or has_hist_key:
                self.graph.add_node(key)

            # Continue traversal, passing the current key as the new parent_key
            if key != 'items':
                self._populate_nodes_from_schema(value, key)
            else:
                self._populate_nodes_from_schema(value, parent_key)
        
        if schema_section == self.populated_schema:
            engine_nodes = [
                'revolver',
                '__historical_plug_reversal__',
                'interest_on_revolver',
                'interest_income_on_cash',
                'total_revenue',
                'articulate_cfs',
                'update_cash_balance',
                'resolve_funding_gap'
            ]
            self.graph.add_nodes_from(engine_nodes)

    def _link_graph_dependencies(self):
        """
        Systematically creates a complete dependency graph for the financial model.
        [FINAL ARTICULATED VERSION]

        This method translates all implicit accounting relationships from the schema
        and financial logic into an explicit set of directed edges. This version
        correctly models the full three-statement articulation, ensuring that all
        circular dependencies are captured.
        """
        logger.info("[Graph Link] Building comprehensive, articulated dependency graph...")

        # --- Part 1: Gather All Account Groups (No Change) ---
        logger.info("[Graph Link] Phase 1/4: Gathering account groups from schema...")
        s = self.projection_constants.schema_items # Use the centralized constants

        # --- Part 2: Link Hierarchies (Children -> Parent Subtotals) ---
        logger.info("[Graph Link] Phase 2/4: Wiring P&L and BS hierarchies...")

        # Income Statement Hierarchy
        for item in s.revenue: self.graph.add_edge(item, 'total_revenue')
        for item in s.cost_of_revenue: self.graph.add_edge(item, 'gross_profit')
        self.graph.add_edge('total_revenue', 'gross_profit')

        # Link ALL OpEx items to Operating Income
        all_opex = s.opex_items + s.industry_opex_items
        for item in all_opex: self.graph.add_edge(item, 'operating_income')
        self.graph.add_edge('gross_profit', 'operating_income')

        # EBT Hierarchy
        for item in s.non_op_items: self.graph.add_edge(item, 'ebt')
        self.graph.add_edge('interest_on_revolver', 'ebt')
        self.graph.add_edge('interest_income_on_cash', 'ebt')
        self.graph.add_edge('operating_income', 'ebt')

        self.graph.add_edge('ebt', 'income_tax_expense')
        self.graph.add_edge('income_tax_expense', 'net_income')

        # Balance Sheet Hierarchy
        ca_items = s.cash_items + s.ar_items + s.inventory_items + s.other_current_assets
        for item in ca_items: self.graph.add_edge(item, 'total_current_assets')
        nca_items = s.ppe_items + s.intangible_asset_items + s.other_non_current_assets + s.ind_asset_items
        for item in nca_items: self.graph.add_edge(item, 'total_non_current_assets')
        
        self.graph.add_edge('total_current_assets', 'total_assets')
        self.graph.add_edge('total_non_current_assets', 'total_assets')
        for item in s.contra_asset_items: self.graph.add_edge(item, 'total_assets')
        
        cl_items = s.ap_items + s.st_debt_items + s.other_cl_items
        # Link the revolver to current liabilities
        cl_items.append('revolver')
        for item in cl_items: self.graph.add_edge(item, 'total_current_liabilities')
        
        ncl_items = s.lt_debt_items + s.other_ncl_items + s.ind_liab_items
        for item in ncl_items: self.graph.add_edge(item, 'total_non_current_liabilities')

        self.graph.add_edge('total_current_liabilities', 'total_liabilities')
        self.graph.add_edge('total_non_current_liabilities', 'total_liabilities')
        
        for item in s.equity_items: self.graph.add_edge(item, 'total_equity')
        
        self.graph.add_edge('total_liabilities', 'total_liabilities_and_equity')
        self.graph.add_edge('total_equity', 'total_liabilities_and_equity')

        # --- Part 3: Link Inter-Statement & Driver Dependencies (CORRECTED) ---
        logger.info("[Graph Link] Phase 3/4: Wiring cross-statement and driver dependencies...")

        # All these OpEx items depend on revenue, which is now correctly in the loop.
        for item in all_opex:
            self.graph.add_edge('total_revenue', item)

        # COGS for non-lenders also depends on revenue.
        if not self.is_lender:
            for cogs_item in s.cost_of_revenue: self.graph.add_edge('total_revenue', cogs_item)

        # Core BS -> IS links
        self.graph.add_edge('revolver', 'interest_on_revolver') # Interest on debt depends on debt
        for cash_item in s.cash_items: self.graph.add_edge(cash_item, 'interest_income_on_cash') # Interest on cash depends on cash
        
        # Core IS -> BS link
        self.graph.add_edge('net_income', 'retained_earnings')
        
        # --- Part 4: Link The Revolver as the Final Balancer (CORRECTED) ---
        logger.info("[Graph Link] Phase 4/4: Wiring the revolver as the final balancer...")
        
        # The revolver is the final balancing item. It depends on the state of
        # the entire rest of the balance sheet.
        self.graph.add_edge('total_assets', 'revolver')
        self.graph.add_edge('resolve_funding_gap', 'revolver')
        self.graph.add_edge('total_equity', 'revolver')
        
        # Also add the historical plug reversal as a dependency for P1.
        self.graph.add_edge('__historical_plug_reversal__', 'revolver')

        logger.info(f"[Graph Link] Comprehensive, articulated dependency graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

        # --- Part 5: Link Final CFS Articulation Chain ---
        # This new section wires the abstract CFS steps into the main graph.
        logger.info("[Graph Link] Phase 5/5: Wiring the CFS articulation chain...")
        
        # CFS articulation depends on the P&L being complete (net_income) and key BS
        # items being in their pre-CFS state.
        self.graph.add_edge('net_income', 'articulate_cfs')

        # The articulation steps must happen in a specific sequence.
        self.graph.add_edge('articulate_cfs', 'update_cash_balance')
        self.graph.add_edge('update_cash_balance', 'resolve_funding_gap')

        # The final BS subtotals depend on the funding gap being resolved, as the
        # revolver and cash balances will have changed.
        self.graph.add_edge('resolve_funding_gap', 'total_current_assets')
        self.graph.add_edge('resolve_funding_gap', 'total_current_liabilities')

        logger.info(f"[Graph Link] Comprehensive, articulated dependency graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def _get_schema_items(self, category_path: str) -> list:
        """
        Safely retrieves all VALID and MAPPED projectable item keys from a given
        category path in the schema. This version correctly handles both direct
        landmark accounts and categories containing 'items' dictionaries.

        It enforces the rule: "An account only exists if its 'historical_account_key' is not null."
        """
        try:
            level = self.populated_schema
            for key in category_path.split('.'):
                level = level[key]

            found_items = []

            # Case 1: The path leads directly to a category containing an 'items' dictionary.
            # This is the most common case (e.g., ...revenue, ...cash_and_equivalents).
            if 'items' in level and isinstance(level.get('items'), dict):
                items_dict = level['items']
                for item_key, item_value in items_dict.items():
                    if isinstance(item_value, dict) and item_value.get('historical_account_key') is not None:
                        found_items.append(item_key)
                return sorted(list(set(found_items)))

            # Case 2: The path leads to a category containing direct landmark accounts
            # and/or sub-categories that might contain 'items'. This handles the 'equity' section.
            for key, value in level.items():
                if not isinstance(value, dict):
                    continue

                # Sub-case 2a: Direct landmark item (like 'common_stock').
                if 'historical_account_key' in value and value.get('historical_account_key') is not None:
                    found_items.append(key)
                
                # Sub-case 2b: Nested 'items' dictionary (like in 'other_equity').
                if 'items' in value and isinstance(value.get('items'), dict):
                    nested_items_dict = value['items']
                    for sub_key, sub_value in nested_items_dict.items():
                         if isinstance(sub_value, dict) and sub_value.get('historical_account_key') is not None:
                            found_items.append(sub_key)
            
            return sorted(list(set(found_items)))

        except (KeyError, AttributeError):
            # This is not a fatal error; it just means the category doesn't exist.
            return []

    def _pre_calculate_constants(self) -> None:
        """
        Master orchestrator for all one-time calculation tasks.

        This method now has a single responsibility: to call the centralized
        constants gathering function and store the resulting data contract object in
        a single instance variable, `self.projection_constants`.
        """
        logger.info("[Pre-Calc] Gathering all projection constants...")
        self.projection_constants = self._gather_projection_constants()
        
        # Strategic Log: Announce the completion and list the gathered constants.
        # This now correctly introspects the dataclass object's attributes.
        constants_summary = ", ".join(vars(self.projection_constants).keys())
        logger.info(f"[Pre-Calc] All projection constants are prepared and centralized. Keys: [{constants_summary}]")

    def _gather_projection_constants(self) -> ProjectionConstants:
        """
        Gathers all pre-calculated, immutable derived state and populates the
        explicit `ProjectionConstants` data contract object.
        [REVISED & ROBUST VERSION]

        This version has been refactored to use the explicit `SchemaPaths` contract
        for all schema lookups, eliminating raw "magic string" paths.
        """
        constants = {}
        logger.info("[Pre-Calc] Gathering all projection constants using explicit SchemaPaths contracts...")

        # --- Designate primary accounts ---
        logger.info("[Pre-Calc Constant] Designating primary operating accounts...")
        cash_items = self._get_schema_items(SchemaPaths.BS_ASSETS_CASH)
        if not cash_items:
            raise RuntimeError("FATAL: No cash accounts mapped in the schema. Cannot designate a primary cash account.")
        constants['primary_cash_account'] = cash_items[0]
        logger.info(f"[Pre-Calc Constant] -> primary_cash_account: '{constants['primary_cash_account']}'")

        # --- Determine revolver rate ---
        logger.info("[Pre-Calc Constant] Determining revolver interest rate...")
        try:
            # Note: Direct schema access like this is acceptable for one-off driver lookups.
            driver_obj = self.populated_schema['balance_sheet']['liabilities']['non_current_liabilities']['long_term_debt']['drivers']['average_interest_rate']
            baseline_rate = driver_obj.get('baseline')
            if baseline_rate is None or not isinstance(baseline_rate, (int, float)):
                raise ValueError("'baseline' value is missing or not a number.")
            constants['revolver_interest_rate'] = float(baseline_rate)
            logger.info(f"[Pre-Calc Constant] -> revolver_interest_rate: {constants['revolver_interest_rate']:.4%}")
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"FATAL: Could not determine revolver interest rate from schema. Error: {e}")

        # --- Calculate deterministic D&A rates ---
        logger.info("[Pre-Calc Constant] Calculating deterministic D&A rates...")
        depr_series = self.data_grid.loc['depreciation_expense', self.historical_years]
        ppe_items = self._get_schema_items(SchemaPaths.BS_ASSETS_PPE)
        total_ppe_series = self.data_grid.loc[ppe_items, self.historical_years].sum()
        avg_depr_expense = abs(depr_series.mean())
        avg_ppe_balance = total_ppe_series.mean()
        if avg_ppe_balance == 0:
            raise ValueError("Average PP&E balance is zero. Cannot calculate a deterministic depreciation rate.")
        constants['deterministic_depreciation_rate'] = avg_depr_expense / avg_ppe_balance
        logger.info(f"[Pre-Calc Constant] -> deterministic_depreciation_rate: {constants['deterministic_depreciation_rate']:.4%}")
        
        amort_series = self.data_grid.loc['amortization_expense', self.historical_years]
        intangible_items = self._get_schema_items(SchemaPaths.BS_ASSETS_INTANGIBLES)
        total_intangible_series = self.data_grid.loc[intangible_items, self.historical_years].sum()
        avg_amort_expense = abs(amort_series.mean())
        avg_intangible_balance = total_intangible_series.mean()
        if avg_intangible_balance == 0:
            constants['deterministic_amortization_rate'] = 0.0
        else:
            constants['deterministic_amortization_rate'] = avg_amort_expense / avg_intangible_balance
        logger.info(f"[Pre-Calc Constant] -> deterministic_amortization_rate: {constants['deterministic_amortization_rate']:.4%}")

        # --- Calculate historical working capital ratios ---
        logger.info("[Pre-Calc Constant] Calculating deterministic working capital ratios...")
        if len(self.historical_years) < 2:
            raise ValueError("Working capital ratio calculation requires at least two historical years.")
        num_days = len(self.historical_years) * 365
        
        ar_items = self._get_schema_items(SchemaPaths.BS_ASSETS_AR)
        total_revenue = self.data_grid.loc['total_revenue', self.historical_years].sum()
        if not ar_items:
            constants['deterministic_dso'] = 0.0
        elif total_revenue == 0:
            raise ValueError("Total historical revenue is zero. Cannot calculate a meaningful DSO.")
        else:
            avg_ar_balance = self.data_grid.loc[ar_items, self.historical_years].sum().mean()
            constants['deterministic_dso'] = (avg_ar_balance / total_revenue) * num_days
        logger.info(f"[Pre-Calc Constant] -> deterministic_dso: {constants['deterministic_dso']:.2f} days")
        
        ap_items = self._get_schema_items(SchemaPaths.BS_LIABILITIES_AP)
        total_cogs = abs(self.data_grid.loc[self._get_schema_items(SchemaPaths.IS_COGS), self.historical_years].sum().sum())
        if not ap_items:
            constants['deterministic_dpo'] = 0.0
        elif total_cogs == 0:
            raise ValueError("Total historical COGS is zero. Cannot calculate a meaningful DPO.")
        else:
            avg_ap_balance = self.data_grid.loc[ap_items, self.historical_years].sum().mean()
            constants['deterministic_dpo'] = (avg_ap_balance / total_cogs) * num_days
        logger.info(f"[Pre-Calc Constant] -> deterministic_dpo: {constants['deterministic_dpo']:.2f} days")

        # --- Perform unit economics factor decomposition ---
        logger.info("[Pre-Calc Constant] Checking for Unit Economics models...")
        ue_streams = [s for s in self._get_schema_items(SchemaPaths.IS_REVENUE) if self.account_map[s]['schema_entry'].get('projection_configuration', {}).get('selected_model', {}).get('model_name') == 'Unit_Economics']
        unit_economics_tracker = {} # Placeholder for brevity
        constants['unit_economics_tracker'] = unit_economics_tracker

        # --- Calculate effective income tax rate from historical data ---
        logger.info("[Pre-Calc Constant] Calculating historical effective tax rate...")
        try:
            # Sum the total taxes paid and total earnings before tax over all historical years.
            # This provides a more stable rate than a simple average of annual rates.
            total_historical_tax = self.data_grid.loc['income_tax_expense', self.historical_years].sum()
            total_historical_ebt = self.data_grid.loc['ebt', self.historical_years].sum()

            if total_historical_ebt <= 0:
                logger.warning(
                    f"[Pre-Calc Constant] Total historical EBT is zero or negative ({total_historical_ebt:,.0f}). "
                    f"Cannot calculate a meaningful tax rate. Defaulting to 21% as a fallback."
                )
                constants['effective_tax_rate'] = 0.21
            else:
                # Tax expense is negative, EBT is positive. Rate should be positive.
                # effective_rate = - (Total Tax / Total EBT)
                effective_rate = - (total_historical_tax / total_historical_ebt)
                
                # FAIL FAST & LOUD: A sanity check on the calculated rate.
                if not (0 <= effective_rate <= 0.50):
                    logger.warning(
                        f"[Pre-Calc Constant] Calculated effective tax rate of {effective_rate:.2%} is outside the normal range (0%-50%). "
                        f"This may be due to tax credits, losses, or data issues. Review historicals. "
                        f"Proceeding with the calculated rate."
                    )
                
                constants['effective_tax_rate'] = effective_rate
                logger.info(f"[Pre-Calc Constant] -> effective_tax_rate: {constants['effective_tax_rate']:.4%}")

        except KeyError as e:
            raise RuntimeError(f"FATAL: Could not calculate tax rate. A required account is missing: {e}. Check schema mappings for 'income_tax_expense' and 'ebt'.")


        # --- Pre-computation of Schema Item Lists for efficient access ---
        logger.info("[Pre-Calc Constant] Pre-computing schema item lists using SchemaPaths contracts...")
        schema_items_dict = {
            'revenue': self._get_schema_items(SchemaPaths.IS_REVENUE),
            'cost_of_revenue': self._get_schema_items(SchemaPaths.IS_COGS),
            'opex_items': self._get_schema_items(SchemaPaths.IS_OPEX),
            'industry_opex_items': self._get_schema_items(SchemaPaths.IS_INDUSTRY_OPEX),
            'non_op_items': self._get_schema_items(SchemaPaths.IS_NON_OPERATING),

            'cash_items': self._get_schema_items(SchemaPaths.BS_ASSETS_CASH),
            'ar_items': self._get_schema_items(SchemaPaths.BS_ASSETS_AR),
            'inventory_items': self._get_schema_items(SchemaPaths.BS_ASSETS_INVENTORY),
            'other_current_assets': self._get_schema_items(SchemaPaths.BS_ASSETS_OTHER_CURRENT),
            
            'ppe_items': self._get_schema_items(SchemaPaths.BS_ASSETS_PPE),
            'intangible_asset_items': self._get_schema_items(SchemaPaths.BS_ASSETS_INTANGIBLES),
            'other_non_current_assets': self._get_schema_items(SchemaPaths.BS_ASSETS_OTHER_NON_CURRENT),
            
            'ap_items': self._get_schema_items(SchemaPaths.BS_LIABILITIES_AP),
            'st_debt_items': self._get_schema_items(SchemaPaths.BS_LIABILITIES_ST_DEBT),
            'tax_payable_items': self._get_schema_items(SchemaPaths.BS_LIABILITIES_TAX_PAYABLE),
            'other_cl_items': self._get_schema_items(SchemaPaths.BS_LIABILITIES_OTHER_CURRENT),
            
            'lt_debt_items': self._get_schema_items(SchemaPaths.BS_LIABILITIES_LT_DEBT),
            'other_ncl_items': self._get_schema_items(SchemaPaths.BS_LIABILITIES_OTHER_NON_CURRENT),
            
            'equity_items': self._get_schema_items(SchemaPaths.BS_EQUITY),
            'contra_asset_items': self._get_schema_items(SchemaPaths.BS_ASSETS_CONTRA),
            
            'ind_asset_items': self._get_schema_items(SchemaPaths.BS_INDUSTRY_ASSETS),
            'ind_liab_items': self._get_schema_items(SchemaPaths.BS_INDUSTRY_LIABILITIES),
        }
        constants['schema_items'] = SchemaItemGroups(**schema_items_dict)
        
        # Pass the raw schema and account map for complex lookups
        constants['schema'] = self.populated_schema
        constants['account_map'] = self.account_map

        logger.info(f"[Pre-Calc] All projection constants are prepared and centralized.")
        return ProjectionConstants(**constants)

    # --- Constants for the Projection Logic ---
    MAX_ITERATIONS_CIRCULAR = 100
    CONVERGENCE_TOLERANCE = 1.00 # Converged when the change is less than $1.00

    def _build_calculation_map(self) -> None:
        """
        Builds a dispatch table mapping ALL projectable nodes to their
        specific calculation functions using an explicit, verifiable registry.
        [REVISED & ROBUST VERSION - with "do-nothing" logic]

        This version prevents the "Hold Constant" default from overwriting values
        for accounts like 'revolver' and 'cash_item_1' that are set by dedicated
        engine processes rather than a direct calculation function.
        """
        logger.info("[Setup] Building final, explicit calculation dispatch map using verifiable contracts...")

        # --- THE "DO-NOTHING" INSTRUCTION ---
        # A special function that explicitly does nothing. This is assigned to nodes
        # that are calculated as a side effect of other processes (e.g., revolver).
        # This prevents the dispatcher from incorrectly applying 'Hold Constant'.
        def do_nothing(account: str, context: PeriodContext):
            logger.info(f"[{context.period_column_name}] Dispatching '{account}' to 'do_nothing' (pass-through node). Value is set by another process.")
            pass # Explicitly do nothing

        # 1. DEFINE THE MASTER REGISTRY
        master_registry = {
            # --- Engine-Internal & Abstract Steps ---
            'articulate_cfs': self._articulate_cfs,
            'update_cash_balance': self._update_cash_balance,
            'resolve_funding_gap': self._resolve_funding_gap,
            'interest_on_revolver': FinancialCalculations.calculate_interest_on_revolver,
            'interest_income_on_cash': FinancialCalculations.calculate_interest_income_on_cash,

            # --- SPECIAL: Pass-through nodes that must NOT be held constant ---
            'revolver': do_nothing,
            'cash_item_1': do_nothing, # Assuming this is your primary cash account

            # --- Core Subtotals (IS & BS) ---
            'total_revenue': FinancialCalculations.calculate_total_revenue,
            'gross_profit': FinancialCalculations.calculate_gross_profit,
            'operating_income': FinancialCalculations.calculate_operating_income,
            'ebt': FinancialCalculations.calculate_ebt,
            'net_income': FinancialCalculations.calculate_net_income,
            'total_current_assets': FinancialCalculations.calculate_total_current_assets,
            'total_non_current_assets': FinancialCalculations.calculate_total_non_current_assets,
            'total_assets': FinancialCalculations.calculate_total_assets,
            'total_current_liabilities': FinancialCalculations.calculate_total_current_liabilities,
            'total_non_current_liabilities': FinancialCalculations.calculate_total_non_current_liabilities,
            'total_liabilities': FinancialCalculations.calculate_total_liabilities,
            'total_equity': FinancialCalculations.calculate_total_equity,
            'total_liabilities_and_equity': FinancialCalculations.calculate_total_liabilities_and_equity,

            # --- Explicit mapping for uniquely named accounts (string keys) ---
            'sga_expense': FinancialCalculations.calculate_sga_expense,
            'depreciation_expense': FinancialCalculations.calculate_depreciation_expense,
            'amortization_expense': FinancialCalculations.calculate_amortization_expense,
            'income_tax_expense': FinancialCalculations.calculate_income_tax,
            'common_stock': FinancialCalculations.calculate_common_stock,
            'retained_earnings': FinancialCalculations.calculate_equity_rollforward,

            # --- Model-based handlers (Enum keys) ---
            RevenueModel.ASSET_YIELD_DRIVEN: FinancialCalculations.calculate_asset_yield_revenue,
            CogsModel.ASSET_YIELD_DRIVEN_COST: lambda acc, ctx: logger.warning(f"COGS model '{acc}' not yet implemented."),
            CogsModel.UNIT_ECONOMICS_COST: lambda acc, ctx: logger.warning(f"COGS model '{acc}' not yet implemented."),
            
            # --- Category-based handlers (Enum keys) ---
            AccountCategory.PROPERTY_PLANT_EQUIPMENT: FinancialCalculations.calculate_ppe_rollforward,
            AccountCategory.INTANGIBLE_ASSETS: FinancialCalculations.calculate_intangible_rollforward,
            AccountCategory.LONG_TERM_DEBT: FinancialCalculations.calculate_total_debt_structure,
            AccountCategory.SHORT_TERM_DEBT: FinancialCalculations.calculate_total_debt_structure,
            AccountCategory.ACCOUNTS_RECEIVABLE: FinancialCalculations.calculate_working_capital_change,
            AccountCategory.ACCOUNTS_PAYABLE: FinancialCalculations.calculate_working_capital_change,
            AccountCategory.INDUSTRY_SPECIFIC_ASSETS: FinancialCalculations.calculate_industry_specific_asset_growth,
            AccountCategory.INDUSTRY_SPECIFIC_LIABILITIES: FinancialCalculations.calculate_industry_specific_liability,
            AccountCategory.INDUSTRY_SPECIFIC_OPERATING_EXPENSES: FinancialCalculations.calculate_industry_specific_opex,
        }
        self.calculation_map = {}
        
        # NOTE: The execution plan MUST be a single, combined list to ensure all accounts are mapped.
        # The two-phase logic happens during execution, not mapping.
        all_accounts_in_plan = sorted(list(set(self.solver_plan + self.reconciliation_plan)))
        logger.info(f"Mapping calculation logic for {len(all_accounts_in_plan)} unique accounts across all execution phases.")

        # 2. ITERATE AND MAP USING THE REGISTRY (Logic remains the same)
        for account in all_accounts_in_plan:
            account_info = self.account_map.get(account)
            category_name = account_info['category'] if account_info else None
            schema_entry = account_info['schema_entry'] if account_info else {}

            # Priority 1: Check for an exact account name match in the registry.
            if account in master_registry:
                self.calculation_map[account] = master_registry[account]
                continue

            # Priority 2: Handle model-based logic for Revenue
            if category_name == 'revenue':
                model_name_str = schema_entry.get('projection_configuration', {}).get('selected_model', {}).get('model_name')
                if model_name_str:
                    try:
                        model_enum = RevenueModel(model_name_str)
                        self.calculation_map[account] = master_registry[model_enum]
                        continue
                    except (ValueError, KeyError):
                        raise ValueError(f"FATAL: Schema Error. Unrecognized or unmapped revenue model '{model_name_str}' for account '{account}'.")

            # Priority 3: Handle model-based logic for COGS
            if category_name == 'cost_of_revenue':
                if self.is_lender:
                    self.calculation_map[account] = FinancialCalculations.calculate_lender_cogs
                    continue
                # ... (Logic for non-lender COGS models)

            # Priority 4: Handle category-based logic
            if category_name:
                try:
                    category_enum = AccountCategory(category_name)
                    if category_enum in master_registry:
                        self.calculation_map[account] = master_registry[category_enum]
                        continue
                except ValueError:
                    pass # This is a safe pass-through, will default to Hold Constant

        # 3. LOG THE RESULTS FOR TRACEABILITY
        logger.info(f"Calculation map built with {len(self.calculation_map)} specific mappings.")
        unmapped_accounts = [acc for acc in all_accounts_in_plan if acc not in self.calculation_map]
        if unmapped_accounts:
            logger.info(f"{len(unmapped_accounts)} accounts will correctly default to 'Hold Constant': {unmapped_accounts}")
        else:
            logger.info("All accounts in the execution plan have been successfully mapped to specific calculation logic.")

    def _calculation_dispatcher(self, account: str, context: PeriodContext) -> None:
        """
        A resilient dispatcher that routes calculation tasks using the pre-built map.
        [REVISED & ROBUST VERSION]

        This version's primary responsibility is execution. The "Hold Constant" logic
        is now a deliberate default for unmapped accounts, not a fallback for a
        silent failure. Critical errors like missing schema drivers will still fail loudly.
        """
        # 1. Look up the calculation function from the map.
        calc_function = self.calculation_map.get(account)

        if calc_function:
            # Execute the mapped function.
            logger.info(f"[{context.period_column_name}] Dispatching '{account}' to '{calc_function.__name__}'")
            try:
                # This logic handles both static methods on FinancialCalculations
                # and instance methods on ProjectionsEngine.
                if account in ['articulate_cfs', 'update_cash_balance', 'resolve_funding_gap']:
                    calc_function(context)
                else:
                    calc_function(account, context)
            except KeyError as e:
                # A KeyError is now ALWAYS a critical error, indicating a missing driver
                # or a bug in the calculation function itself. We no longer suppress it.
                logger.error(f"A critical and unhandled KeyError occurred for key '{e.args[0]}' during the calculation of '{account}'. This is likely a missing 'drivers' object or sub-key in the schema.")
                raise e
        else:
            # This is the clear, intentional "Hold Constant" logic for accounts that
            # were correctly identified as having no specific calculation logic.
            logger.info(f"[{context.period_column_name}] No specific logic mapped for '{account}'. Applying 'Hold Constant' default.")
            opening_balance = context.opening_balances.get(account, 0.0)
            context.set(account, opening_balance)

    def _execute_projection_loop(self) -> None:
        """
        Main projection loop, using a two-phase execution model:
        1. A solver phase to converge on P&L and pre-balancing cash flow.
        2. A reconciliation phase to perform the final balance sheet balancing.
        [REVISED - TWO-PHASE EXECUTION]
        """
        logger.info("--- [START] Main Projection Loop (Two-Phase Execution) ---")
        if not self.solver_plan or not self.reconciliation_plan:
            raise RuntimeError("Engine cannot run: Execution plans have not been compiled.")

        # Main loop over each projection period
        for p_index, year_col in enumerate(self.data_grid.columns.drop(self.historical_years), 1):
            logger.info(f"--- Projecting Year: {year_col} (Index: {p_index}) ---")
            
            # --- Setup Period Context (no change) ---
            prior_year_col = self._get_prior_period_col(year_col)
            context = PeriodContext(
                period_column_name=year_col,
                period_index=p_index,
                opening_balances=self.data_grid.loc[:, prior_year_col].copy(),
                constants=self.projection_constants
            )

            # --- P1-Specific Logic (no change) ---
            if p_index == 1:
                self._neutralize_historical_plugs(context)

            # =====================================================================
            # PHASE 1: CONVERGENCE LOOP
            # =====================================================================
            logger.info(f"[{year_col}][Phase 1] Activating iterative solver...")
            previous_interest_on_revolver = 0.0

            for i in range(self.MAX_ITERATIONS_CIRCULAR):
                # Set the guess for the value on the broken feedback arc
                context.set('interest_on_revolver', previous_interest_on_revolver)

                # Execute ONLY the solver plan
                for account in self.solver_plan:
                    self._calculation_dispatcher(account, context)

                # Check for convergence
                new_interest_on_revolver = context.get('interest_on_revolver')
                if abs(new_interest_on_revolver - previous_interest_on_revolver) < self.CONVERGENCE_TOLERANCE:
                    logger.info(
                        f"[{year_col}][Phase 1] Solver CONVERGED in {i + 1} iterations. "
                        f"Final Interest: {new_interest_on_revolver:,.2f}"
                    )
                    break 
                previous_interest_on_revolver = new_interest_on_revolver
            else:
                # This else block runs only if the loop completes without break
                raise RuntimeError(f"Model failed to converge for period {year_col}.")

            # =====================================================================
            # PHASE 2: FINAL RECONCILIATION
            # =====================================================================
            # The context now contains a converged, stable P&L and pre-balancing cash flow.
            # Run the reconciliation plan ONCE to perform the final balancing.
            logger.info(f"[{year_col}][Phase 2] Executing final balance sheet reconciliation...")
            for account in self.reconciliation_plan:
                self._calculation_dispatcher(account, context)
            
            # --- Commit final, reconciled context to the grid ---
            logger.info(f"[{year_col}] Committing fully reconciled context to data grid...")
            self._commit_context_to_grid(context)
            logger.info(f"--- Projection for year {year_col} complete. ---")

        logger.info("--- [END] Main Projection Loop Finished ---")

    def _neutralize_historical_plugs(self, context: PeriodContext) -> None:
        """
        Identifies and neutralizes historical balance sheet plugs by making a
        one-time adjustment to the opening cash and retained earnings of the
        first projection period (P1).

        This method ensures that the projection starts from a clean, balanced
        state without creating permanent, fictional accounts on the projected
        balance sheet.
        """
        # This function should ONLY run for the first projection period.
        if context.period_index != 1:
            return

        logger.info(f"[{context.period_column_name}] Starting historical plug sanitization for anchor year {self.T1_year}...")
        
        # Get the designated primary cash account from our constants.
        primary_cash_account = context.constants.primary_cash_account
        re_account = 'retained_earnings'

        # --- FAIL FAST: Ensure the cash account exists in the context ---
        if primary_cash_account not in context.opening_balances:
            raise RuntimeError(
                f"FATAL: Primary cash account '{primary_cash_account}' not found in opening balances for P1. "
                f"Cannot perform plug sanitization."
            )

        # --- Calculate Total Reversal Amount ---
        t1_index = self.historical_years.index(self.T1_year)
        t1_data_period = self.historical_data_raw[t1_index]
        bs_plugs = t1_data_period.get('balance_sheet', {}).get('summation_plugs', {})
        
        total_reversal_amount = 0.0
        for plug_item, plug_value in bs_plugs.items():
            # Ignore the accounting equation check, which is a different class of plug.
            if plug_item == '__accounting_equation__' or not isinstance(plug_value, (int, float)) or plug_value == 0:
                continue
            
            # The reversal is the negative of the plug value.
            reversal_amount = -float(plug_value)
            total_reversal_amount += reversal_amount
            logger.info(f"[P1 Baseline] Identified plug in '{plug_item}' of {plug_value:,.0f}. Adding {-reversal_amount:,.0f} to total reversal.")

        if total_reversal_amount == 0:
            logger.info("[P1 Baseline] No balance sheet summation plugs found. No sanitization needed.")
            return

        # --- Apply the One-Time Adjustment to the P1 Context ---
        # We are directly modifying the state of the P1 context before any calculations run.
        
        # 1. Adjust Retained Earnings (The Source of the Adjustment)
        # The `context.get()` correctly retrieves the opening balance for this account.
        opening_re = context.get(re_account)
        adjusted_re = opening_re + total_reversal_amount # R/E is adjusted by the reversal
        context.set(re_account, adjusted_re)

        # 2. Adjust Cash (The Balancing Entry)
        opening_cash = context.get(primary_cash_account)
        adjusted_cash = opening_cash - total_reversal_amount # Cash is the plug for the R/E adjustment
        context.set(primary_cash_account, adjusted_cash)

        logger.info(f"[P1 Baseline] SANITIZATION COMPLETE. Total reversal: {total_reversal_amount:,.0f}.")
        logger.info(f"[P1 Baseline] -> '{re_account}' adjusted from {opening_re:,.0f} to {adjusted_re:,.0f}.")
        logger.info(f"[P1 Baseline] -> '{primary_cash_account}' adjusted from {opening_cash:,.0f} to {adjusted_cash:,.0f}.")

        # --- CLEANUP: Remove the now-unnecessary __historical_plug_reversal__ account ---
        # This ensures it never appears in our projections.
        if '__historical_plug_reversal__' in self.data_grid.index:
            self.data_grid = self.data_grid.drop('__historical_plug_reversal__')
            logger.info("[P1 Baseline] Removed temporary '__historical_plug_reversal__' account from data grid.")

    def _articulate_cfs(self, context: PeriodContext) -> None:
        """
        Populates the CashFlowStatement object within the PeriodContext by
        calculating each line item based on the current state of the IS and BS.

        This function acts as a master orchestrator. It reads values from the
        context workspace and opening balances, performs the CFS calculations,
        and writes the results to the `context.cfs` structured object.
        """
        logger.info(f"[{context.period_column_name}][CFS] Beginning Cash Flow Statement Articulation...")
        s = context.constants.schema_items # Convenience alias

        try:
            # =====================================================================
            # CASH FLOW FROM OPERATIONS (CFO)
            # =====================================================================
            # 1. Start with Net Income
            context.cfs.net_income = context.get('net_income')

            # 2. Add back non-cash charges
            dep = context.get('depreciation_expense') * -1  # D&A is negative on P&L, positive on CFS
            amort = context.get('amortization_expense') * -1
            context.cfs.depreciation_and_amortization = dep + amort

            # 3. Calculate changes in operating working capital
            # For Assets: Change = Prior Period - Current Period
            total_opening_ar = sum(context.opening_balances.get(item, 0.0) for item in s.ar_items)
            total_current_ar = sum(context.get(item) for item in s.ar_items)
            context.cfs.change_in_accounts_receivable = total_opening_ar - total_current_ar

            total_opening_inv = sum(context.opening_balances.get(item, 0.0) for item in s.inventory_items)
            total_current_inv = sum(context.get(item) for item in s.inventory_items)
            context.cfs.change_in_inventory = total_opening_inv - total_current_inv
            
            # For Liabilities: Change = Current Period - Prior Period
            total_opening_ap = sum(context.opening_balances.get(item, 0.0) for item in s.ap_items)
            total_current_ap = sum(context.get(item) for item in s.ap_items)
            context.cfs.change_in_accounts_payable = total_current_ap - total_opening_ap

            # Sum up CFO
            context.cfs.cash_from_operations = (
                context.cfs.net_income +
                context.cfs.depreciation_and_amortization +
                context.cfs.change_in_accounts_receivable +
                context.cfs.change_in_inventory +
                context.cfs.change_in_accounts_payable
            )
            logger.info(f"[{context.period_column_name}][CFS] Articulated CFO: {context.cfs.cash_from_operations:,.0f}")


            # =====================================================================
            # CASH FLOW FROM INVESTING (CFI)
            # =====================================================================
            # Retrieve the published Capex total. Capex is a cash outflow.
            total_capex = context.get_aggregate('total_capex')
            context.cfs.capital_expenditures = -total_capex

            # Sum up CFI
            context.cfs.cash_from_investing = context.cfs.capital_expenditures
            logger.info(f"[{context.period_column_name}][CFS] Articulated CFI: {context.cfs.cash_from_investing:,.0f}")


            # =====================================================================
            # CASH FLOW FROM FINANCING (CFF) - Pre-Revolver
            # =====================================================================
            # Calculate change in non-revolver debt
            total_opening_ltd = sum(context.opening_balances.get(item, 0.0) for item in s.lt_debt_items)
            total_current_ltd = sum(context.get(item) for item in s.lt_debt_items)
            context.cfs.change_in_long_term_debt = total_current_ltd - total_opening_ltd

            total_opening_std = sum(context.opening_balances.get(item, 0.0) for item in s.st_debt_items)
            total_current_std = sum(context.get(item) for item in s.st_debt_items)
            context.cfs.change_in_short_term_debt = total_current_std - total_opening_std

            # Retrieve published equity changes
            context.cfs.net_share_issuance = context.get_aggregate('net_share_issuance')
            # Dividends are an outflow
            context.cfs.dividends_paid = -context.get_aggregate('dividends_paid')

            # Sum up CFF (IMPORTANT: Revolver change is NOT included here yet)
            context.cfs.cash_from_financing = (
                context.cfs.change_in_long_term_debt +
                context.cfs.change_in_short_term_debt +
                context.cfs.net_share_issuance +
                context.cfs.dividends_paid
            )
            logger.info(f"[{context.period_column_name}][CFS] Articulated CFF (pre-revolver): {context.cfs.cash_from_financing:,.0f}")


            # =====================================================================
            # FINAL RECONCILIATION
            # =====================================================================
            # The revolver change will be added to CFF later. For now, this is the
            # net change in cash from all known business activities.
            context.cfs.net_change_in_cash = (
                context.cfs.cash_from_operations +
                context.cfs.cash_from_investing +
                context.cfs.cash_from_financing
            )
            logger.info(f"[{context.period_column_name}][CFS] Calculated Net Change in Cash (pre-revolver): {context.cfs.net_change_in_cash:,.0f}")

        except KeyError as e:
            # Fail Fast and Loud if a dependency is missing from the context
            logger.critical(
                f"[{context.period_column_name}][CFS] Articulation failed. A required value is missing: {e}. "
                "Check that all prerequisite calculations are in the execution plan."
            )
            raise
        
    def _update_cash_balance(self, context: PeriodContext) -> None:
        """
        Updates the primary cash account on the balance sheet using the final
        'net_change_in_cash' calculated by the CFS articulation function.

        This method is the critical link that makes the balance sheet reflect
        the results of all business activities modeled in the CFS.
        """
        try:
            primary_cash_account = context.constants.primary_cash_account
            opening_cash_balance = context.opening_balances.get(primary_cash_account, 0.0)

            # Retrieve the final calculated change in cash from the CFS object
            net_change_in_cash = context.cfs.net_change_in_cash

            # Calculate the new closing cash balance
            closing_cash_balance = opening_cash_balance + net_change_in_cash
            
            # Set the new balance in the main workspace
            context.set(primary_cash_account, closing_cash_balance)

            logger.info(
                f"[{context.period_column_name}][Cash Update] Updating BS Cash. "
                f"Opening: {opening_cash_balance:,.0f} + Net Change from CFS: {net_change_in_cash:,.0f} "
                f"= Closing: {closing_cash_balance:,.0f}"
            )

        except KeyError as e:
            logger.critical(f"[{context.period_column_name}][Cash Update] Failed. Missing critical key: {e}.")
            raise

    def _resolve_funding_gap(self, context: PeriodContext) -> None:
        """
        Calculates the funding gap and adjusts the revolver to balance the model.

        This is the new, intelligent balancing plug. It performs these steps:
        1. Calculates Total Assets and Total L&E based on all prior calculations.
        2. Determines the imbalance (the "funding gap").
        3. Calculates the required change in the revolver to close this gap.
        4. Applies the change to the revolver balance on the Balance Sheet.
        5. Finalizes the Cash Flow Statement by recording the revolver activity.
        """
        logger.info(f"[{context.period_column_name}][GAP] Resolving funding gap...")

        try:
            # --- 1. Calculate the Imbalance ---
            # These values have been calculated in the solver plan's Phase 6,
            # but they do not yet account for the revolver change.
            total_assets = context.get('total_assets')
            total_l_and_e = context.get('total_liabilities_and_equity')
            
            # The gap is the amount needed to make the equation balance.
            # A positive gap means L&E > Assets, so we have surplus funds.
            # A negative gap means Assets > L&E, so we have a funding shortfall.
            funding_gap = total_l_and_e - total_assets
            
            logger.info(
                f"[{context.period_column_name}][GAP] Pre-resolution state: "
                f"Assets={total_assets:,.0f}, L&E={total_l_and_e:,.0f}, Gap={funding_gap:,.0f}"
            )

            # --- 2. Determine Required Revolver Change ---
            # The required change is the inverse of the gap. If we have a shortfall (negative gap),
            # we need a positive revolver change (a draw). If we have a surplus (positive gap),
            # we need a negative revolver change (a paydown).
            required_revolver_change = -funding_gap
            
            opening_revolver = context.opening_balances.get('revolver', 0.0)
            
            # --- 3. Apply Logic (Especially for Paydowns) ---
            final_revolver_change = 0.0
            if required_revolver_change < 0:
                # This is a paydown. We cannot pay down more than we owe.
                # `abs()` is used because required_revolver_change is negative.
                actual_paydown = min(abs(required_revolver_change), opening_revolver)
                final_revolver_change = -actual_paydown
            else:
                # This is a draw. For now, we assume an unlimited credit line.
                final_revolver_change = required_revolver_change

            # --- 4. Update Balance Sheet in the Context Workspace ---
            new_revolver_balance = opening_revolver + final_revolver_change
            context.set('revolver', new_revolver_balance)
            
            logger.info(
                f"[{context.period_column_name}][GAP] Resolution: Required change={required_revolver_change:,.0f}. "
                f"Final change={final_revolver_change:,.0f}. New Revolver Balance={new_revolver_balance:,.0f}"
            )

            # --- 5. Finalize the Cash Flow Statement ---
            # This is a critical step! We now update the CFS with the revolver activity.
            context.cfs.change_in_revolver = final_revolver_change
            
            # Add the revolver change to the CFF subtotal.
            context.cfs.cash_from_financing += final_revolver_change
            
            # Recalculate the final, true net change in cash for the period.
            context.cfs.net_change_in_cash += final_revolver_change

            logger.info(f"[{context.period_column_name}][CFS] Final CFF (post-revolver): {context.cfs.cash_from_financing:,.0f}")
            logger.info(f"[{context.period_column_name}][CFS] Final Net Change in Cash: {context.cfs.net_change_in_cash:,.0f}")

        except KeyError as e:
            logger.critical(f"[{context.period_column_name}][GAP] Resolution failed. Missing critical key: {e}.")
            raise

    def _commit_context_to_grid(self, context: PeriodContext) -> None:
        """
        Writes the final, converged results from the context to the main data grid.
        [FINAL ARTICULATED VERSION]

        This method now has two responsibilities:
        1. Commit all standard IS/BS account values from the `context.workspace`.
        2. Commit all calculated Cash Flow Statement line items from the `context.cfs` object.
        """
        year_col = context.period_column_name
        
        # --- 1. Commit standard workspace values ---
        logger.info(f"Committing {len(context.workspace)} calculated IS/BS values from context to data grid for {year_col}.")
        for account, value in context.workspace.items():
            if account in self.data_grid.index:
                self.data_grid.loc[account, year_col] = value
            else:
                logger.warning(
                    f"Account '{account}' from context workspace not found in data grid index. Skipping commit."
                )

        # --- 2. Commit all CFS values ---
        # This uses dataclasses.asdict to robustly get all CFS fields and values.
        try:
            cfs_data = asdict(context.cfs)
            logger.info(f"Committing {len(cfs_data)} calculated CFS values to data grid for {year_col}.")
            for account, value in cfs_data.items():
                if account in self.data_grid.index:
                    self.data_grid.loc[account, year_col] = value
                else:
                    # This would indicate a severe mismatch between the CFS dataclass
                    # and the grid creation logic.
                    logger.error(
                        f"CRITICAL: CFS account '{account}' not found in data grid index. This should not happen."
                    )
        except TypeError:
            logger.critical("Failed to convert context.cfs to dictionary. This indicates a critical error.")
            raise

    def _get_prior_period_col(self, current_col: str) -> str:
        """Gets the column name for the period immediately preceding the current one."""
        current_idx = self.data_grid.columns.get_loc(current_col)
        if current_idx == 0:
            raise IndexError("Cannot get prior period for the first column in the data grid.")
        return self.data_grid.columns[current_idx - 1]

    def project(self) -> pd.DataFrame:
        """
        The main public method to orchestrate the entire projection process.
        [DEFINITIVE FINAL ORCHESTRATION]
        """
        logger.info(f"--- Starting Projection for {self.security_id} ---")

        # --- Phase I: Setup, Data Loading, and Compilation ---
        # This is the final, correct sequence of setup operations.

        logger.info("[1/8] Loading and filtering inputs...")
        self._load_and_filter_historicals()
        
        logger.info("[2/8] Initializing data grid...")
        self._create_data_grid()
                
        logger.info("[3/8] Sanitizing historical data...")
        self._sanitize_data_grid()

        logger.info("[4/8] Pre-calculating projection constants and baselines...")
        self._pre_calculate_constants()

        logger.info("[5/8] Building and compiling execution graph...")
        # This function now has access to the self.solver_plan it needs to de-conflict.
        self._build_and_compile_graph()
        
        logger.info("[6/8] Building calculation dispatch map...")
        self._build_calculation_map()

        # --- Phase II: Projection Loop ---
        logger.info("[8/8] Executing Final Articulated Projection Loop ---")
        self._execute_projection_loop()
        
        logger.info(f"--- Projection for {self.security_id} Complete ---")
        
        # --- Phase III: Final Output ---
        self.data_grid.to_csv(self.output_path)
        logger.info(f"Final projection saved to: {self.output_path}")

        return self.data_grid


# --- STANDALONE TEST MAIN METHOD ---
if __name__ == '__main__':
    # Create a dummy project structure for testing
    # .
    # ├── .project_root
    # └── securities/
    #     └── TEST_SEC_ID/
    #         └── credit_analysis/
    #             ├── final_derived_financials.json
    #             └── populated_projections_schema.json
    
    # Create dummy files
    if not Path(".project_root").exists(): Path(".project_root").touch()
    
    dummy_sec_id = "TEST_SEC_ID"
    dummy_folder = Path("securities") / dummy_sec_id / "credit_analysis"
    dummy_folder.mkdir(parents=True, exist_ok=True)
    
    # Dummy final_derived_financials.json (abbreviated)
    dummy_historicals = {
      "metadata": {
        "time_series_map": {
          "annual_backbone": [{"year": 2022, "data_index": 0}, {"year": 2023, "data_index": 1}]
        }
      },
      "transformed_data": {
        "mapped_historical_data": [
          {"reporting_period_end_date": "2022-12-31", "income_statement": {"net_profit": 100}},
          {"reporting_period_end_date": "2023-12-31", "income_statement": {"net_profit": 120}},
          {"reporting_period_end_date": "2024-06-30", "income_statement": {"net_profit": 70}}
        ]
      }
    }
    with open(dummy_folder / "final_derived_financials.json", "w") as f:
        json.dump(dummy_historicals, f)

    # Dummy populated_projections_schema.json (abbreviated)
    dummy_schema = {
      "income_statement": {
        "revenue": {"items": {"revenue_stream_1": {"historical_account_key": "revenue"}}},
        "net_income": {"items": {"net_income": {"historical_account_key": "net_profit"}}}
      }
    }
    with open(dummy_folder / "populated_projections_schema.json", "w") as f:
        json.dump(dummy_schema, f)

    # --- Arg Parsing and Execution ---
    security_id_to_process = sys.argv[1] if len(sys.argv) > 1 else dummy_sec_id
    years = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    print(f"\n--- Starting Standalone Projection for Security: {security_id_to_process} for {years} years ---")
    
    try:
        engine = ProjectionsEngine(security_id=security_id_to_process, projection_years=years)
        # Note: The 'project' method is stubbed out. Running it will demonstrate the setup phase.
        final_projection_df = engine.project()

        pd.set_option('display.max_rows', 200)
        pd.set_option('display.float_format', '{:,.2f}'.format)
        
        print("\n--- PROJECTION SETUP COMPLETE ---")
        # In a full run, this would print the final projections. Here it shows the initial grid.
        print(engine.data_grid.head())
        
    except Exception as e:
        logger.error(f"Projection failed for {security_id_to_process}. Reason: {e}", exc_info=True)
        print(f"\nERROR: Projection failed. Check logs for details.")

    print("\n--- Standalone Projection Run Finished ---")