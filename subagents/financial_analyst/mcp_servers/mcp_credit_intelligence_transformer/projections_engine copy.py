import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import List

import networkx as nx
import numpy as np
import pandas as pd

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
        self.execution_plan = []
        self.circular_group = set()
        self.revolver_interest_rate = 0.0

        logger.info(f"ProjectionsEngine initialized for Security ID: {self.security_id} for {self.projection_periods} years.")

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
    
    # --- Private Helper Methods for Grid Creation ---

    def _build_account_map(self) -> dict:
        """
        Recursively traverses the schema to build a flat map of account names to their
        schema definition, while also capturing the parent statement ('income_statement', etc.).
        """
        account_map = {}
        # We start recursion from the top-level statements.
        top_level_statements = ['income_statement', 'balance_sheet', 'cash_flow_statement']

        def recurse(d, statement_name):
            for key, value in d.items():
                if isinstance(value, dict):
                    # An item is considered a "projectable account" if it has a direct 'historical_account_key'.
                    if 'historical_account_key' in value:
                        if key in account_map:
                            raise ValueError(f"Duplicate account key '{key}' found. Account names must be unique.")
                        # Store the schema entry AND the parent statement name.
                        account_map[key] = {
                            'schema_entry': value,
                            'statement': statement_name
                        }
                    # Continue traversal
                    recurse(value, statement_name)

        for statement in top_level_statements:
            if statement in self.populated_schema:
                recurse(self.populated_schema[statement], statement)

        if not account_map:
            raise ValueError("Schema parsing resulted in an empty account map. Check schema structure.")
            
        logger.info(f"Built schema account map with {len(account_map)} entries, capturing parent statement context.")
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
        Initializes the main DataFrame, populates it with filtered historical data by
        mapping from the schema, and calculates historical subtotals for consistency.

        This updated version ensures all necessary engine-calculated subtotals and policy
        items are added to the grid's index from the start.

        Raises:
            KeyError: If a required account for historical subtotal calculation is missing
                    from the schema or mapped data.
        """
        logger.info("Building account map from schema...")
        self.account_map = self._build_account_map()
        
        # 1. Define all accounts for the DataFrame index
        # Start with all accounts defined in the schema
        all_accounts = list(self.account_map.keys())
        
        # Manually add all special, engine-calculated accounts for subtotals,
        # interest aggregation, and policy items. This is the critical update.
        engine_specific_accounts = [
            '__historical_plug_reversal__', # For neutralizing historical errors
            
            # Core Income Statement Subtotals
            'total_revenue', 
            'gross_profit',
            'operating_income', 
            'ebt', 
            'net_income',

            # Interest Subtotals for EBT calculation
            'interest_on_revolver',
            # 'interest_expense',
            
            # Balance Sheet Policy Items
            'min_cash_balance' # For the new cash target policy
        ]
        all_accounts.extend(engine_specific_accounts)
        
        # 2. Initialize the empty data grid
        # Use sorted(list(set(...))) to ensure a unique, ordered index
        projection_cols = [f"P{i}" for i in range(1, self.projection_periods + 1)]
        all_cols = self.historical_years + projection_cols
        self.data_grid = pd.DataFrame(
            index=sorted(list(set(all_accounts))),
            columns=all_cols,
            dtype=np.float64
        ).fillna(0.0)
        logger.info(f"Initialized data grid with shape {self.data_grid.shape}. Populating historicals...")

        # 3. Populate historical data from source files
        for account_name, map_entry in self.account_map.items():
            schema_entry = map_entry['schema_entry']
            statement = map_entry['statement']
            
            hist_key = schema_entry.get('historical_account_key')
            if not hist_key:
                continue # Skip accounts with no historical mapping

            # The historical_account_key from the schema is the full lookup path.
            full_lookup_path = hist_key
            
            for period_data, year in zip(self.historical_data_raw, self.historical_years):
                value = self._get_value_from_dot_path(period_data, full_lookup_path)
                self.data_grid.loc[account_name, year] = value
        
        logger.info("Completed mapping of raw historical data.")

        # 4. Calculate historical subtotals to ensure consistency.
        # This step validates the integrity of the mapped data. It does not touch
        # the new, purely projection-based accounts like 'interest_on_revolver'.
        logger.info("Calculating historical subtotals for internal consistency...")
        try:
            for year in self.historical_years:
                # --- Income Statement Totals ---
                rev_streams = [acc for acc in self.data_grid.index if acc.startswith('revenue_stream')]
                cogs_streams = [acc for acc in self.data_grid.index if acc.startswith('cost_of_revenue')]
                sga_streams = [acc for acc in self.data_grid.index if acc.startswith('sga_expense')]
                prov_streams = [acc for acc in self.data_grid.index if acc.startswith('provision_for_credit_losses')]
                
                self.data_grid.loc['total_revenue', year] = self.data_grid.loc[rev_streams, year].sum()
                self.data_grid.loc['gross_profit', year] = self.data_grid.loc['total_revenue', year] - self.data_grid.loc[cogs_streams, year].sum()
                
                self.data_grid.loc['operating_income', year] = (
                    self.data_grid.loc['gross_profit', year] - 
                    self.data_grid.loc[sga_streams, year].sum() -
                    self.data_grid.loc[prov_streams, year].sum()
                )
                
                # EBT and Net Income are derived from mapped historicals. If 'ebt' was mapped,
                # we could validate it here. For now, we trust the mapped values for these totals.
                # Example validation: ebt_calc = op_inc - interest_exp. Discrepancy = mapped_ebt - ebt_calc.

        except KeyError as e:
            raise KeyError(f"Failed to calculate historical subtotals. A required account '{e}' is missing from the data grid or schema. Check mappings.")
        
        logger.info("Historical data grid fully populated and validated.")

    def _neutralize_historical_plugs(self) -> None:
        """
        Identifies summation plugs in the final historical balance sheet (T-1)
        and posts an offsetting reversal in the first projection period (P1)
        to a dedicated account, ensuring the projection starts from a balanced state.

        This method has side-effects: it modifies self.data_grid.
        
        Raises:
            RuntimeError: If the raw data for the anchor year cannot be found.
            KeyError: If the required '__historical_plug_reversal__' account is missing.
        """
        logger.info(f"Starting historical plug neutralization for anchor year {self.T1_year}...")

        # 1. Find the raw data dictionary for the anchor year (T-1). Fail fast if not found.
        try:
            t1_index = self.historical_years.index(self.T1_year)
            t1_data_period = self.historical_data_raw[t1_index]
        except (ValueError, IndexError):
            raise RuntimeError(f"FATAL: Could not find raw data for anchor year {self.T1_year}. Internal state is inconsistent.")

        # 2. Safely access the summation plugs. It's not an error if they don't exist.
        bs_plugs = t1_data_period.get('balance_sheet', {}).get('summation_plugs', {})

        if not bs_plugs:
            logger.info(f"No balance sheet summation plugs found in anchor year {self.T1_year}. No neutralization needed.")
            return

        # 3. Ensure the dedicated reversal account exists in our grid. Fail fast if not.
        reversal_account = '__historical_plug_reversal__'
        first_projection_col = "P1"
        if reversal_account not in self.data_grid.index:
            raise KeyError(f"FATAL: The required '{reversal_account}' account is not present in the data_grid index. Cannot neutralize plugs.")

        # 4. Iterate, validate, and post the offsetting reversals to the P1 column.
        total_reversal_amount = 0.0
        for plug_item, plug_value in bs_plugs.items():
            # The accounting equation plug is for reporting, not for balancing assets/liabilities. Ignore it.
            if plug_item == '__accounting_equation__':
                continue
                
            if not isinstance(plug_value, (int, float)):
                logger.warning(f"Skipping non-numeric plug value for '{plug_item}' in year {self.T1_year}. Value: {plug_value}")
                continue
            
            # The core logic: post the NEGATIVE of the plug value to reverse it.
            reversal_amount = -float(plug_value)
            self.data_grid.loc[reversal_account, first_projection_col] += reversal_amount
            total_reversal_amount += reversal_amount
        
        if total_reversal_amount == 0.0:
            logger.info("Found plug structure, but all values were zero or non-numeric. No reversal action taken.")
        else:
            # Post the total reversal amount to the dedicated reporting line item.
            logger.info(f"Neutralized {len(bs_plugs)-1 if '__accounting_equation__' in bs_plugs else len(bs_plugs)} plug(s) from T-1. Total reversal of {total_reversal_amount:,.2f} posted to '{reversal_account}' in {first_projection_col}.")
            
            # CRITICAL FIX: To maintain balance sheet integrity, the plug reversal must be offset in equity.
            # We adjust the opening Retained Earnings of the first projection period.
            self.data_grid.loc['retained_earnings', first_projection_col] += total_reversal_amount
            logger.info(f"Posted offsetting adjustment of {total_reversal_amount:,.2f} to Retained Earnings in {first_projection_col} to balance plug reversal.")

    def _resolve_historical_source_conflicts(self) -> None:
        """
        Orchestrates the detection and resolution of known historical data source conflicts.
        
        This method acts as a fail-fast gatekeeper, calling specialized handlers to surgically
        clean the historical data in the `data_grid` before projections are run.
        It ensures that the projection starts from a sanitized and economically sound base.
        
        Raises:
            RuntimeError: If any conflict resolution fails, with detailed error information.
        """
        logger.info("[CONFLICT RESOLUTION] Starting historical source conflict detection...")
        logger.info(f"[DEBUG] Account map keys available: {list(self.account_map.keys()) if hasattr(self, 'account_map') else 'No account_map found'}")
        
        try:
            # Handle each conflict type in sequence
            self._handle_depreciation_amortization_conflict()
            self._handle_operating_expense_conflict()
            self._handle_cash_item_conflict()
            
            logger.info("[CONFLICT RESOLUTION] All conflict checks completed successfully")
            
        except Exception as e:
            logger.error(f"[CRITICAL ERROR] Historical source conflict resolution failed: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"FAIL FAST: Historical source conflict resolution failed - {e}")
            
        logger.info("Historical source conflict scan complete.")
        logger.info("=== HISTORICAL SOURCE CONFLICT RESOLUTION END ===")

    def _get_total_balance_for_category(self, category_name: str, sub_category_name: str, hist_col: str) -> float:
        """
        Calculates the total historical balance for all line items within a given schema category
        by accessing the category using its direct path from the universal schema.

        Args:
            category_name: The exact category key from the universal schema (e.g., 'property_plant_equipment').
            sub_category_name: The key for the dictionary of items (usually 'items').
            hist_col: The historical year column to look up in the data grid.

        Returns:
            The total aggregated balance for the category.

        Raises:
            RuntimeError: If an item from the schema is not found in the data grid.
        """
        total_balance = 0.0
        logger.info(f"[DEBUG] Aggregating balances for category '{category_name}' in year {hist_col}")

        try:
            # Access the category directly using its canonical path from the universal schema.
            category_items = self.populated_schema['balance_sheet'][category_name][sub_category_name]
            item_names = list(category_items.keys())
            logger.info(f"[DEBUG] Found items for '{category_name}': {item_names}")
        except KeyError:
            # This is a valid scenario if the JIT schema doesn't map any items to this category.
            logger.warning(f"[DEBUG] Category '{category_name}' not found in populated schema or has no items. Assuming balance of 0.")
            return 0.0

        for item_name in item_names:
            try:
                balance = self.data_grid.loc[item_name, hist_col]
                total_balance += balance
                logger.info(f"[DEBUG] Fetched balance for '{item_name}': {balance:,.2f}")
            except KeyError:
                # FAIL FAST: If an item from the schema is not in the data grid, it's a critical error.
                error_msg = f"CRITICAL: Item '{item_name}' from schema category '{category_name}' not found in data grid for {hist_col}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        
        logger.info(f"[DEBUG] Total aggregated balance for '{category_name}' in {hist_col}: {total_balance:,.2f}")
        return total_balance

    def _handle_depreciation_amortization_conflict(self) -> None:
        """
        Detects and resolves conflicts where Depreciation and Amortization are mapped
        to the same historical source account.
        
        If a conflict is found, it splits the combined historical value pro-rata based
        on the historical balances of Property, Plant & Equipment (PPE) and Intangible Assets.
        This modification is performed directly on `self.data_grid`.
        
        Raises:
            RuntimeError: If essential accounts for the split are not in the schema or data grid.
        """
        logger.info("[CONFLICT CHECK] Depreciation & Amortization...")
        
        # Get the historical account keys from the populated schema
        try:
            depreciation_schema = self.populated_schema['income_statement']['depreciation']['items']['depreciation_expense']
            depreciation_key = depreciation_schema.get('historical_account_key')
        except KeyError:
            depreciation_key = None
            
        try:
            amortization_schema = self.populated_schema['income_statement']['amortization']['items']['amortization_expense']
            amortization_key = amortization_schema.get('historical_account_key')
        except KeyError:
            amortization_key = None
        
        logger.info(f"[DEBUG] Depreciation mapped to: {depreciation_key}")
        logger.info(f"[DEBUG] Amortization mapped to: {amortization_key}")
        
        # If either is not mapped, no conflict possible
        if not depreciation_key or not amortization_key:
            logger.info("[RESULT] No conflict - one or both accounts not mapped")
            return
            
        # If they're mapped to different sources, no conflict
        if depreciation_key != amortization_key:
            logger.info("[RESULT] No conflict - mapped to different historical sources")
            return
            
        # CONFLICT DETECTED - fail fast approach
        conflict_key = depreciation_key
        logger.error(f"[CONFLICT DETECTED] Both 'depreciation_expense' and 'amortization_expense' mapped to '{conflict_key}'")
        logger.info("[RESOLUTION] Attempting pro-rata asset split for all historical years...")

        if not self.historical_years:
            raise RuntimeError("No historical years found in data grid - cannot resolve D&A conflict")

        for hist_col in self.historical_years:
            logger.info(f"[DEBUG] Processing historical column: {hist_col}")

            # Get the combined expense value - FAIL FAST if not found
            try:
                combined_expense = self.data_grid.loc['depreciation_expense', hist_col]
                logger.info(f"[DEBUG] Combined historical D&A expense for {hist_col}: {combined_expense:,.2f}")
            except KeyError:
                raise RuntimeError(f"CRITICAL: 'depreciation_expense' not found in data grid for {hist_col} despite conflict detection for '{conflict_key}'")

            # Get aggregated asset balances for pro-rata calculation using the correct universal schema keys
            ppe_balance = self._get_total_balance_for_category(
                'property_plant_equipment', 'items', hist_col
            )
            intangible_balance = self._get_total_balance_for_category(
                'intangible_assets', 'items', hist_col
            )

            # Calculate pro-rata split - FAIL FAST if total is zero
            total_asset_base = ppe_balance + intangible_balance
            if total_asset_base == 0:
                logger.warning(f"[WARNING] Total asset base for {hist_col} is zero. Skipping pro-rata split for this year.")
                continue

            # Calculate split ratios
            dep_ratio = ppe_balance / total_asset_base
            amort_ratio = intangible_balance / total_asset_base
            
            depreciation_split = combined_expense * dep_ratio
            amortization_split = combined_expense * amort_ratio

            # Verify split adds up (sanity check)
            split_total = depreciation_split + amortization_split
            if abs(split_total - combined_expense) > 0.01:  # Allow for small rounding differences
                raise RuntimeError(f"CRITICAL: Split calculation error for {hist_col} - original: {combined_expense:,.2f}, split total: {split_total:,.2f}")

            # Update data grid with split values
            self.data_grid.loc['depreciation_expense', hist_col] = depreciation_split
            self.data_grid.loc['amortization_expense', hist_col] = amortization_split
            
            logger.info(f"[SUCCESS] D&A conflict for {hist_col} resolved. New Depreciation: {depreciation_split:,.2f}, New Amortization: {amortization_split:,.2f}")

        logger.info(f"[RESULT] Completed D&A conflict resolution for all historical years.")

    def _handle_operating_expense_conflict(self) -> None:
        """
        Detects and resolves conflicts where Cost of Revenue and SG&A are mapped
        to the same historical source, representing a vague 'Total Operating Costs'.
        
        If a conflict is found, it applies a deterministic 50/50 split to the
        combined historical value.
        
        Raises:
            RuntimeError: If essential accounts are not found in the data grid.
        """
        logger.info("[CONFLICT CHECK] Operating Expenses...")
        
        # Get the historical account keys from the populated schema
        try:
            cor_schema = self.populated_schema['income_statement']['cost_of_revenue']['items']['cost_of_revenue_1']
            cor_key = cor_schema.get('historical_account_key')
        except KeyError:
            cor_key = None
            
        try:
            sga_schema = self.populated_schema['income_statement']['operating_expenses']['items']['sga_expense']
            sga_key = sga_schema.get('historical_account_key')
        except KeyError:
            sga_key = None
        
        logger.info(f"[DEBUG] Cost of Revenue mapped to: {cor_key}")
        logger.info(f"[DEBUG] SG&A Expense mapped to: {sga_key}")
        
        # If either is not mapped, no conflict possible
        if not cor_key or not sga_key:
            logger.info("[RESULT] No conflict - one or both accounts not mapped")
            return
            
        # If they're mapped to different sources, no conflict
        if cor_key != sga_key:
            logger.info("[RESULT] No conflict - mapped to different historical sources")
            return
            
        # CONFLICT DETECTED - fail fast approach
        conflict_key = cor_key
        logger.error(f"[CONFLICT DETECTED] Both 'cost_of_revenue_1' and 'sga_expense' mapped to '{conflict_key}'")
        logger.info("[RESOLUTION] Attempting 50/50 split for all historical years...")

        if not self.historical_years:
            raise RuntimeError("No historical years found in data grid - cannot resolve operating expense conflict")

        for hist_col in self.historical_years:
            logger.info(f"[DEBUG] Processing historical column: {hist_col}")

            # Get the combined expense value - FAIL FAST if not found
            try:
                combined_expense = self.data_grid.loc['cost_of_revenue_1', hist_col]
                logger.info(f"[DEBUG] Combined historical operating expense for {hist_col}: {combined_expense:,.2f}")
            except KeyError:
                raise RuntimeError(f"CRITICAL: 'cost_of_revenue_1' not found in data grid for {hist_col} despite conflict detection for '{conflict_key}'")

            # Calculate 50/50 split
            split_value = combined_expense / 2.0
            
            # Verify we can access the SG&A account in data grid before updating
            if 'sga_expense' not in self.data_grid.index:
                 raise RuntimeError("CRITICAL: 'sga_expense' not found in data grid - cannot apply split")

            # Update data grid with split values
            self.data_grid.loc['cost_of_revenue_1', hist_col] = split_value
            self.data_grid.loc['sga_expense', hist_col] = split_value
            
            logger.info(f"[SUCCESS] Operating expense conflict for {hist_col} resolved. New CoR: {split_value:,.2f}, New SG&A: {split_value:,.2f}")

        logger.info(f"[RESULT] Completed operating expense conflict resolution for all historical years.")



    def _handle_cash_item_conflict(self) -> None:
        """
        Detects and resolves conflicts where multiple cash items are mapped to the
        same historical source, which would inflate the starting cash balance.
        
        If a conflict is found, it consolidates the value into the first mapped cash
        item and zeroes out the others for all historical years.
        
        Raises:
            RuntimeError: If cash schema structure is invalid or data grid access fails.
        """
        logger.info("[CONFLICT CHECK] Cash Items...")
        
        # Get cash items from schema
        try:
            cash_schema = self.populated_schema['balance_sheet']['cash_and_equivalents']['items']
        except KeyError:
            logger.info("[RESULT] No cash items schema found - skipping cash conflict check")
            return
            
        # Build source mapping to find conflicts
        source_map = {}
        for item_name, item_schema in cash_schema.items():
            key = item_schema.get('historical_account_key')
            if not key:
                logger.info(f"[DEBUG] Cash item '{item_name}' has no historical mapping - skipping")
                continue
                
            if key not in source_map:
                source_map[key] = []
            source_map[key].append(item_name)
        
        logger.info(f"[DEBUG] Built cash source map: {source_map}")

        # Find and resolve conflicts for each historical year
        for key, items in source_map.items():
            if len(items) <= 1:
                continue  # No conflict for this source key

            logger.error(f"[CONFLICT DETECTED] Multiple cash items mapped to '{key}': {items}")
            logger.info("[RESOLUTION] Consolidating into first item and zeroing out others...")

            primary_item = items[0]
            conflicting_items = items[1:]
            logger.info(f"Primary item (will keep value): {primary_item}")
            logger.info(f"Conflicting items (will be zeroed): {conflicting_items}")

            if not self.historical_years:
                raise RuntimeError("No historical years found in data grid - cannot resolve cash item conflict")

            for hist_col in self.historical_years:
                # The value is already present under all conflicting account names.
                # We just need to zero out the duplicates, leaving the primary item's value intact.
                for item_to_zero in conflicting_items:
                    try:
                        self.data_grid.loc[item_to_zero, hist_col] = 0.0
                        logger.info(f"[SUCCESS] Zeroed out '{item_to_zero}' for year {hist_col}")
                    except KeyError:
                        # This is a critical failure, as the item should be in the grid if it was mapped
                        raise RuntimeError(f"CRITICAL: Cash item '{item_to_zero}' not found in data grid for {hist_col}")

        logger.info("[RESULT] Completed cash item conflict resolution.")
        
        # Check for conflicts
        conflicts_found = False
        hist_col = None
        
        for source_key, mapped_items in source_map.items():
            if len(mapped_items) > 1:
                conflicts_found = True
                logger.error(f"[CONFLICT DETECTED] {len(mapped_items)} cash items {mapped_items} mapped to '{source_key}'")
                
                # Get historical column on first conflict
                if hist_col is None:
                    if not self.historical_years:
                        raise RuntimeError("No historical years found in data grid - cannot resolve cash item conflict")
                    hist_col = self.historical_years[-1]
                    logger.info(f"[DEBUG] Using historical column: {hist_col}")
                
                # Consolidate to first item, zero out others
                primary_item = mapped_items[0]
                redundant_items = mapped_items[1:]
                
                logger.info(f"[RESOLUTION] Consolidating to primary item '{primary_item}', zeroing out {redundant_items}")
                
                # Get the value from primary item - FAIL FAST if not found
                try:
                    consolidated_value = self.data_grid.loc[primary_item, hist_col]
                    logger.info(f"[DEBUG] Value to consolidate: {consolidated_value:,.2f}")
                except KeyError:
                    raise RuntimeError(f"CRITICAL: Primary cash item '{primary_item}' not found in data grid despite conflict detection")
                
                # Zero out redundant items - FAIL FAST if any not found
                for redundant_item in redundant_items:
                    try:
                        original_value = self.data_grid.loc[redundant_item, hist_col]
                        self.data_grid.loc[redundant_item, hist_col] = 0.0
                        logger.info(f"[DEBUG] Zeroed out '{redundant_item}' (was {original_value:,.2f})")
                    except KeyError:
                        raise RuntimeError(f"CRITICAL: Redundant cash item '{redundant_item}' not found in data grid")
                
                logger.info(f"[SUCCESS] Cash conflict resolved for source '{source_key}'")
                logger.info(f"[RESULT] Consolidated value {consolidated_value:,.2f} in '{primary_item}', zeroed {len(redundant_items)} duplicates")
        
        if not conflicts_found:
            logger.info("[RESULT] No cash item conflicts detected")

    # --- Private Method to be Added ---

    def _build_and_compile_graph(self) -> None:
        """
        Builds a dependency graph, dynamically incorporates model-specific interdependencies,
        identifies the final circular group via SCC analysis, and compiles a deterministic,
        two-stage topological execution plan. This version dynamically constructs the
        formula map to ensure all nodes are explicitly defined before use.

        This method has side-effects: it populates self.graph, self.circular_group,
        and self.execution_plan.

        Raises:
            RuntimeError: If more than one circular dependency group is detected.
            nx.NetworkXUnfeasible: If a topological sort of the pre-circular graph fails.
        """
        logger.info("Defining financial logic map...")

        # --- Dynamic Map Construction (THE FINAL FIX) ---
        # Start with a base map of known relationships.
        _FORMULA_MAP = {
            'cost_of_revenue_1': ['total_revenue'],
            'sga_expense': ['total_revenue'],
            'provision_for_credit_losses': ['main_receivables'],
            'main_inventory': ['cost_of_revenue_1'],
            'main_payables': ['cost_of_revenue_1'],
            'net_ppe': ['total_revenue'],
            'income_tax_expense': ['ebt'],
            'min_cash_balance': ['total_revenue'],
            'total_revenue': [acc for acc in self.data_grid.index if acc.startswith('revenue_stream')],
            'gross_profit': ['total_revenue', 'cost_of_revenue_1'],
            'operating_income': ['gross_profit', 'sga_expense', 'provision_for_credit_losses'],
            'ebt': ['operating_income', 'interest_on_revolver', 'interest_expense'],
            'net_income': ['ebt', 'income_tax_expense'],
            'retained_earnings': ['net_income'],
            'interest_on_revolver': ['revolver'],
    
            'depreciation_amortization': ['net_ppe'],
        }

        # Dynamically add all simple roll-forward accounts to the map as nodes with no
        # intra-period dependencies. This is CRITICAL to ensure they exist in the graph
        # before other items declare a dependency on them.
        for debt_name in self.scheduled_debt_items:
            _FORMULA_MAP[debt_name] = [] # No intra-period dependencies

        _FORMULA_MAP['common_stock'] = [] # No intra-period dependencies

        # Now, with all nodes guaranteed to exist, define the dependencies for aggregate items.
        _FORMULA_MAP['interest_expense'] = self.scheduled_debt_items
        _FORMULA_MAP['revolver'] = [
            'main_receivables', 'main_inventory', 'net_ppe', '__historical_plug_reversal__',
            'main_payables', 'common_stock', 'retained_earnings', 'min_cash_balance'
        ] + self.scheduled_debt_items
        _FORMULA_MAP['cash'] = [
            'revolver', 'main_receivables', 'main_inventory', 'net_ppe', '__historical_plug_reversal__',
            'main_payables', 'common_stock', 'retained_earnings'
        ] + self.scheduled_debt_items

        # Dynamically handle operational interdependencies
        logger.info("Checking for model-specific interdependencies...")
        # Handle all revenue streams in the schema
        try:
            revenue_items = self.populated_schema['income_statement']['revenue']['items']
            for stream_name, stream_config in revenue_items.items():
                if stream_name.startswith('revenue_stream'):
                    # Ensure the revenue stream is added to the data grid
                    if stream_name not in self.data_grid.index:
                        self.data_grid.loc[stream_name] = 0.0
                    
                    model_config = stream_config['projection_configuration']
                    if model_config.get('model_name') == 'Asset_Yield_Driven':
                        # Get target asset name
                        target_asset = model_config['target_asset_account'].split('.')[-1]
                        
                        # Add to formula map
                        _FORMULA_MAP[stream_name] = [target_asset]
                        logger.info(f"Detected 'Asset_Yield_Driven' model for {stream_name}. Adding dependency on {target_asset}.")
                    elif model_config.get('model_name') == 'Top_Line_Growth':
                        _FORMULA_MAP[stream_name] = []
                        logger.info(f"Detected 'Top_Line_Growth' model for {stream_name}. No dependencies.")
                        
            logger.info(f"Added {len([k for k in _FORMULA_MAP if k.startswith('revenue_stream')])} revenue streams to formula map.")
        except KeyError as e:
            raise RuntimeError(f"Failed to parse revenue model configuration from schema. Check structure. Error: {e}")
            
        # Make sure revenue streams are included in the total_revenue calculation
        _FORMULA_MAP['total_revenue'] = [acc for acc in self.data_grid.index if acc.startswith('revenue_stream')]

        logger.info(f"Building dependency graph...")
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(_FORMULA_MAP.keys())
        for account, dependencies in _FORMULA_MAP.items():
            for dep in dependencies:
                if dep not in self.graph:
                    # This should not happen with the new map structure, but it is a robust safeguard.
                    self.graph.add_node(dep)
                self.graph.add_edge(dep, account)

        logger.info(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

        # --- Isolate ALL Circularities (SCC Analysis) ---
        logger.info("Performing Strongly Connected Components (SCC) analysis...")
        sccs = list(nx.strongly_connected_components(self.graph))
        circular_components = [scc for scc in sccs if len(scc) > 1]

        if not circular_components:
            raise RuntimeError("FATAL: No circular dependencies found. The core revolver-interest loop is missing from the graph logic.")
        elif len(circular_components) > 1:
            raise RuntimeError(f"FATAL: Found {len(circular_components)} disjoint circular dependency groups. Engine is designed for one unified group. Groups: {circular_components}")
        
        self.circular_group = circular_components[0]
        logger.info(f"Successfully identified the unified circular dependency group: {self.circular_group}")

        # --- Generate Final Execution Plan ---
        logger.info("Compiling final topological execution plan...")
        try:
            pre_circular_nodes = [n for n in self.graph.nodes() if n not in self.circular_group]
            pre_circular_subgraph = self.graph.subgraph(pre_circular_nodes)
            pre_circular_plan = list(nx.topological_sort(pre_circular_subgraph))
            
            self.execution_plan = {
                'pre_circular': pre_circular_plan,
                'circular': list(self.circular_group)
            }
        except nx.NetworkXUnfeasible:
            raise RuntimeError("Failed to create a topological sort of the pre-circular graph. Check _FORMULA_MAP for unintended cycles.")

        logger.info(f"Execution plan compiled: {len(pre_circular_plan)} pre-circular steps, {len(self.circular_group)} circular steps.")
        

    def _calculate_revolver_rate(self) -> None:
        """
        Calculates a representative interest rate for the revolver by averaging the
        baseline rates of all specified scheduled debt instruments.

        This method is a critical setup step for the circular solver. It fails
        fast if no scheduled debt with a defined interest rate is found.

        This method has a side-effect: it populates self.revolver_interest_rate.
        """
        logger.info("Calculating revolver interest rate based on average of scheduled debt...")
        
        # 1. Identify all scheduled debt instruments from the schema's structure
        try:
            scheduled_debt_items = self.populated_schema.get('balance_sheet', {}).get('debt', {}).get('scheduled_debt', {}).get('items', {})
        except AttributeError:
            # This handles cases where the nested keys don't exist, treating it as no debt found.
            scheduled_debt_items = {}

        if not scheduled_debt_items:
            raise ValueError("FATAL: Cannot calculate revolver rate. The 'scheduled_debt.items' object in the schema is missing or empty.")

        # 2. Extract individual interest rates from the drivers of each instrument
        rates_to_average = []
        sources = [] # For logging and auditability

        for instrument_name, instrument_details in scheduled_debt_items.items():
            try:
                # Navigate safely through the nested driver structure
                rate_driver = instrument_details.get('drivers', {}).get('interest_rate', {})
                baseline_rate = rate_driver.get('baseline')

                if baseline_rate is not None and isinstance(baseline_rate, (int, float)):
                    rates_to_average.append(float(baseline_rate))
                    sources.append(instrument_name)
                else:
                    # This is a warning, not a failure. An instrument might exist without a rate driver.
                    logger.warning(f"Scheduled debt instrument '{instrument_name}' is missing a valid 'baseline' interest rate. It will be excluded from the revolver rate calculation.")

            except Exception as e:
                # Catch any unexpected structural issues for a specific instrument
                logger.error(f"Error processing interest rate for instrument '{instrument_name}': {e}")
                continue # Move to the next instrument

        # 3. Fail-Fast Check: Ensure we found at least one valid rate
        if not rates_to_average:
            raise ValueError("FATAL: Could not determine revolver interest rate. No scheduled debt instruments with a valid baseline interest rate were found in the schema.")

        # 4. Calculate the average, store, and log the result
        self.revolver_interest_rate = sum(rates_to_average) / len(rates_to_average)
        
        logger.info(f"Calculated average interest rate for revolver: {self.revolver_interest_rate:.4%}")
        logger.info(f"--> Based on {len(sources)} source(s): {', '.join(sources)}")

    # --- Constants for the Projection Logic ---
    MAX_ITERATIONS_CIRCULAR = 100
    MAX_ITERATIONS_PRE_CIRCULAR = 5 # Number of passes to stabilize pre-circular dependencies
    CONVERGENCE_TOLERANCE = 1.00 # Converged when the change is less than $1.00

    def _get_prior_period_col(self, current_col: str) -> str:
        """Gets the column name for the period immediately preceding the current one."""
        current_idx = self.data_grid.columns.get_loc(current_col)
        if current_idx == 0:
            raise IndexError("Cannot get prior period for the first column in the data grid.")
        return self.data_grid.columns[current_idx - 1]

    def _get_driver_value(self, driver_obj: dict, p_year_index: int) -> float:
        """
        Extracts the correct driver value (baseline, short_term, etc.) for a given
        projection year index (1-based).
        """
        # This check is critical for drivers that might be missing the 'trends' key
        if not isinstance(driver_obj, dict):
            return 0.0

        trends = driver_obj.get('trends', {})
        if p_year_index == 1:
            return driver_obj.get('baseline', 0.0)
        elif 2 <= p_year_index <= 3:
            return trends.get('short_term', 0.0)
        elif 4 <= p_year_index <= 9:
            return trends.get('medium_term', 0.0)
        else: # Year 10 and onwards
            return trends.get('terminal', 0.0)

    def _calculate_delta(self, account: str, year_col: str, p_year_index: int, opening_balances=None) -> float:
        """
        Calculates the CHANGE (delta) for a single account during a single period.
        This function uses opening_balances for prior state if provided, otherwise
        it refers to the prior period column.

        Args:
            account: The account name (row index) to calculate the delta for.
            year_col: The period (column name) to calculate for.
            p_year_index: The 1-based index of the projection year.
            opening_balances: Series containing opening balances for this period.
                If provided, these values are used instead of looking back to prior_year_col.

        Returns:
            The calculated float value representing the change for the account.
        """
        prior_year_col = self._get_prior_period_col(year_col)
        schema_entry = self.account_map.get(account, {}).get('schema_entry', {})
        
        # Determine which source to use for prior period values
        # If opening_balances provided, use it; otherwise use prior_year_col from data_grid
        def get_prior_value(acc):
            if opening_balances is not None:
                return opening_balances.loc[acc]
            else:
                return self.data_grid.loc[acc, prior_year_col]

        # --- P&L Items and Derived Values ---
        # For these items, the "delta" is their full calculated value for the period.
        if account == 'revenue_stream_1':
            # Access the model details through the selected_model key
            config = schema_entry['projection_configuration']['selected_model']
            model_name = config['model_name']
            if model_name == "Top_Line_Growth":
                driver = self._get_driver_value(config['driver']['revenue_growth_rate'], p_year_index)
                prior_revenue = get_prior_value(account)
                # Apply growth rate to get new revenue - ensure it's different from prior period
                new_revenue = prior_revenue * (1 + driver)
                # Log for debugging
                logger.info(f"[{account}][{year_col}] Growth rate: {driver:.2%}, Prior: {prior_revenue:,.2f}, New: {new_revenue:,.2f}")
                return new_revenue
            elif model_name == "Asset_Yield_Driven":
                driver = self._get_driver_value(config['driver']['asset_yield'], p_year_index)
                # Get target asset from config path
                path_elements = config['target_asset_account'].split('.')
                schema_asset_name = path_elements[-1]

                # Resolve the historical account key to get the correct data grid row name.
                # The data is stored in the grid under the schema name, which should have been populated
                # from the historical key. However, the dependency might be on the key itself.
                # The most robust approach is to use the schema name directly, as the grid is built from it.
                target_asset_for_data = schema_asset_name

                # Check if asset exists in data_grid
                if target_asset_for_data not in self.data_grid.index:
                    logger.error(f"Asset_Yield_Driven model error: Target asset '{target_asset_for_data}' not found in data_grid")
                    return 0.0
                    
                opening_asset = get_prior_value(target_asset_for_data)
                closing_asset = self.data_grid.loc[target_asset_for_data, year_col]
                avg_asset = (opening_asset + closing_asset) / 2
                
                # Calculate revenue based on yield
                revenue = avg_asset * driver
                logger.info(f"[{account}][{year_col}] Asset: {target_asset_for_data}, Yield: {driver:.2%}, Asset value: {avg_asset:,.2f}, Revenue: {revenue:,.2f}")
                return revenue
            else:
                raise NotImplementedError(f"Revenue model '{model_name}' is not supported.")
        
        elif account in self.cost_of_revenue_items:
            driver = self._get_driver_value(schema_entry['drivers']['cogs_as_percent_of_revenue'], p_year_index)
            return self.data_grid.loc['total_revenue', year_col] * driver * -1
        
        elif account in self.operating_expense_items and 'sga_as_percent_of_revenue' in schema_entry.get('drivers', {}):
            driver = self._get_driver_value(schema_entry['drivers']['sga_as_percent_of_revenue'], p_year_index)
            return self.data_grid.loc['total_revenue', year_col] * driver * -1
            
        elif account == 'provision_for_credit_losses':
            driver = self._get_driver_value(schema_entry['drivers']['provision_rate_on_receivables'], p_year_index)
            # Dynamically calculate average receivables from all receivable accounts
            opening_receivables = sum(get_prior_value(acc) for acc in self.receivables_accounts)
            closing_receivables = self.data_grid.loc[self.receivables_accounts, year_col].sum()
            avg_receivables = (opening_receivables + closing_receivables) / 2
            return avg_receivables * driver * -1

        elif account == 'interest_expense':
            # This is the correct, detailed logic that was previously unreachable.
            total_interest = 0.0
            
            # Use self.scheduled_debt_items, which is the definitive list for the engine.
            # scheduled_debt_items from the local scope might not be defined.
            if not hasattr(self, 'scheduled_debt_items'):
                 return 0.0 # No debt, no interest

            for debt_name in self.scheduled_debt_items:
                try:
                    logger.info(f"[{account}][{year_col}] Processing interest for: {debt_name}")
                    
                    debt_schema_entry = self.account_map[debt_name]['schema_entry']
                    
                    # Correctly pass the driver object to the helper function
                    rate_driver_obj = debt_schema_entry['drivers']['interest_rate']
                    rate = self._get_driver_value(rate_driver_obj, p_year_index)
                    logger.info(f"[{account}][{year_col}] {debt_name} interest rate: {rate:.4f}")
                    
                    # Calculate the average balance for the period for an accurate interest calculation
                    prior_balance = get_prior_value(debt_name)
                    current_balance = self.data_grid.loc[debt_name, year_col]
                    avg_balance = (prior_balance + current_balance) / 2
                    logger.info(f"[{account}][{year_col}] {debt_name} avg balance: {avg_balance:,.2f} (prior: {prior_balance:,.2f}, current: {current_balance:,.2f})")
                    
                    # Calculate interest for this debt
                    interest = avg_balance * rate
                    total_interest += interest
                    logger.info(f"[{account}][{year_col}] {debt_name} interest: {interest:,.2f}, running total: {total_interest:,.2f}")
                
                except KeyError as e:
                    logger.error(f"FATAL: Schema misconfiguration for debt instrument '{debt_name}'. Missing key: {e}")
                    raise RuntimeError(f"Could not calculate interest for {debt_name} due to missing schema key: {e}")

            # Final interest amount (negated for P&L)
            final_amount = total_interest * -1
            logger.info(f"[{account}][{year_col}] Final interest on scheduled debt: {final_amount:,.2f}")
            return final_amount

        elif account == 'interest_on_revolver':
            avg_balance = (get_prior_value('revolver') + self.data_grid.loc['revolver', year_col]) / 2
            # Interest expense must be negative
            return avg_balance * self.revolver_interest_rate * -1
        
        elif account == 'income_tax_expense':
            tax_rate_driver = self._get_driver_value(schema_entry['drivers']['effective_tax_rate'], p_year_index)
            ebt = self.data_grid.loc['ebt', year_col]
            # Allow for tax benefits (negative tax) in case of losses (negative EBT)
            return ebt * tax_rate_driver

        elif account == 'min_cash_balance':
            cash_policy_drivers = self.populated_schema['balance_sheet']['cash_and_equivalents']['cash_policy']['drivers']
            driver = self._get_driver_value(cash_policy_drivers['min_cash_as_percent_of_revenue'], p_year_index)
            return self.data_grid.loc['total_revenue', year_col] * driver

        # --- Subtotals ---
        elif account == 'total_revenue':
            return self.data_grid.loc[[acc for acc in self.data_grid.index if acc.startswith('revenue_stream')], year_col].sum()
        elif account == 'gross_profit':
            cogs_sum = self.data_grid.loc[[acc for acc in self.data_grid.index if acc.startswith('cost_of_revenue')], year_col].sum()
            return self.data_grid.loc['total_revenue', year_col] + cogs_sum # Add because COGS is negative
        elif account == 'operating_income':
            op_ex_sum = self.data_grid.loc[['sga_expense', 'provision_for_credit_losses'], year_col].sum()
            return self.data_grid.loc['gross_profit', year_col] + op_ex_sum # Add because OpEx is negative
        elif account == 'ebt':
            # EBT = Operating Income - Interest Expense
            operating_income = self.data_grid.loc['operating_income', year_col]
            # For lenders, interest_expense is already part of cost of revenue, so only include interest_on_revolver
            # Check if it's Asset_Yield_Driven model AND if cost_of_revenue includes interest in historical mapping
            is_asset_yield_model = self.populated_schema['income_statement']['revenue']['items']['revenue_stream_1']['projection_configuration']['selected_model']['model_name'] == 'Asset_Yield_Driven'
            has_interest_in_cor = False
            try:
                cor_historical_account = self.populated_schema['income_statement']['cost_of_revenue']['items'].get('cost_of_revenue_1', {}).get('historical_account_key', '').lower()
                has_interest_in_cor = 'interest' in cor_historical_account
            except (KeyError, AttributeError):
                pass
                
            is_lender = is_asset_yield_model and has_interest_in_cor
            
            if is_lender:
                interest_sum = self.data_grid.loc['interest_on_revolver', year_col]
                logger.info(f"[{account}][{year_col}] Lender detected, only using revolver interest")
            else:
                try:
                    interest_sum = self.data_grid.loc['interest_on_revolver', year_col] + self.data_grid.loc['interest_expense', year_col]
                except KeyError:
                    # If interest_expense isn't calculated yet, just use revolver interest as a fallback
                    interest_sum = self.data_grid.loc['interest_on_revolver', year_col]
                    logger.warning(f"[{account}][{year_col}] interest_expense not found, using only revolver interest")
                    
            logger.info(f"[{account}][{year_col}] Operating Income: {operating_income:,.2f}, Interest: {interest_sum:,.2f}, EBT: {operating_income - interest_sum:,.2f}")
            return operating_income - interest_sum
        
        elif account == 'net_income':
            return self.data_grid.loc['ebt', year_col] - self.data_grid.loc['income_tax_expense', year_col]

        # --- Balance Sheet Items: Return ONLY the change for the period ---
        # Note: For BS items driven by formulas like DSO, we calculate the closing balance
        # and then subtract the opening balance to get the true delta.
        elif account == 'main_receivables':
            opening_balance = get_prior_value(account)
            driver = self._get_driver_value(schema_entry['drivers']['days_sales_outstanding'], p_year_index)
            closing_balance = (driver / 365.0) * self.data_grid.loc['total_revenue', year_col] if self.data_grid.loc['total_revenue', year_col] != 0 else 0.0
            return closing_balance - opening_balance

        elif account == 'main_inventory':
            opening_balance = get_prior_value(account)
            driver = self._get_driver_value(schema_entry['drivers']['days_inventory_held'], p_year_index)
            closing_balance = (driver / 365.0) * self.data_grid.loc['cost_of_revenue_1', year_col] if self.data_grid.loc['cost_of_revenue_1', year_col] != 0 else 0.0
            return closing_balance - opening_balance
            
        elif account == 'main_payables':
            opening_balance = get_prior_value(account)
            driver = self._get_driver_value(schema_entry['drivers']['days_payable_outstanding'], p_year_index)
            closing_balance = (driver / 365.0) * self.data_grid.loc['cost_of_revenue_1', year_col] if self.data_grid.loc['cost_of_revenue_1', year_col] != 0 else 0.0
            return closing_balance - opening_balance



        elif account == 'retained_earnings':
            # The change in retained earnings is Net Income minus Dividends Paid.
            net_income = self.data_grid.loc['net_income', year_col]
            # Access the dividend payout ratio using the correct, full path from the schema.
            dividends_paid = 0.0
            try:
                payout_ratio_driver = self.populated_schema['balance_sheet']['equity']['items']['retained_earnings']['drivers']['dividend_payout_ratio']
                payout_ratio = self._get_driver_value(payout_ratio_driver, p_year_index)
                dividends_paid = net_income * payout_ratio
            except KeyError:
                # If the driver is not defined in the schema, assume zero dividends.
                logger.warning(f"'dividend_payout_ratio' driver not found in schema for '{account}'. Assuming 0 dividends.")
            logger.info(f"[{account}][{year_col}] Net Income: {net_income:,.2f}, Dividends: {dividends_paid:,.2f}, Change: {net_income - dividends_paid:,.2f}")
            return net_income - dividends_paid

        elif account in self.ppe_accounts:
            # ASSET DELTA: Calculate change in a single PPE asset (Capex - Depreciation)
            capex_driver = self._get_driver_value(schema_entry['drivers']['capex_as_percent_of_revenue'], p_year_index)
            depreciation_driver = self._get_driver_value(schema_entry['drivers']['depreciation_rate'], p_year_index)
            capex = self.data_grid.loc['total_revenue', year_col] * capex_driver
            depreciation = get_prior_value(account) * depreciation_driver
            # Aggregate depreciation for this year
            logger.info(f"[DEBUG][{account}][{year_col}] Calculated depreciation value: {depreciation:,.2f}")
            self.aggregated_depreciation[year_col] = self.aggregated_depreciation.get(year_col, 0.0) + (depreciation * -1)
            logger.info(f"[DEBUG][{account}][{year_col}] Updated aggregated_depreciation for {year_col}: {self.aggregated_depreciation[year_col]:,.2f}")
            # Only aggregate if in a projection year (not historical)
            if year_col not in self.historical_years:
                self.aggregated_depreciation[year_col] = self.aggregated_depreciation.get(year_col, 0.0) + (depreciation * -1)
            return capex - depreciation

        elif account in self.intangible_asset_accounts:
            # ASSET DELTA: Calculate change in a single intangible asset (Additions - Amortization)
            additions_driver = self._get_driver_value(schema_entry['drivers']['intangible_additions_annual'], p_year_index)
            amortization_driver = self._get_driver_value(schema_entry['drivers']['amortization_rate'], p_year_index)
            additions = additions_driver
            amortization = get_prior_value(account) * amortization_driver
            # Aggregate amortization for this year
            logger.info(f"[DEBUG][{account}][{year_col}] Calculated amortization value: {amortization:,.2f}")
            self.aggregated_amortization[year_col] = self.aggregated_amortization.get(year_col, 0.0) + (amortization * -1)
            logger.info(f"[DEBUG][{account}][{year_col}] Updated aggregated_amortization for {year_col}: {self.aggregated_amortization[year_col]:,.2f}")
            if year_col not in self.historical_years:
                self.aggregated_amortization[year_col] = self.aggregated_amortization.get(year_col, 0.0) + (amortization * -1)
            return additions - amortization

        elif account == 'depreciation_expense':
            # Return the total depreciation aggregated during PPE asset delta calculations
            retrieved_value = self.aggregated_depreciation.get(year_col, 0.0)
            logger.info(f"[DEBUG][{account}][{year_col}] Retrieving aggregated depreciation. Value: {retrieved_value:,.2f}")
            return retrieved_value

        elif account == 'amortization_expense':
            # Return the total amortization aggregated during intangible asset delta calculations
            retrieved_value = self.aggregated_amortization.get(year_col, 0.0)
            logger.info(f"[DEBUG][{account}][{year_col}] Retrieving aggregated amortization. Value: {retrieved_value:,.2f}")
            return retrieved_value

        elif account in self.scheduled_debt_items:
            # Principal flow is NEGATIVE for repayments (reduces debt balance)
            principal_flow = self._get_driver_value(schema_entry['drivers']['scheduled_principal_flow'], p_year_index)
            # Log for debugging
            opening_balance = get_prior_value(account)
            logger.info(f"[{account}][{year_col}] Opening balance: {opening_balance:,.2f}, Principal flow: {principal_flow:,.2f}")
            return principal_flow
            
        elif account == 'common_stock':
            return self._get_driver_value(schema_entry['drivers']['net_share_issuance'], p_year_index)
        
        elif account == 'retained_earnings':
            # For retained earnings, we only return the change during THIS period
            # Net income and dividends for THIS period only
            payout_driver = self._get_driver_value(schema_entry['drivers']['dividend_payout_ratio'], p_year_index)
            net_income = self.data_grid.loc['net_income', year_col]
            dividends = max(0, net_income * payout_driver)
            retained_earnings_change = net_income - dividends
            logger.info(f"[retained_earnings][{year_col}] Net Income: {net_income:,.2f}, Dividends: {dividends:,.2f}, Change: {retained_earnings_change:,.2f}")
            return retained_earnings_change

        # If an account is not in the logic (e.g., revolver, cash), its delta is zero by default.
        # The main loop is responsible for their state management.
        return 0.0


    def _find_leaf_accounts(self, schema_section: dict) -> List[str]:
        """Extract the leaf account names from a schema section.
        
        Args:
            schema_section: A section of the schema containing items.
            
        Returns:
            A list of account name strings.
        """
        result = []
        items_dict = schema_section.get('items', {})
        
        for account_name, details in items_dict.items():
            # Check for historical account mapping
            historical_key = details.get('historical_account_key')
            
            # If there's a historical account key and it's in our data grid, use that instead
            if historical_key and historical_key in self.data_grid.index:
                logger.info(f"Mapping schema account '{account_name}' to historical account '{historical_key}'")
                result.append(historical_key)
            else:
                result.append(account_name)
        
        return result

    def _execute_projection_loop(self) -> None:
        """Main execution loop for the projections engine."""
        projection_cols = [col for col in self.data_grid.columns if isinstance(col, str) and col.startswith('P')]
        if not projection_cols:
            raise ValueError("No projection periods found in data grid.")
        
        logger.info("--- [START] Main Projection Loop ---")
        
        # Extract lists of accounts by type from the schema for future calculations
        schema_bs = self.populated_schema.get('balance_sheet', {})
        
        # Dynamically find all cash accounts from the schema
        cash_accounts = self._find_leaf_accounts(schema_bs.get('cash_and_equivalents', {}))
        if not cash_accounts:
            logger.warning("Could not find cash accounts in schema. Defaulting to ['cash'].")
            cash_accounts = ['cash']
        primary_cash_account = cash_accounts[0]

        non_cash_asset_items = self.receivables_accounts + \
                              self._find_leaf_accounts(schema_bs.get('inventory', {})) + \
                              self.ppe_accounts + \
                              self.intangible_asset_accounts + \
                              self._find_leaf_accounts(schema_bs.get('other_assets', {}))
        liabilities_no_revolver_items = self._find_leaf_accounts(schema_bs.get('liabilities', {}))
        liabilities_no_revolver_items = [item for item in liabilities_no_revolver_items if item != 'revolver']
        # Verify scheduled debt items are included in liabilities
        scheduled_debt_items = self.populated_schema['balance_sheet']['debt']['scheduled_debt'].get('items', [])
        for debt_item in scheduled_debt_items:
            if debt_item not in liabilities_no_revolver_items and debt_item != 'revolver':
                liabilities_no_revolver_items.append(debt_item)
        
        # Ensure we have all equity accounts
        equity_items = self._find_leaf_accounts(schema_bs.get('equity', {}))
        # Verify retained_earnings is included in equity
        if 'retained_earnings' not in equity_items:
            equity_items.append('retained_earnings')
            
        # Define complete list of balance sheet accounts
        balance_sheet_accounts = non_cash_asset_items + liabilities_no_revolver_items + equity_items + cash_accounts + ['revolver']
        # Make sure to include the historical plug reversal if it exists
        if '__historical_plug_reversal__' in self.data_grid.index:
            balance_sheet_accounts.append('__historical_plug_reversal__')

        circular_plan = self.execution_plan['circular']
        projection_cols = [col for col in self.data_grid.columns if isinstance(col, str) and col.startswith('P')]

        for p_index, year_col in enumerate(projection_cols, 1):
            logger.info(f"--- Projecting Year: {year_col} (Index: {p_index}) ---")
            prior_year_col = self._get_prior_period_col(year_col)

            # --- PHASE A: BALANCE SHEET ROLL-FORWARD ---
            logger.info(f"[Phase A] Rolling forward BS from {prior_year_col} to set opening balances for {year_col}.")
            self.data_grid.loc[balance_sheet_accounts, year_col] = self.data_grid.loc[balance_sheet_accounts, prior_year_col]

            # For the first projection year, neutralize any historical plugs *after* rolling forward.
            if p_index == 1:
                self._neutralize_historical_plugs()
            
            # Take a snapshot of the opening balances for this period's calculations.
            opening_balances = self.data_grid[year_col].copy()
            logger.info(f"[DEBUG][{year_col}] After Phase A (Opening Balances):")
            # Log all scheduled debt instruments dynamically
            scheduled_debt_items = self.populated_schema.get('balance_sheet', {}).get('debt', {}).get('scheduled_debt', {}).get('items', {}).keys()
            for debt_item in scheduled_debt_items:
                if debt_item in opening_balances.index:
                    logger.info(f"  - {debt_item}: {opening_balances.loc[debt_item]:,.2f}")
            logger.info(f"  - retained_earnings: {opening_balances.loc['retained_earnings']:,.2f}")
            
            # First calculate all revenue streams outside the circular loop to initialize them
            # Correctly identify ALL revenue streams from the dependency graph, not just circular ones.
            revenue_accounts = [node for node in self.graph.nodes() if node.startswith('revenue_stream')]
            logger.info(f"Calculating {len(revenue_accounts)} revenue streams first: {revenue_accounts}")
            for account in revenue_accounts:
                delta = self._calculate_delta(account, year_col, p_index, opening_balances)
                self.data_grid.loc[account, year_col] = delta
                logger.info(f"Revenue stream {account} for {year_col}: {delta:,.2f}")
            
            # Calculate total revenue based on revenue streams
            if 'total_revenue' in self.data_grid.index:
                total_revenue = self.data_grid.loc[[acc for acc in self.data_grid.index if acc.startswith('revenue_stream')], year_col].sum()
                self.data_grid.loc['total_revenue', year_col] = total_revenue
                logger.info(f"Total revenue for {year_col}: {total_revenue:,.2f}")

            # Pre-calculate all non-circular P&L items before the solver, in logical order.
            # This ensures the solver has the necessary inputs (e.g., gross_profit for operating_income).
            logger.info(f"Pre-calculating non-circular P&L items for {year_col}...")

            # --- [SURGICAL FIX START] ---
            # Calculate asset deltas to specifically populate the D&A aggregation dictionaries.
            # This is the minimal change required to fix the zero D&A issue.
            logger.info(f"[Pre-Solver] Calculating asset deltas to populate Depreciation & Amortization for {year_col}...")
            
            asset_accounts_for_da = self.ppe_accounts + self.intangible_asset_accounts
            for account in asset_accounts_for_da:
                if account in self.data_grid.index:
                    # Calculate the change (e.g., Capex - Depreciation)
                    delta = self._calculate_delta(account, year_col, p_index, opening_balances)
                    # Apply the change to the opening balance to get the closing balance
                    self.data_grid.loc[account, year_col] += delta

            # Now that the aggregation dictionaries are populated, calculate the P&L expense lines.
            self.data_grid.loc['depreciation_expense', year_col] = self._calculate_delta('depreciation_expense', year_col, p_index)
            self.data_grid.loc['amortization_expense', year_col] = self._calculate_delta('amortization_expense', year_col, p_index)
            
            logger.info(f"  - Calculated 'depreciation_expense': {self.data_grid.loc['depreciation_expense', year_col]:,.2f}")
            logger.info(f"  - Calculated 'amortization_expense': {self.data_grid.loc['amortization_expense', year_col]:,.2f}")
            # --- [SURGICAL FIX END] ---


            non_circular_pnl_accounts = ['cost_of_revenue_1', 'gross_profit', 'sga_expense', 'provision_for_credit_losses', 'operating_income']
            for account in non_circular_pnl_accounts:
                if account in self.data_grid.index:
                    delta = self._calculate_delta(account, year_col, p_index, opening_balances)
                    self.data_grid.loc[account, year_col] = delta
                    logger.info(f"  - Calculated {account}: {delta:,.2f}")

            # --- PHASE B: INTRA-PERIOD CIRCULAR SOLVER ---
            logger.info(f"[Phase B] Activating circular solver for {year_col} to calculate intra-period activity.")

            # Define which circular accounts are P&L vs. BS for the solver loop
            # Ensure critical P&L subtotals are explicitly included
            pl_change_accounts = [acc for acc in circular_plan if acc not in balance_sheet_accounts]
            # Explicitly add key P&L subtotal accounts
            essential_pl_subtotals = ['operating_income', 'ebt', 'net_income']
            for subtotal in essential_pl_subtotals:
                if subtotal not in pl_change_accounts and subtotal in circular_plan:
                    pl_change_accounts.append(subtotal)
            bs_change_accounts = [acc for acc in circular_plan if acc in balance_sheet_accounts]

            # Iteratively solve for circular dependencies
            for i in range(self.MAX_ITERATIONS_CIRCULAR):
                old_revolver_balance = self.data_grid.loc['revolver', year_col]

                # Define a stable, logical calculation order for the circular accounts to ensure convergence.
                # This is critical because a topological sort is not possible on a cycle.
                # Corrected Solver Order: Calculate all expenses before subtotals
                solver_order = [
                    'provision_for_credit_losses',
                    'interest_on_revolver',
                    'interest_expense',
                    'operating_income', # Depends on Gross Profit (pre-calc) and Provision for Credit Losses
                    'ebt', # Depends on Operating Income and Interest
                    'income_tax_expense',
                    'net_income',
                    'retained_earnings',
                ]

                # DEBUG: Log the solver order to verify interest_expense is included
                logger.info(f"[DEBUG][{year_col}] Circular solver accounts order: {solver_order}")
                logger.info(f"[DEBUG][{year_col}] Checking if interest_expense is in solver: {'interest_expense' in solver_order}")
                logger.info(f"[DEBUG][{year_col}] Checking if interest_expense exists in account_map: {'interest_expense' in self.account_map}")

                for account in solver_order:
                    if account not in circular_plan:
                        continue

                    # Add debug logging for interest_expense
                    if account == 'interest_expense':
                        logger.info(f"[DEBUG][{year_col}] About to calculate interest_expense...")
                    
                    # Calculate the delta for this account
                    try:
                        delta = self._calculate_delta(account, year_col, p_index, opening_balances)
                        
                        # Add debug logging after calculation
                        if account == 'interest_expense':
                            logger.info(f"[DEBUG][{year_col}] interest_expense calculated: {delta}")
                    except Exception as e:
                        logger.error(f"Error calculating delta for {account}: {e}")
                        logger.error(traceback.format_exc())
                        raise
                    
                    if account in pl_change_accounts:
                        self.data_grid.loc[account, year_col] = delta
                        logger.info(f"P&L account '{account}' calculated: {delta:,.2f}")
                    elif account in bs_change_accounts:
                        opening_balance = opening_balances.loc[account]
                        closing_balance = opening_balance + delta
                        self.data_grid.loc[account, year_col] = closing_balance
                        logger.info(f"BS account '{account}' updated: {opening_balance:,.2f} + {delta:,.2f} = {closing_balance:,.2f}")

                # --- [PLUG CALCULATION] --- #
                # Determine the funding requirement to balance the sheet, then solve for revolver and cash.
                total_non_cash_assets = self.data_grid.loc[non_cash_asset_items, year_col].sum()
                total_liabilities_no_revolver = self.data_grid.loc[liabilities_no_revolver_items, year_col].sum()
                total_equity = self.data_grid.loc[equity_items, year_col].sum()
                plug_reversal = self.data_grid.loc['__historical_plug_reversal__', year_col] if p_index == 1 and '__historical_plug_reversal__' in self.data_grid.index else 0.0

                # --- [PLUG CALCULATION - CORRECTED] --- #
                # 1. Calculate the cash position before any new revolver activity.
                # This is the sum of all cash accounts, calculated implicitly.
                pre_plug_cash_position = (total_liabilities_no_revolver + total_equity + plug_reversal) - total_non_cash_assets

                # 2. Determine target minimum cash.
                target_min_cash = self._calculate_delta('min_cash_balance', year_col, p_index, opening_balances)

                # 3. Compare pre-plug cash to the minimum target to determine shortfall or surplus.
                if pre_plug_cash_position < target_min_cash:
                    # Shortfall: Draw on revolver to meet minimum cash.
                    revolver_draw = target_min_cash - pre_plug_cash_position
                    final_revolver = old_revolver_balance + revolver_draw
                    final_cash = target_min_cash
                else:
                    # Surplus: Use excess cash to pay down the revolver.
                    cash_surplus = pre_plug_cash_position - target_min_cash
                    revolver_paydown = min(cash_surplus, old_revolver_balance)
                    final_revolver = old_revolver_balance - revolver_paydown
                    # Final cash is the pre-plug position less the amount used for paydown.
                    final_cash = pre_plug_cash_position - revolver_paydown

                self.data_grid.loc[primary_cash_account, year_col] = final_cash
                self.data_grid.loc['revolver', year_col] = final_revolver
                new_revolver_balance = final_revolver

                if abs(new_revolver_balance - old_revolver_balance) < self.CONVERGENCE_TOLERANCE:
                    logger.info(f"Circular solver converged in {i+1} iterations for {year_col}. Revolver Delta: {abs(new_revolver_balance - old_revolver_balance):.2f}")
                    converged = True
                    break
            
            if not converged:
                logger.warning(f"Circular solver did not converge after {self.MAX_ITERATIONS_CIRCULAR} iterations for {year_col}.")
                logger.warning(f"Final revolver delta: {abs(new_revolver_balance - old_revolver_balance):.2f}")
                # Continue execution rather than failing entirely

            # --- Final Validation ---
            logger.info(f"[DEBUG][{year_col}] Final State After Convergence:")
            # Log all scheduled debt instruments dynamically
            scheduled_debt_items = self.populated_schema.get('balance_sheet', {}).get('debt', {}).get('scheduled_debt', {}).get('items', {}).keys()
            for debt_item in scheduled_debt_items:
                if debt_item in self.data_grid.index:
                    logger.info(f"  - {debt_item}: {self.data_grid.loc[debt_item, year_col]:,.2f}")
            logger.info(f"  - retained_earnings: {self.data_grid.loc['retained_earnings', year_col]:,.2f}")
            logger.info(f"  - revolver: {self.data_grid.loc['revolver', year_col]:,.2f}")
            total_cash = self.data_grid.loc[cash_accounts, year_col].sum()
            logger.info(f"  - total_cash: {total_cash:,.2f} (from {', '.join(cash_accounts)})")
            
            # Enhanced balance sheet check with component details
            total_non_cash_assets = self.data_grid.loc[non_cash_asset_items, year_col].sum()
            total_assets = total_non_cash_assets + total_cash
            
            total_liabilities_no_revolver = self.data_grid.loc[liabilities_no_revolver_items, year_col].sum()
            revolver = self.data_grid.loc['revolver', year_col]
            total_liabilities = total_liabilities_no_revolver + revolver
            
            total_equity = self.data_grid.loc[equity_items, year_col].sum()
            bs_check = total_assets - (total_liabilities + total_equity)
            
            logger.info(f"Year {year_col} projection complete. Final BS Check (Assets - (L+E)): {bs_check:,.2f}")
            logger.info(f"  - Total Assets: {total_assets:,.2f} (Non-Cash: {total_non_cash_assets:,.2f}, Cash: {total_cash:,.2f})")
            logger.info(f"  - Total Liabilities: {total_liabilities:,.2f} (Non-Revolver: {total_liabilities_no_revolver:,.2f}, Revolver: {revolver:,.2f})")
            logger.info(f"  - Total Equity: {total_equity:,.2f}")

            if abs(bs_check) > self.CONVERGENCE_TOLERANCE * 10:
                logger.warning(f"Significant balance sheet discrepancy detected in {year_col}.")

        logger.info("--- [END] Main Projection Loop Finished ---")

    def project(self) -> pd.DataFrame:
        """
        The main public method to orchestrate the entire projection process.
        """
        logger.info(f"--- Starting Projection for {self.security_id} ---")
        
        # --- Phase I: Setup & Compilation ---
        logger.info("[1/5] Loading and filtering inputs...")
        self._load_and_filter_historicals()

        # --- Comprehensive Account Discovery ---
        # Discover all dynamic accounts upfront and store as class attributes. This must be done
        # after loading the schema but before any logic that relies on these lists.
        schema_bs = self.populated_schema.get('balance_sheet', {})
        self.receivables_accounts = self._find_leaf_accounts(schema_bs.get('accounts_receivable', {}))
        self.ppe_accounts = self._find_leaf_accounts(schema_bs.get('property_plant_equipment', {}))
        self.intangible_asset_accounts = self._find_leaf_accounts(schema_bs.get('intangible_assets', {}))

        schema_is = self.populated_schema.get('income_statement', {})
        self.cost_of_revenue_items = self._find_leaf_accounts(schema_is.get('cost_of_revenue', {}))
        self.operating_expense_items = self._find_leaf_accounts(schema_is.get('operating_expenses', {}))
        self.scheduled_debt_items = self._find_leaf_accounts(schema_bs.get('debt', {}).get('scheduled_debt', {}))
        
        logger.info("[2/5] Initializing data grid...")
        self._create_data_grid()
        # TODO: This is where the real data population from historicals would go.
        # This part is highly dependent on the final JSON structure and is complex.
        # For this example, we assume it's populated for the logic to be demonstrated.
        
        logger.info("[2.5/5] Resolving historical source conflicts...")
        self._resolve_historical_source_conflicts()
        
        logger.info("[3/5] Sanitizing data...")
        
        logger.info("[4/5] Building and compiling execution graph...")
        self._build_and_compile_graph() # This populates self.execution_plan
        
        logger.info("[5/5] Calculating revolver interest rate...")
        self._calculate_revolver_rate() # Populates self.revolver_interest_rate

        # --- Phase II: Projection Loop ---
        logger.info("--- Executing Projection Loop ---")
        # Initialize aggregation dictionaries for this projection run.
        self.aggregated_depreciation = {}
        self.aggregated_amortization = {}
        logger.info("[DEBUG] Initialized/reset aggregation dictionaries for new projection run.")
        self._execute_projection_loop()

        # --- Phase III: Final Articulation ---
        # logger.info("--- Articulating Final Cash Flow Statement ---")
        # self._articulate_cfs()
        
        logger.info(f"--- Projection for {self.security_id} Complete ---")
        
        # Save the output
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