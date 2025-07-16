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
        self.is_lender = False

        self.deterministic_depreciation_rate = 0.0
        self.deterministic_amortization_rate = 0.0
        self.unit_economics_tracker = {} # Will store derived historical units/prices
        self.aggregated_depreciation = {}
        self.aggregated_amortization = {}
        self.primary_cash_account = None
        self.CASH_SWEEP_PERCENTAGE = 1.0 # Using 100% of surplus cash

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
    
    # ===================================================================
    # STAGE 1: DATA GRID SANITIZATION - MASTER ORCHESTRATOR
    # ===================================================================
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

    # ===================================================================
    # STAGE 1, PART A: HISTORICAL DATA DISAGGREGATION
    # ===================================================================
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
            
    # ===================================================================
    # STAGE 1, PART B: T+1 PROJECTION BASELINING (REPLACEMENT METHOD)
    # ===================================================================
    def _neutralize_historical_plugs(self, p1_col: str) -> None:
        """
        Identifies summation plugs in the final historical balance sheet (T-1)
        and posts a one-time, offsetting DOUBLE-ENTRY adjustment in the first
        projection period (P1) to ensure the projection starts from a perfectly
        balanced state.
        
        This method has side-effects: it modifies the p1_col of self.data_grid.
        
        Args:
            p1_col: The string name of the first projection column (e.g., "P1").
        
        Raises:
            RuntimeError: If the raw data for the anchor year cannot be found.
            KeyError: If the required '__historical_plug_reversal__' account is missing.
        """
        logger.info(f"[P1 Baseline] Starting historical plug neutralization for anchor year {self.T1_year}...")

        # Find the raw data dictionary for the anchor year (T-1)
        try:
            t1_index = self.historical_years.index(self.T1_year)
            t1_data_period = self.historical_data_raw[t1_index]
        except (ValueError, IndexError):
            # FAIL FAST & LOUD principle
            raise RuntimeError(f"FATAL: Could not find raw data for anchor year {self.T1_year}. Internal state is inconsistent.")

        # Safely access the balance sheet summation plugs.
        bs_plugs = t1_data_period.get('balance_sheet', {}).get('summation_plugs', {})
        if not bs_plugs:
            logger.info(f"[P1 Baseline] No balance sheet summation plugs found in T-1. No neutralization needed.")
            return

        reversal_account = '__historical_plug_reversal__'
        if reversal_account not in self.data_grid.index:
            # FAIL FAST & LOUD principle
            raise KeyError(f"FATAL: The required '{reversal_account}' account is not in the data_grid index.")

        total_offset_to_equity = 0.0
        for plug_item, plug_value in bs_plugs.items():
            if plug_item == '__accounting_equation__':
                continue # This is a reporting plug, not a balancing plug.
                
            if not isinstance(plug_value, (int, float)) or plug_value == 0:
                continue

            # The core logic: a robust, self-balancing double-entry adjustment.
            # Entry 1: Reverse the plug on the asset side to fix the subtotal.
            reversal_amount = -float(plug_value)
            self.data_grid.loc[reversal_account, p1_col] += reversal_amount
            
            # Entry 2: Post the equal and opposite amount to equity to keep A=L+E balanced.
            self.data_grid.loc['retained_earnings', p1_col] -= reversal_amount
            total_offset_to_equity -= reversal_amount

            logger.info(
                f"[P1 Baseline] Neutralizing '{plug_item}' plug of {plug_value:,.2f}. "
                f"Posted {reversal_amount:,.2f} to '{reversal_account}' and {-reversal_amount:,.2f} to 'retained_earnings'."
            )
        
        if total_offset_to_equity != 0.0:
            logger.info(f"[P1 Baseline] Total one-time adjustment to opening Retained Earnings: {total_offset_to_equity:,.2f}.")
        else:
            logger.info("[P1 Baseline] Plug structure found, but all values were zero. No adjustments made.")

    # --- Private Helper Methods for Grid Creation ---

    def _build_account_map(self) -> dict:
        """
        Recursively traverses the schema to build a flat map of account names to their
        schema definition, parent statement, and parent CATEGORY.
        It now correctly includes subtotal landmarks regardless of historical mapping.
        """
        account_map = {}
        
        def recurse(d, statement_name, parent_key=""):
            for key, value in d.items():
                if not isinstance(value, dict): continue

                is_subtotal_category = parent_key.startswith('subtotals')
                has_hist_key = value.get('historical_account_key') is not None
                
                # An item is added to the map if it's in a subtotal category OR has a valid hist_key
                if is_subtotal_category or has_hist_key:
                    if key in account_map:
                        raise ValueError(f"Duplicate account key '{key}' found. Account names must be unique.")
                    
                    # Determine category: for items in 'items', it's the parent key, otherwise the current key
                    category_name = parent_key if key in value.get('items', {}) else key
                    account_map[key] = {'schema_entry': value, 'statement': statement_name, 'category': category_name}

                # Continue traversal, passing the current key as the new parent_key
                if key != 'items': # Avoid paths like 'revenue.items'
                    recurse(value, statement_name, key)
                else:
                    recurse(value, statement_name, parent_key) # For items in 'items', keep parent category

        # Start recursion from the top-level statements
        for statement in ['income_statement', 'balance_sheet']:
            if statement in self.populated_schema:
                recurse(self.populated_schema[statement], statement, statement)
        
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
            'revolver',
            '__historical_plug_reversal__',
            'interest_on_revolver',
            'interest_income_on_cash',
            'total_revenue', # Added as a core engine aggregate
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

    # ===================================================================
    # STAGE 2: DYNAMIC DEPENDENCY GRAPH CONSTRUCTION (FINAL REPLACEMENT)
    # ===================================================================
    def _build_and_compile_graph(self) -> None:
        """
        Builds a dependency graph by dynamically interpreting the populated schema,
        enforcing financial logic rules, identifying the circular group via SCC analysis,
        and compiling a deterministic execution plan. This is the "brain" of the engine.
        """
        logger.info("[Graph] Initializing dynamic dependency graph construction...")
        self.graph = nx.DiGraph()

        # Step 1: Create a node for every single projectable item in the schema.
        logger.info("[Graph] Step 1/4: Populating nodes from schema...")
        self._populate_nodes_from_schema()
        logger.info(f"[Graph] Created {self.graph.number_of_nodes()} initial nodes from schema and engine internals.")

        # Step 2: Diagnose company profile (Lender vs. Non-Lender).
        # This is the new, correct location for this logic block.
        logger.info("[Graph] Step 2/4: Diagnosing company profile...")
        # Safely check each revenue stream for the 'is_lender' flag.
        for stream_name in self._get_schema_items('income_statement.revenue'):
            # This logic is moved from the linking function to here.
            model = self.populated_schema['income_statement']['revenue']['items'][stream_name].get('projection_configuration', {}).get('selected_model', {})
            if model and model.get('is_lender') is True:
                self.is_lender = True
                break # Found one, no need to check further.
        logger.info(f"[Graph] Company Profile Diagnosis: is_lender = {self.is_lender}")
        
        # Step 3: Link nodes based on the now-confirmed diagnosis.
        logger.info("[Graph] Step 3/4: Linking node dependencies based on schema rules...")
        self._link_graph_dependencies()
        logger.info(f"[Graph] Dependency linking complete. Graph has {self.graph.number_of_edges()} edges.")
        
        # Step 4: Isolate circularities and compile the final execution plan.
        logger.info("[Graph] Step 4/4: Performing SCC analysis and compiling execution plan...")
        sccs = list(nx.strongly_connected_components(self.graph))
        circular_components = [scc for scc in sccs if len(scc) > 1]

        if not circular_components:
            raise RuntimeError("FATAL: No circular dependencies found. The core revolver-interest loop is missing from the graph logic.")
        if len(circular_components) > 1:
            circular_groups_str = "; ".join([str(c) for c in circular_components])
            raise RuntimeError(f"FATAL: Found {len(circular_components)} disjoint circular dependency groups. Engine is designed for one unified group only. Groups found: {circular_groups_str}")
        
        self.circular_group = circular_components[0]
        logger.info(f"[Graph] Successfully identified the single circular dependency group: {self.circular_group}")

        pre_circular_nodes = [n for n in self.graph.nodes() if n not in self.circular_group]
        try:
            pre_circular_subgraph = self.graph.subgraph(pre_circular_nodes)
            pre_circular_plan = list(nx.topological_sort(pre_circular_subgraph))
        except nx.NetworkXUnfeasible as e:
            raise RuntimeError(f"FATAL: Failed to create a topological sort of the pre-circular graph. Check for unintended cycles. Details: {e}")

        self.execution_plan = {
            'pre_circular': pre_circular_plan,
            'circular': list(self.circular_group)
        }
        logger.info(f"[Graph] Execution plan compiled: {len(pre_circular_plan)} pre-circular steps, {len(self.circular_group)} circular steps.")
        logger.info("[Graph] Graph construction complete.")

    # ===================================================================
    # STAGE 2: HELPER METHODS FOR GRAPH CONSTRUCTION (FINAL REPLACEMENTS)
    # ===================================================================
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
                'revolver', '__historical_plug_reversal__', 'interest_on_revolver',
                'interest_income_on_cash', 'total_revenue'
            ]
            self.graph.add_nodes_from(engine_nodes)

    def _link_graph_dependencies(self):
        """
        Systematically creates edges in the dependency graph based on the engine's
        hardcoded understanding of the universal schema's structure and vocabulary.
        This version is hardened to explicitly handle all defined projection models.
        """
        logger.info("[Graph Link] Linking dependencies...")

        # --- Sub-Task 1: Diagnose Profile & Link Revenue/COGS ---
        revenue_items = self._get_schema_items('income_statement.revenue')
        cogs_items = self._get_schema_items('income_statement.cost_of_revenue')
        debt_items = self._get_schema_items('balance_sheet.liabilities.non_current_liabilities.long_term_debt')
        
        for stream in revenue_items:
            self.graph.add_edge(stream, 'total_revenue')
            model = self.populated_schema['income_statement']['revenue']['items'][stream].get('projection_configuration', {}).get('selected_model', {})
            if not model:
                raise RuntimeError(f"FATAL: Revenue stream '{stream}' has no 'selected_model'. Schema is incomplete.")

            model_name = model.get('model_name')
            if model_name == 'Asset_Yield_Driven':
                target_hist_key = model.get('target_asset_account')
                target_node = next((name for name, entry in self.account_map.items() if entry['schema_entry'].get('historical_account_key') == target_hist_key), None)
                if not target_node: raise RuntimeError(f"FATAL: Revenue model for '{stream}' depends on '{target_hist_key}', but no item is mapped to this key.")
                self.graph.add_edge(target_node, stream)
                logger.info(f"[Graph Link] Revenue: Linked '{stream}' to asset '{target_node}'.")

            elif model_name == 'Unit_Economics':
                # EXPLICIT HANDLING FOR UNIT ECONOMICS
                logger.info(f"[Graph Link] Revenue: Found 'Unit_Economics' model for '{stream}'. Treating as a source node (no intra-period dependencies).")
            
            else:
                # FAIL FAST for any unknown or null model names
                raise RuntimeError(f"FATAL: Unknown or null revenue model_name '{model_name}' for stream '{stream}'.")

        # Link COGS based on diagnosed Lender status
        if self.is_lender:
            logger.info("[Graph Link] Lender Logic: Linking COGS directly to interest-bearing liabilities.")
            for debt_item in debt_items:
                for cogs_item in cogs_items:
                    self.graph.add_edge(debt_item, cogs_item)
        else:
            logger.info("[Graph Link] Standard Logic: Linking COGS to total_revenue.")
            for cogs_item in cogs_items:
                self.graph.add_edge('total_revenue', cogs_item)

        # --- Sub-Task 2: Link Core P&L Hierarchy ---
        logger.info("[Graph Link] P&L: Linking core hierarchy.")
        self.graph.add_edge('total_revenue', 'gross_profit')
        for item in cogs_items: self.graph.add_edge(item, 'gross_profit')
        
        self.graph.add_edge('gross_profit', 'operating_income')
        for landmark in self._get_schema_items('income_statement.operating_expenses'):
            self.graph.add_edge(landmark, 'operating_income')
        for item in self._get_schema_items('income_statement.industry_specific_operating_expenses'):
            self.graph.add_edge(item, 'operating_income')

        self.graph.add_edge('operating_income', 'ebt')
        self.graph.add_edge('interest_on_revolver', 'ebt')
        self.graph.add_edge('interest_income_on_cash', 'ebt')
        if not self.is_lender:
            self.graph.add_edge('interest_expense', 'ebt')
        
        self.graph.add_edge('ebt', 'income_tax_expense')
        self.graph.add_edge('income_tax_expense', 'net_income')

        # --- Sub-Task 3: Link Driver-Based & Identity Dependencies ---
        logger.info("[Graph Link] Drivers & Identities: Linking specific calculation dependencies.")
        self.graph.add_edge('total_revenue', 'sga_expense')
        for item in self._get_schema_items('balance_sheet.assets.non_current_assets.property_plant_equipment'):
            self.graph.add_edge('total_revenue', item)
            self.graph.add_edge(item, 'depreciation_expense')

        for item in self._get_schema_items('balance_sheet.assets.non_current_assets.intangible_assets'):
            self.graph.add_edge(item, 'amortization_expense')
        
        if not self.is_lender:
            for item in debt_items:
                self.graph.add_edge(item, 'interest_expense')
        
        for cash_item in self._get_schema_items('balance_sheet.assets.current_assets.cash_and_equivalents'):
             self.graph.add_edge(cash_item, 'interest_income_on_cash')
        
        self.graph.add_edge('net_income', 'retained_earnings')

        # --- Sub-Task 4: Link Revolver Dependencies ---
        logger.info("[Graph Link] Revolver: Linking all BS items and cash policy to the Revolver.")
        self.graph.add_edge('revolver', 'interest_on_revolver')
        bs_nodes = [n for n, d in self.account_map.items() if d['statement'] == 'balance_sheet' and n != 'revolver']
        for node in bs_nodes:
            self.graph.add_edge(node, 'revolver')
        self.graph.add_edge('total_revenue', 'revolver')

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

#<--- PASTE/REPLACE THIS CODE INSIDE YOUR ProjectionsEngine CLASS --->

    # =====================================================================
    # STAGE 2.5: PRE-CALCULATION OF CONSTANTS & BASELINES (MASTER METHOD)
    # =====================================================================
    def _pre_calculate_constants(self) -> None:
        """
        Master orchestrator for all one-time calculation tasks that must happen
        after the graph is built but before the projection loop begins.
        This populates the engine with the necessary rates and derived histories.
        """

        self._designate_primary_accounts()

        # Task 1: Determine the interest rate for the revolver.
        self._calculate_revolver_rate()
        
        # Task 2: Calculate deterministic D&A rates from historicals.
        self._calculate_deterministic_da_rates()
        
        # Task 3: If applicable, derive historical units and prices for Unit Economics models.
        self._perform_unit_economics_factor_decomposition()

        self._calculate_historical_working_capital_ratios()
        
        logger.info("[Pre-Calc] All projection constants and baselines are prepared.")

    # ===================================================================
    # STAGE 2.5, PART A: REVOLVER RATE (REPLACEMENT METHOD)
    # ===================================================================
    def _calculate_revolver_rate(self) -> None:
        """
        Calculates the interest rate for the revolver by reading the aggregate
        interest rate driver directly from its correct location in the schema.
        
        This method has a side-effect: it populates self.revolver_interest_rate.
        """
        logger.info("[Pre-Calc] Determining revolver interest rate from schema...")
        try:
            # Navigate safely to the correct driver location.
            driver_obj = self.populated_schema['balance_sheet']['liabilities']['non_current_liabilities']['long_term_debt']['drivers']['average_interest_rate']
            baseline_rate = driver_obj.get('baseline')

            if baseline_rate is None or not isinstance(baseline_rate, (int, float)):
                raise ValueError("'baseline' value is missing or not a number.")
            
            self.revolver_interest_rate = float(baseline_rate)
            logger.info(f"[Pre-Calc] Revolver interest rate set to {self.revolver_interest_rate:.4%} based on the aggregate long-term debt driver.")

        except (KeyError, ValueError) as e:
            # FAIL FAST & LOUD
            raise RuntimeError(f"FATAL: Could not determine revolver interest rate from schema. The path or driver is missing/invalid. Path: balance_sheet.liabilities.non_current_liabilities.long_term_debt.drivers.average_interest_rate. Error: {e}")

    # ===================================================================
    # STAGE 2.5, PART B: DETERMINISTIC D&A RATES (NEW METHOD)
    # ===================================================================
    def _calculate_deterministic_da_rates(self) -> None:
        """
        Calculates historical blended depreciation and amortization rates.
        This makes D&A projection deterministic and independent of LLM input.
        
        This method has side-effects: it populates self.deterministic_depreciation_rate
        and self.deterministic_amortization_rate.
        """
        logger.info("[Pre-Calc] Calculating deterministic D&A rates from historicals...")
        
        # --- Depreciation Rate ---
        depr_series = self.data_grid.loc['depreciation_expense', self.historical_years]
        ppe_items = self._get_schema_items('balance_sheet.assets.non_current_assets.property_plant_equipment')
        # Sum up all PPE items for each historical year to get a total PPE series
        total_ppe_series = self.data_grid.loc[ppe_items, self.historical_years].sum()

        avg_depr_expense = abs(depr_series.mean()) # Use absolute value as expense is negative
        avg_ppe_balance = total_ppe_series.mean()

        if avg_ppe_balance > 0:
            self.deterministic_depreciation_rate = avg_depr_expense / avg_ppe_balance
            logger.info(f"[Pre-Calc] Calculated deterministic depreciation rate: {self.deterministic_depreciation_rate:.4%} (Avg Expense: {avg_depr_expense:,.0f} / Avg Asset: {avg_ppe_balance:,.0f})")
        else:
            self.deterministic_depreciation_rate = 0.10 # Hardcoded, reasonable default
            logger.warning(f"[Pre-Calc] Average PPE balance is zero or negative. Cannot calculate historical rate. Defaulting depreciation rate to {self.deterministic_depreciation_rate:.0%}.")

        # --- Amortization Rate ---
        amort_series = self.data_grid.loc['amortization_expense', self.historical_years]
        intangible_items = self._get_schema_items('balance_sheet.assets.non_current_assets.intangible_assets')
        total_intangible_series = self.data_grid.loc[intangible_items, self.historical_years].sum()
        
        avg_amort_expense = abs(amort_series.mean())
        avg_intangible_balance = total_intangible_series.mean()

        if avg_intangible_balance > 0:
            self.deterministic_amortization_rate = avg_amort_expense / avg_intangible_balance
            logger.info(f"[Pre-Calc] Calculated deterministic amortization rate: {self.deterministic_amortization_rate:.4%} (Avg Expense: {avg_amort_expense:,.0f} / Avg Asset: {avg_intangible_balance:,.0f})")
        else:
            self.deterministic_amortization_rate = 0.20 # Intangibles often amortize faster
            logger.warning(f"[Pre-Calc] Average intangible asset balance is zero or negative. Cannot calculate historical rate. Defaulting amortization rate to {self.deterministic_amortization_rate:.0%}.")

    # ===================================================================
    # STAGE 2.5, PART C: UNIT ECONOMICS DECOMPOSITION (NEW METHOD)
    # ===================================================================
    def _perform_unit_economics_factor_decomposition(self) -> None:
        """
        For any revenue stream using the Unit_Economics model, this method derives
        a consistent historical baseline for unit and price indices using the
        "Factor Decomposition" technique.
        
        This method has a side-effect: it populates self.unit_economics_tracker.
        """
        logger.info("[Pre-Calc] Checking for Unit Economics models to perform factor decomposition...")
        
        if len(self.historical_years) < 2:
            # This method requires at least two data points to work.
            logger.warning("[Pre-Calc] Skipping factor decomposition: requires at least two historical years of data.")
            return

        ue_streams = []
        for stream in self._get_schema_items('income_statement.revenue'):
            model = self.populated_schema['income_statement']['revenue']['items'][stream].get('projection_configuration', {}).get('selected_model', {})
            if model and model.get('model_name') == 'Unit_Economics':
                ue_streams.append(stream)

        if not ue_streams:
            logger.info("[Pre-Calc] No Unit Economics models found. Skipping decomposition.")
            return
            
        logger.info(f"[Pre-Calc] Found {len(ue_streams)} Unit Economics stream(s): {', '.join(ue_streams)}. Deriving historical indices...")

        # Anchor on the two most recent historical years
        t1_year, t2_year = self.historical_years[-1], self.historical_years[-2]

        for stream in ue_streams:
            try:
                # Step 1: Get required inputs
                total_revenue_t1 = self.data_grid.loc[stream, t1_year]
                total_revenue_t2 = self.data_grid.loc[stream, t2_year]
                drivers = self.populated_schema['income_statement']['revenue']['items'][stream]['projection_configuration']['selected_model']['drivers']
                baseline_unit_growth = drivers['unit_growth']['baseline']
                
                # Step 2: Anchor the final historical period (T-1)
                price_index_t1 = 1.0
                if total_revenue_t1 == 0: # Avoid division by zero
                     unit_index_t1 = 0.0
                else:
                     unit_index_t1 = total_revenue_t1 / price_index_t1
                
                # Step 3: Back-calculate the prior historical period (T-2)
                unit_index_t2 = unit_index_t1 / (1 + baseline_unit_growth)
                if unit_index_t2 == 0: # Avoid division by zero
                    price_index_t2 = 0.0
                else:
                    price_index_t2 = total_revenue_t2 / unit_index_t2
                
                # Step 4: Store the derived historical indices for the projection loop to use
                self.unit_economics_tracker[stream] = {
                    'units_history': [unit_index_t2, unit_index_t1],
                    'price_history': [price_index_t2, price_index_t1]
                }
                logger.info(f"[Pre-Calc] Decomposition for '{stream}' complete. T-1 (Units: {unit_index_t1:,.1f}, Price Idx: {price_index_t1:.3f}), T-2 (Units: {unit_index_t2:,.1f}, Price Idx: {price_index_t2:.3f})")

            except (KeyError, ZeroDivisionError) as e:
                raise RuntimeError(f"FATAL: Failed to perform factor decomposition for stream '{stream}'. Check schema drivers or historical data. Error: {e}")

    # ===================================================================
    # STAGE 3, PART A: ACCOUNT MAP ENHANCEMENT (NEW METHOD)
    # ===================================================================

    # --- Constants for the Projection Logic ---
    MAX_ITERATIONS_CIRCULAR = 100
    CONVERGENCE_TOLERANCE = 1.00 # Converged when the change is less than $1.00

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

    def _calculate_asset_yield_revenue(self, account: str, year_col: str, p_year_index: int, current_period_data: pd.Series, opening_balances: pd.Series) -> float:
        """Calculates revenue for an Asset_Yield_Driven model."""
        model = self.account_map[account]['schema_entry']['projection_configuration']['selected_model']
        yield_driver = model['driver']['asset_yield']
        
        # Get the yield for the current projection year
        current_yield = self._get_driver_value(yield_driver, p_year_index)
        
        # Find the target asset node name from the historical key
        target_hist_key = model.get('target_asset_account')
        target_node = next((name for name, entry in self.account_map.items() if entry['schema_entry'].get('historical_account_key') == target_hist_key), None)
        if not target_node:
            raise RuntimeError(f"FATAL: Revenue model for '{account}' depends on '{target_hist_key}', but no item is mapped to this key.")
        
        # Calculate average balance of the target asset
        opening_asset_balance = opening_balances.loc[target_node]
        closing_asset_balance = self.data_grid.loc[target_node, year_col] # This assumes the asset has been projected
        avg_asset_balance = (opening_asset_balance + closing_asset_balance) / 2
        
        revenue = avg_asset_balance * current_yield
        logger.info(f"[{account}][{year_col}] Asset Yield: {current_yield:.4%}, Avg Asset '{target_node}': {avg_asset_balance:,.0f}, Revenue: {revenue:,.0f}")
        return revenue

    def _calculate_lender_cogs(self, account: str, year_col: str, p_year_index: int, current_period_data: pd.Series, opening_balances: pd.Series) -> float:
        """Calculates COGS for a Lender, which is their primary interest expense."""
        # For a lender, COGS is their funding cost, driven by the aggregate debt driver
        drivers = self.populated_schema['balance_sheet']['liabilities']['non_current_liabilities']['long_term_debt']['drivers']
        rate_driver = drivers['average_interest_rate']
        
        current_rate = self._get_driver_value(rate_driver, p_year_index)
        
        # Sum the average balance of all interest-bearing debt items
        debt_items = self._get_schema_items('balance_sheet.liabilities.non_current_liabilities.long_term_debt')
        total_avg_debt_balance = 0
        for debt_item in debt_items:
            opening_debt = opening_balances.loc[debt_item]
            closing_debt = self.data_grid.loc[debt_item, year_col] # Assumes debt has been projected
            total_avg_debt_balance += (opening_debt + closing_debt) / 2
        
        # Expense should be negative
        cogs = total_avg_debt_balance * current_rate * -1
        logger.info(f"[{account}][{year_col}] Lender Funding Cost: Rate: {current_rate:.4%}, Avg Debt: {total_avg_debt_balance:,.0f}, COGS: {cogs:,.0f}")
        return cogs
        
    def _calculate_ppe_rollforward(self, account: str, year_col: str, p_year_index: int, opening_balances: pd.Series) -> float:
        """Calculates the CHANGE in a single PP&E item for the period (Capex - Depreciation)."""
        schema_entry = self.account_map[account]['schema_entry']
        
        # Calculate Capex for the period
        capex_driver = schema_entry['drivers']['capex_as_percent_of_revenue']
        capex_rate = self._get_driver_value(capex_driver, p_year_index)
        capex = self.data_grid.loc['total_revenue', year_col] * capex_rate
        
        # Calculate Depreciation for the period using our deterministic rate
        opening_balance = opening_balances.loc[account]
        depreciation = opening_balance * self.deterministic_depreciation_rate
        
        # Aggregate this period's depreciation amount for the P&L. The expense is negative.
        self.aggregated_depreciation[year_col] = self.aggregated_depreciation.get(year_col, 0.0) + (depreciation * -1)
        
        change_in_ppe = capex - depreciation
        logger.info(f"[{account}][{year_col}] PP&E Δ: Capex: {capex:,.0f} - Depr: {depreciation:,.0f} = {change_in_ppe:,.0f}")
        return change_in_ppe

    def _calculate_intangible_rollforward(self, account: str, year_col: str, p_year_index: int, opening_balances: pd.Series) -> float:
        """Calculates the CHANGE in a single Intangible Asset for the period (Additions - Amortization)."""
        schema_entry = self.account_map[account]['schema_entry']

        # Get annual additions for the period
        additions_driver = schema_entry['drivers']['intangible_additions_annual']
        additions = self._get_driver_value(additions_driver, p_year_index)
        
        # Calculate Amortization for the period using our deterministic rate
        opening_balance = opening_balances.loc[account]
        amortization = opening_balance * self.deterministic_amortization_rate
        
        # Aggregate this period's amortization amount for the P&L. The expense is negative.
        self.aggregated_amortization[year_col] = self.aggregated_amortization.get(year_col, 0.0) + (amortization * -1)
        
        change_in_intangibles = additions - amortization
        logger.info(f"[{account}][{year_col}] Intangible Δ: Additions: {additions:,.0f} - Amort: {amortization:,.0f} = {change_in_intangibles:,.0f}")
        return change_in_intangibles
        
    def _calculate_industry_specific_asset_growth(self, account: str, year_col: str, p_year_index: int, opening_balances: pd.Series) -> float:
        """Calculates the CHANGE for an industry-specific asset driven by a growth rate."""
        schema_entry = self.account_map[account]['schema_entry']
        
        # Get the growth rate for the current projection year
        growth_driver = schema_entry['drivers']['industry_specific_asset_growth_rate']
        growth_rate = self._get_driver_value(growth_driver, p_year_index)
        
        opening_balance = opening_balances.loc[account]
        closing_balance = opening_balance * (1 + growth_rate)
        
        change_in_asset = closing_balance - opening_balance
        logger.info(f"[{account}][{year_col}] Asset Growth Δ: Rate: {growth_rate:.4%}, Opening: {opening_balance:,.0f}, Change: {change_in_asset:,.0f}")
        return change_in_asset

    def _calculate_capital_structure_and_debt(self, account: str, year_col: str, p_year_index: int, current_period_data: pd.Series, opening_balances: pd.Series) -> float:
        """
        Calculates the required CHANGE in a single debt instrument based on a
        proportional allocation of the total required change in the company's
        capital structure.
        """
        # --- Step 1: Calculate Aggregate Values ---

        # First, ensure all non-debt, non-equity assets and liabilities are projected for an accurate Total Assets figure.
        # This part of the logic assumes that the DAG has correctly scheduled those items before this one.
        # We manually calculate total assets here to ensure we have the most up-to-date figure.
        asset_items = [
            acc for acc, info in self.account_map.items() 
            if info['statement'] == 'balance_sheet' and 'asset' in info['category']
        ]
        current_total_assets = current_period_data.loc[asset_items].sum()

        # Get the aggregate capital structure drivers from the schema.
        try:
            drivers = self.populated_schema['balance_sheet']['liabilities']['non_current_liabilities']['long_term_debt']['drivers']
            target_ratio_driver = drivers['target_debt_as_percent_of_assets']
            target_ratio = self._get_driver_value(target_ratio_driver, p_year_index)
        except (KeyError, TypeError):
            logger.warning(f"[{account}][{year_col}] Could not find long-term debt drivers. Holding debt constant.")
            return 0.0

        # Calculate the target total debt based on the current total assets.
        target_total_debt = current_total_assets * target_ratio
        
        # Get all long-term debt instruments to calculate the total opening balance.
        debt_items = self._get_schema_items('balance_sheet.liabilities.non_current_liabilities.long_term_debt')
        opening_total_debt = opening_balances.loc[debt_items].sum()

        # Calculate the total change required for the entire debt structure.
        total_required_change = target_total_debt - opening_total_debt

        # --- Step 2: Allocate the Change Proportionally (The New Logic) ---

        # Get the opening balance of the specific debt instrument we are currently processing.
        opening_balance_of_this_account = opening_balances.loc[account]

        # Calculate this instrument's proportion of the total opening debt.
        if opening_total_debt != 0:
            proportion = opening_balance_of_this_account / opening_total_debt
        else:
            # If there's no opening debt, assume the change is split equally among all instruments.
            # This handles the case of a company taking on debt for the first time.
            proportion = 1.0 / len(debt_items) if len(debt_items) > 0 else 0.0

        # The change for this specific account is its proportion of the total required change.
        change_for_this_account = total_required_change * proportion

        logger.info(f"[{account}][{year_col}] Debt Pro-Rata Δ: Proportion: {proportion:.2%}, Total Δ: {total_required_change:,.0f}, This Acct Δ: {change_for_this_account:,.0f}")
        
        return change_for_this_account

    def _calculate_equity_rollforward(self, account: str, year_col: str, p_year_index: int, opening_balances: pd.Series) -> float:
        """Calculates the CHANGE for an equity account for the period."""
        schema_entry = self.account_map[account]['schema_entry']
        category = self.account_map[account]['category']

        if category == 'common_stock':
            issuance_driver = schema_entry['drivers']['net_share_issuance']
            return self._get_driver_value(issuance_driver, p_year_index)
            
        elif category == 'retained_earnings':
            # The change in retained earnings is Net Income minus Dividends Paid.
            net_income = self.data_grid.loc['net_income', year_col]
            payout_driver = schema_entry['drivers']['dividend_payout_ratio']
            payout_ratio = self._get_driver_value(payout_driver, p_year_index)
            dividends_paid = net_income * payout_ratio
            
            # Ensure dividends are only paid out of profits, not losses.
            if dividends_paid < 0:
                dividends_paid = 0
                logger.warning(f"[retained_earnings][{year_col}] Net income is negative. Setting dividends to zero.")
            
            change = net_income - dividends_paid
            logger.info(f"[{account}][{year_col}] RE Δ: Net Income: {net_income:,.0f} - Dividends: {dividends_paid:,.0f} = {change:,.0f}")
            return change
            
        # For other equity items like 'contra_equity', hold constant by returning 0 change.
        return 0.0

    def _designate_primary_accounts(self) -> None:
        """
        Designates the primary operating cash account for the projection.
        This is a one-time setup step to ensure cash flow from financing activities
        is routed to a single, deterministic account.
        """
        logger.info("[Pre-Calc] Designating primary operating accounts...")
        
        cash_items = self._get_schema_items('balance_sheet.assets.current_assets.cash_and_equivalents')
        if not cash_items:
            # FAIL FAST & LOUD
            raise RuntimeError("FATAL: No cash accounts mapped in the schema. Cannot designate a primary cash account.")
            
        self.primary_cash_account = cash_items[0]
        logger.info(f"[Pre-Calc] Primary Operating Cash Account designated as: '{self.primary_cash_account}'")

    def _calculate_historical_working_capital_ratios(self) -> None:
        """
        Calculates historical blended DSO and DPO from historical data.
        This makes working capital projection deterministic and independent of LLM input.

        This method has side-effects: it populates self.deterministic_dso and
        self.deterministic_dpo.
        """
        logger.info("[Pre-Calc] Calculating deterministic working capital ratios from historicals...")

        if len(self.historical_years) < 2:
            logger.warning("[Pre-Calc] Working capital ratio calculation requires at least two historical years. Skipping.")
            self.deterministic_dso = 30.0 # Default fallback
            self.deterministic_dpo = 30.0 # Default fallback
            return

        # --- Days Sales Outstanding (DSO) ---
        ar_items = self._get_schema_items('balance_sheet.assets.current_assets.accounts_receivable')
        ar_series = self.data_grid.loc[ar_items, self.historical_years].sum()
        revenue_series = self.data_grid.loc['total_revenue', self.historical_years]
        
        # Use average of year-end balances for a more stable calculation
        avg_ar_balance = ar_series.mean()
        # Use total revenue over the period
        total_revenue = revenue_series.sum()
        num_days = len(self.historical_years) * 365

        if total_revenue > 0:
            self.deterministic_dso = (avg_ar_balance / total_revenue) * num_days if total_revenue > 0 else 0.0
            logger.info(f"[Pre-Calc] Calculated deterministic DSO: {self.deterministic_dso:.2f} days")
        else:
            self.deterministic_dso = 30.0 # Reasonable default
            logger.warning("[Pre-Calc] Total historical revenue is zero. Defaulting DSO to 30 days.")

        # --- Days Payables Outstanding (DPO) ---
        ap_items = self._get_schema_items('balance_sheet.liabilities.current_liabilities.accounts_payable')
        ap_series = self.data_grid.loc[ap_items, self.historical_years].sum()
        cogs_items = self._get_schema_items('income_statement.cost_of_revenue')
        cogs_series = self.data_grid.loc[cogs_items, self.historical_years].sum()
        
        avg_ap_balance = ap_series.mean()
        # Use absolute value as COGS is a negative number
        total_cogs = abs(cogs_series.sum())

        if total_cogs > 0:
            self.deterministic_dpo = (avg_ap_balance / total_cogs) * num_days if total_cogs > 0 else 0.0
            logger.info(f"[Pre-Calc] Calculated deterministic DPO: {self.deterministic_dpo:.2f} days")
        else:
            self.deterministic_dpo = 30.0 # Reasonable default
            logger.warning("[Pre-Calc] Total historical COGS is zero. Defaulting DPO to 30 days.")

    def _calculate_working_capital_change(self, account: str, category: str, year_col: str, current_period_data: pd.Series, opening_balances: pd.Series) -> float:
        """
        Calculates the required CHANGE in an Accounts Receivable or Accounts Payable
        item for the period to meet the target DSO or DPO.
        """
        # --- Dispatch based on category ---
        if category == 'accounts_receivable':
            # Target AR is based on Total Revenue
            projected_revenue = current_period_data.loc['total_revenue']
            
            # Note: This logic correctly calculates the target for the *entire* AR category.
            # We assume for now this change is applied to the first AR item if multiple exist.
            # A more complex model could allocate the change.
            ar_items = self._get_schema_items('balance_sheet.assets.current_assets.accounts_receivable')
            if not ar_items:
                logger.warning(f"No accounts receivable items found in schema for account: {account}")
                return 0.0  # Skip calculation if no AR items are defined

            if account != ar_items[0]:
                return 0.0 # Only calculate the change on the first mapped AR item.
            total_opening_ar = opening_balances.loc[ar_items].sum()

            target_ar_balance = (projected_revenue * self.deterministic_dso) / 365.0
            change_in_ar = target_ar_balance - total_opening_ar
            
            logger.info(f"[{account}][{year_col}] A/R Δ: Target DSO: {self.deterministic_dso:.2f}, Target AR: {target_ar_balance:,.0f}, Required Change: {change_in_ar:,.0f}")
            return change_in_ar

        elif category == 'accounts_payable':
            # Target AP is based on Total COGS
            cogs_items = self._get_schema_items('income_statement.cost_of_revenue')
            # Use absolute value as COGS is negative
            projected_cogs = abs(current_period_data.loc[cogs_items].sum())

            ap_items = self._get_schema_items('balance_sheet.liabilities.current_liabilities.accounts_payable')
            if not ap_items:
                logger.warning(f"No accounts payable items found in schema for account: {account}")
                return 0.0  # Skip calculation if no AP items are defined

            if account != ap_items[0]:
                return 0.0 # Only calculate the change on the first mapped AP item.
            
            total_opening_ap = opening_balances.loc[ap_items].sum()

            target_ap_balance = (projected_cogs * self.deterministic_dpo) / 365.0
            change_in_ap = target_ap_balance - total_opening_ap

            logger.info(f"[{account}][{year_col}] A/P Δ: Target DPO: {self.deterministic_dpo:.2f}, Target AP: {target_ap_balance:,.0f}, Required Change: {change_in_ap:,.0f}")
            return change_in_ap

        return 0.0 # Fallback

    def _calculate_delta(self, account: str, year_col: str, p_year_index: int, current_period_data: pd.Series, opening_balances: pd.Series) -> float:
        """
        Master dispatcher for all financial calculations, using a "Category-First" approach.

        This method determines the correct projection logic for a given account by first
        identifying its category within the financial statements, rather than relying on
        a specific account name. This makes the engine robust and adaptable.

        It returns the calculated CHANGE for a balance sheet item or the
        ABSOLUTE VALUE for an income statement item.
        """
        try:
            # Get the full context for any account defined in the schema
            account_info = self.account_map[account]
            statement = account_info['statement']
            category = account_info['category']
            schema_entry = account_info['schema_entry']
        except KeyError:
            # This block handles engine-internal accounts that are not in the account_map
            if account == 'total_revenue':
                rev_streams = self._get_schema_items('income_statement.revenue')
                return current_period_data.loc[rev_streams].sum()

            if account == 'interest_on_revolver':
                # This logic is self-contained and part of the circular group
                avg_revolver = (opening_balances.loc['revolver'] + current_period_data.loc['revolver']) / 2
                return avg_revolver * self.revolver_interest_rate * -1 # Interest expense is negative

            if account == 'interest_income_on_cash':
                # This logic is self-contained and part of the circular group
                try:
                    policy = self.populated_schema['balance_sheet']['assets']['current_assets']['cash_and_equivalents']['income_policy']
                    yield_driver = policy['drivers']['yield_on_cash_and_investments']
                    current_yield = self._get_driver_value(yield_driver, p_year_index)
                    
                    cash_items = self._get_schema_items('balance_sheet.assets.current_assets.cash_and_equivalents')
                    avg_cash = (opening_balances.loc[cash_items].sum() + current_period_data.loc[cash_items].sum()) / 2
                    return avg_cash * current_yield
                except (KeyError, TypeError):
                    return 0.0 # Return 0 if policy is not defined

            # Revolver and plug reversals are handled by other parts of the loop
            if account in ['revolver', '__historical_plug_reversal__']:
                return 0.0

        # --- HIERARCHICAL DISPATCHER ---

        # === LEVEL 1: Dispatch by Statement ===
        if statement == 'income_statement':
            # --- LEVEL 2: Dispatch by Category ---
            if category == 'revenue':
                model_name = schema_entry.get('projection_configuration', {}).get('selected_model', {}).get('model_name')
                if model_name == 'Asset_Yield_Driven':
                    return self._calculate_asset_yield_revenue(account, year_col, p_year_index, current_period_data, opening_balances)
                ### PLACEHOLDER ### - Logic for other revenue models like 'Unit_Economics' would go here.
                else: return 0.0

            elif category == 'cost_of_revenue':
                if self.is_lender:
                    return self._calculate_lender_cogs(account, year_col, p_year_index, current_period_data, opening_balances)
                ### PLACEHOLDER ### - Logic for non-lender COGS models would go here.
                else: return 0.0
            
            elif category == 'operating_expenses':
                # Check for a specific, known driver structure first
                if 'drivers' in schema_entry and 'sga_expense_as_percent_of_revenue' in schema_entry['drivers']:
                    driver = schema_entry['drivers']['sga_expense_as_percent_of_revenue']
                    rate = self._get_driver_value(driver, p_year_index)
                    return current_period_data.loc['total_revenue'] * rate * -1
                
                # Then check for special case, engine-calculated accounts
                elif account == 'depreciation_expense':
                    return self.aggregated_depreciation.get(year_col, 0.0) # This is a lookup
                elif account == 'amortization_expense':
                    return self.aggregated_amortization.get(year_col, 0.0) # This is a lookup
                
                ### PLACEHOLDER ### - Default behavior for any other OpEx is to hold constant.
                else: return 0.0

            elif category == 'industry_specific_operating_expenses':
                 if 'driver' in schema_entry and 'industry_specific_operating_expense_as_percent_of_revenue' in schema_entry['driver']:
                    driver = schema_entry['driver']['industry_specific_operating_expense_as_percent_of_revenue']
                    rate = self._get_driver_value(driver, p_year_index)
                    return current_period_data.loc['total_revenue'] * rate * -1
                 else: return 0.0

            elif category == 'income_tax_expense':
                ### PLACEHOLDER ### - This is a simplified tax calculation. A full implementation
                # would need to handle tax loss carryforwards and other complexities.
                tax_rate = 0.21 
                ebt_value = current_period_data.loc['ebt']
                # Only apply tax to profits. Expenses are negative.
                return min(0, ebt_value) * tax_rate

        elif statement == 'balance_sheet':
            if category == 'property_plant_equipment':
                return self._calculate_ppe_rollforward(account, year_col, p_year_index, opening_balances)

            elif category == 'intangible_assets':
                return self._calculate_intangible_rollforward(account, year_col, p_year_index, opening_balances)
            
            elif category == 'industry_specific_assets':
                return self._calculate_industry_specific_asset_growth(account, year_col, p_year_index, opening_balances)

            elif category == 'industry_specific_liabilities':
                # This logic handles any item within this category, leveraging the 'items' dict structure.
                try:
                    driver_obj = schema_entry['drivers']['industry_specific_liability_growth_rate']
                    growth_rate = self._get_driver_value(driver_obj, p_year_index)

                    opening_balance = opening_balances.loc[account]
                    change_in_liability = opening_balance * growth_rate

                    logger.info(f"[{account}][{year_col}] Industry Liability Growth Δ: Rate: {growth_rate:.4%}, Opening: {opening_balance:,.0f}, Change: {change_in_liability:,.0f}")
                    return change_in_liability
                except (KeyError, TypeError):
                    # If drivers are not defined for a specific item in this category, hold it constant.
                    logger.warning(f"[{account}][{year_col}] No valid 'industry_specific_liability_growth_rate' driver found. Holding constant.")
                    return 0.0

            elif category == 'long_term_debt':
                ### PLACEHOLDER ### - Using existing capital structure logic. This is complex
                # and would be a candidate for its own helper method in the future.
                return self._calculate_capital_structure_and_debt(account, year_col, p_year_index, current_period_data, opening_balances)
            
            elif category in ['accounts_receivable', 'accounts_payable']:
                return self._calculate_working_capital_change(account, category, year_col, current_period_data, opening_balances)

            elif category == 'common_stock':
                driver = schema_entry['drivers']['net_share_issuance']
                return self._get_driver_value(driver, p_year_index)

            elif category == 'retained_earnings':
                return self._calculate_equity_rollforward(account, year_col, p_year_index, opening_balances)

            else:
                ### PLACEHOLDER ### - The default behavior for all other unhandled BS accounts
                # (e.g., working capital, contra-assets) is to hold them constant.
                # A full implementation would have DSO/DPO/DIO logic here.
                return 0.0
        logger.warning(f"Account '{account}' in category '{category}' fell through all logic. Holding constant.")
        return 0.0

    def _articulate_final_statements(self, year_col: str) -> None:
        """
        Calculates all financial statement totals and subtotals for a given period.
        This version uses explicit, hardcoded paths to ensure robustness and
        avoids the complexity of a generic recursive helper.
        """
        logger.info(f"[{year_col}] Articulating final statement totals and subtotals...")
        
        current_period_series = self.data_grid[year_col]

        # --- Income Statement Articulation (This part was working correctly) ---
        cogs_items = self._get_schema_items('income_statement.cost_of_revenue')
        self.data_grid.loc['gross_profit', year_col] = current_period_series.loc['total_revenue'] + current_period_series.loc[cogs_items].sum()
        opex_items = self._get_schema_items('income_statement.operating_expenses')
        ind_opex_items = self._get_schema_items('income_statement.industry_specific_operating_expenses')
        self.data_grid.loc['operating_income', year_col] = current_period_series.loc['gross_profit'] + current_period_series.loc[opex_items + ind_opex_items].sum()
        non_op_items = self._get_schema_items('income_statement.non_operating_income_expense')
        interest_items = ['interest_on_revolver', 'interest_income_on_cash']
        if not self.is_lender: interest_items.append('interest_expense')
        self.data_grid.loc['ebt', year_col] = current_period_series.loc['operating_income'] + current_period_series.loc[non_op_items + interest_items].sum()
        self.data_grid.loc['net_income', year_col] = current_period_series.loc['ebt'] + current_period_series.loc['income_tax_expense']

        # --- Balance Sheet Articulation (Corrected) ---
        
        # Current Assets
        ca_items = self._get_schema_items('balance_sheet.assets.current_assets')
        self.data_grid.loc['total_current_assets', year_col] = current_period_series.loc[ca_items].sum()

        # Non-Current Assets
        nca_items = self._get_schema_items('balance_sheet.assets.non_current_assets')
        ind_assets = self._get_schema_items('balance_sheet.industry_specific_items.industry_specific_assets')
        self.data_grid.loc['total_non_current_assets', year_col] = current_period_series.loc[nca_items + ind_assets].sum()

        # Total Assets
        contra_assets = self._get_schema_items('balance_sheet.assets.contra_assets')
        asset_sum = current_period_series.loc['total_current_assets'] + current_period_series.loc['total_non_current_assets']
        contra_sum = current_period_series.loc[contra_assets].sum() if contra_assets else 0.0
        self.data_grid.loc['total_assets', year_col] = asset_sum + contra_sum

        # Current Liabilities
        cl_items = self._get_schema_items('balance_sheet.liabilities.current_liabilities')
        self.data_grid.loc['total_current_liabilities', year_col] = current_period_series.loc[cl_items].sum()

        # Non-Current Liabilities (THE FIX IS HERE)
        # We must explicitly get the items from the 'long_term_debt' sub-category.
        ncl_ltd_items = self._get_schema_items('balance_sheet.liabilities.non_current_liabilities.long_term_debt')
        ncl_other_items = self._get_schema_items('balance_sheet.liabilities.non_current_liabilities.other_non_current_liabilities')
        ind_liabilities = self._get_schema_items('balance_sheet.industry_specific_items.industry_specific_liabilities')
        all_ncl_items = ncl_ltd_items + ncl_other_items + ind_liabilities
        self.data_grid.loc['total_non_current_liabilities', year_col] = current_period_series.loc[all_ncl_items].sum()
        
        # Total Liabilities
        self.data_grid.loc['total_liabilities', year_col] = current_period_series.loc['total_current_liabilities'] + current_period_series.loc['total_non_current_liabilities']

        # Total Equity
        eq_items = self._get_schema_items('balance_sheet.equity')
        self.data_grid.loc['total_equity', year_col] = current_period_series.loc[eq_items].sum()
        
        # Total Liabilities & Equity
        self.data_grid.loc['total_liabilities_and_equity', year_col] = current_period_series.loc['total_liabilities'] + current_period_series.loc['total_equity']

    def _execute_projection_loop(self) -> None:
        """
        Main execution loop for the projections engine. (FINAL REVISED VERSION)
        This version contains the complete, robust circular solver logic with FIXED cash sweep.
        """
        logger.info("--- [START] Main Projection Loop (Full Model Execution) ---")
        
        pre_circular_plan = self.execution_plan['pre_circular']
        circular_plan = self.execution_plan['circular']
        
        bs_accounts = [acc for acc, info in self.account_map.items() if info['statement'] == 'balance_sheet']
        bs_accounts.extend(['revolver', '__historical_plug_reversal__'])
        bs_accounts = sorted(list(set(bs_accounts)))
        
        for p_index, year_col in enumerate(self.data_grid.columns.drop(self.historical_years), 1):
            logger.info(f"--- Projecting Year: {year_col} (Index: {p_index}) ---")
            prior_year_col = self._get_prior_period_col(year_col)

            # --- Phase A: Initial Roll-Forward ---
            self.data_grid.loc[bs_accounts, year_col] = self.data_grid.loc[bs_accounts, prior_year_col]
            if p_index > 1: self.data_grid.loc['__historical_plug_reversal__', year_col] = 0.0
            if p_index == 1: self._neutralize_historical_plugs(year_col)
            
            opening_balances = self.data_grid.loc[:, year_col].copy()

            # --- Phase B: Pre-Circular Calculation (Once) ---
            logger.info(f"[Phase B] Executing {len(pre_circular_plan)} pre-circular calculations...")
            self.aggregated_depreciation[year_col], self.aggregated_amortization[year_col] = 0.0, 0.0
            for account in pre_circular_plan:
                # We pass the live data grid for the second argument, as pre-circular items may depend on each other
                value = self._calculate_delta(account, year_col, p_index, self.data_grid[year_col], opening_balances)
                if self.account_map.get(account, {}).get('statement') == 'balance_sheet': 
                    self.data_grid.loc[account, year_col] += value
                else: 
                    self.data_grid.loc[account, year_col] = value
                            
            # --- Phase C: Circular Solver (Iterative) with Diagnostics ---
            logger.info(f"[Phase C] Activating circular solver for {len(circular_plan)} items...")

            # Add diagnostic tracking
            balance_sheet_check_frequency = 10  # Check balance every 10 iterations
            detailed_diagnostics = True

            for i in range(self.MAX_ITERATIONS_CIRCULAR):
                old_revolver_balance = self.data_grid.loc['revolver', year_col]
                
                # Store state before circular recalculation for diagnostics
                if detailed_diagnostics and i % balance_sheet_check_frequency == 0:
                    pre_circular_assets = self.data_grid.loc[bs_accounts, year_col].sum()
                    pre_circular_liab_equity = self.data_grid.loc[bs_accounts, year_col].sum()
                    logger.info(f"[DIAGNOSTIC][{year_col}][Iteration {i+1}] BEFORE circular recalc - Assets: {pre_circular_assets:,.2f}, Liab+Equity: {pre_circular_liab_equity:,.2f}, Diff: {abs(pre_circular_assets - pre_circular_liab_equity):,.2f}")
                
                # Recalculate all items in the circular group using the LIVE data grid state
                for account in circular_plan:
                    if account == 'revolver': continue
                    
                    old_value = self.data_grid.loc[account, year_col]
                    value = self._calculate_delta(account, year_col, p_index, self.data_grid[year_col], opening_balances)
                    
                    if self.account_map.get(account, {}).get('statement') == 'balance_sheet':
                        new_value = opening_balances.loc[account] + value
                        self.data_grid.loc[account, year_col] = new_value
                    else:
                        new_value = value
                        self.data_grid.loc[account, year_col] = new_value
                    
                    # Log significant changes
                    if abs(new_value - old_value) > 1000000:  # Log changes > 1M
                        logger.debug(f"[DIAGNOSTIC][{year_col}][Iteration {i+1}] {account}: {old_value:,.2f} -> {new_value:,.2f} (Δ: {new_value - old_value:,.2f})")

                # Check balance after circular recalculation
                if detailed_diagnostics and i % balance_sheet_check_frequency == 0:
                    post_circular_assets = self.data_grid.loc[bs_accounts, year_col].sum()
                    post_circular_liab_equity = self.data_grid.loc[bs_accounts, year_col].sum()
                    logger.info(f"[DIAGNOSTIC][{year_col}][Iteration {i+1}] AFTER circular recalc - Assets: {post_circular_assets:,.2f}, Liab+Equity: {post_circular_liab_equity:,.2f}, Diff: {abs(post_circular_assets - post_circular_liab_equity):,.2f}")

                # --- FIXED Cash Sweep Calculation ---
                logger.info(f"[CASH SWEEP][{year_col}][Iteration {i+1}] Starting cash sweep calculation...")
                
                # Get cash policy parameters
                cash_policy_driver = self.populated_schema['balance_sheet']['assets']['current_assets']['cash_and_equivalents']['cash_policy']['drivers']['min_cash_as_percent_of_revenue']
                min_cash_ratio = self._get_driver_value(cash_policy_driver, p_index)
                target_min_cash = self.data_grid.loc['total_revenue', year_col] * min_cash_ratio
                
                if i % balance_sheet_check_frequency == 0:  # Reduce log spam
                    logger.info(f"[CASH SWEEP][{year_col}][Iteration {i+1}] Target min cash: {target_min_cash:,.2f} ({min_cash_ratio*100:.1f}% of revenue: {self.data_grid.loc['total_revenue', year_col]:,.2f})")

                # Separate balance sheet components properly using account_map
                asset_accounts = [a for a in bs_accounts if a != self.primary_cash_account and a != 'revolver' 
                                and self.account_map.get(a, {}).get('category') == 'assets']
                liability_accounts = [l for l in bs_accounts if l != 'revolver' 
                                    and self.account_map.get(l, {}).get('category') == 'liabilities']
                equity_accounts = [e for e in bs_accounts if self.account_map.get(e, {}).get('category') == 'equity']
                
                total_non_cash_assets = self.data_grid.loc[asset_accounts, year_col].sum() if asset_accounts else 0.0
                total_liabilities_no_revolver = self.data_grid.loc[liability_accounts, year_col].sum() if liability_accounts else 0.0
                total_equity = self.data_grid.loc[equity_accounts, year_col].sum() if equity_accounts else 0.0

                # Basic accounting equation: Assets = Liabilities + Equity
                # Cash = (Total Liabilities + Revolver + Total Equity) - (Non-cash Assets)
                implied_cash_before_sweep = (total_liabilities_no_revolver + self.data_grid.loc['revolver', year_col] + total_equity) - total_non_cash_assets

                # Calculate surplus/shortfall relative to minimum cash requirement
                cash_surplus_or_shortfall = implied_cash_before_sweep - target_min_cash

                if i % balance_sheet_check_frequency == 0:  # Reduce log spam
                    logger.info(f"[CASH SWEEP][{year_col}][Iteration {i+1}] Cash before sweep: {implied_cash_before_sweep:,.2f}, Surplus/Shortfall: {cash_surplus_or_shortfall:,.2f}")
                    logger.debug(f"[CASH SWEEP][{year_col}][Iteration {i+1}] Components - Non-cash assets: {total_non_cash_assets:,.2f}, Liabilities: {total_liabilities_no_revolver:,.2f}, Equity: {total_equity:,.2f}, Revolver: {self.data_grid.loc['revolver', year_col]:,.2f}")

                if cash_surplus_or_shortfall > 0:  # We have excess cash
                    # Calculate how much we can pay down the revolver
                    current_revolver_balance = self.data_grid.loc['revolver', year_col]
                    max_possible_paydown = min(cash_surplus_or_shortfall, current_revolver_balance)
                    actual_paydown = max_possible_paydown * self.CASH_SWEEP_PERCENTAGE
                    
                    if i % balance_sheet_check_frequency == 0:  # Reduce log spam
                        logger.info(f"[CASH SWEEP][{year_col}][Iteration {i+1}] CASH SURPLUS: {cash_surplus_or_shortfall:,.2f}. Paying down revolver by {actual_paydown:,.2f}.")
                    
                    # Apply the paydown
                    self.data_grid.loc['revolver', year_col] -= actual_paydown
                    # Cash will be the excess after paydown
                    final_cash_balance = target_min_cash + (cash_surplus_or_shortfall - actual_paydown)
                    
                else:  # We have a cash shortfall
                    cash_shortfall = abs(cash_surplus_or_shortfall)
                    
                    if i % balance_sheet_check_frequency == 0:  # Reduce log spam
                        logger.info(f"[CASH SWEEP][{year_col}][Iteration {i+1}] CASH SHORTFALL: {cash_shortfall:,.2f}. Drawing on revolver by {cash_shortfall:,.2f}.")
                    
                    # Draw on revolver to meet minimum cash requirement
                    self.data_grid.loc['revolver', year_col] += cash_shortfall
                    # Cash will be set to minimum target
                    final_cash_balance = target_min_cash

                # Set the final cash balance
                self.data_grid.loc[self.primary_cash_account, year_col] = final_cash_balance

                self._articulate_final_statements(year_col)

                # Verify balance sheet balances
                total_assets_final = sum(self.data_grid.loc[a, year_col] for a in asset_accounts) + final_cash_balance
                total_liab_equity_final = total_liabilities_no_revolver + self.data_grid.loc['revolver', year_col] + total_equity
                balance_diff = abs(total_assets_final - total_liab_equity_final)

                if i % balance_sheet_check_frequency == 0:  # Reduce log spam
                    logger.info(f"[CASH SWEEP][{year_col}][Iteration {i+1}] After sweep - Revolver: {self.data_grid.loc['revolver', year_col]:,.2f}, {self.primary_cash_account}: {final_cash_balance:,.2f}")
                    logger.debug(f"[CASH SWEEP][{year_col}][Iteration {i+1}] Balance check - Assets: {total_assets_final:,.2f}, Liab+Equity: {total_liab_equity_final:,.2f}, Difference: {balance_diff:,.2f}")

                if balance_diff > 0.01:  # Allow for small floating point differences
                    if i % balance_sheet_check_frequency == 0:  # Reduce log spam
                        logger.warning(f"[CASH SWEEP][{year_col}][Iteration {i+1}] WARNING: Balance sheet does not balance! Difference: {balance_diff:,.2f}")
                    
                    # EMERGENCY DIAGNOSTIC: If we're consistently failing to balance, dump more details
                    if i > 20 and balance_diff > 1000000000:  # After 20 iterations with >1B difference
                        logger.error(f"[EMERGENCY DIAGNOSTIC][{year_col}][Iteration {i+1}] PERSISTENT IMBALANCE:")
                        logger.error(f"  Non-cash Assets: {total_non_cash_assets:,.2f}")
                        logger.error(f"  Cash: {final_cash_balance:,.2f}")
                        logger.error(f"  Total Assets: {total_assets_final:,.2f}")
                        logger.error(f"  Liab (no revolver): {total_liabilities_no_revolver:,.2f}")
                        logger.error(f"  Revolver: {self.data_grid.loc['revolver', year_col]:,.2f}")
                        logger.error(f"  Equity: {total_equity:,.2f}")
                        logger.error(f"  Total Liab+Equity: {total_liab_equity_final:,.2f}")
                        logger.error(f"  Asset accounts: {asset_accounts}")
                        logger.error(f"  Liability accounts: {liability_accounts}")
                        logger.error(f"  Equity accounts: {equity_accounts}")
                        logger.error(f"  Circular accounts in this iteration: {circular_plan}")
                        
                        # Break out of the loop to prevent infinite running
                        logger.error(f"[EMERGENCY DIAGNOSTIC] Breaking out of circular solver due to persistent imbalance")
                        break
                        
                else:
                    if i % balance_sheet_check_frequency == 0:  # Reduce log spam
                        logger.info(f"[CASH SWEEP][{year_col}][Iteration {i+1}] Balance sheet balances correctly.")

                # Check convergence
                if abs(self.data_grid.loc['revolver', year_col] - old_revolver_balance) < self.CONVERGENCE_TOLERANCE:
                    logger.info(f"Circular solver converged in {i+1} iterations.")
                    break
            else:
                logger.warning(f"Circular solver did NOT converge after {self.MAX_ITERATIONS_CIRCULAR} iterations.")
            
            logger.info(f"--- Projection for year {year_col} complete. ---")

        logger.info("--- [END] Main Projection Loop Finished ---")

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

    def project(self) -> pd.DataFrame:
        """
        The main public method to orchestrate the entire projection process.
        """
        logger.info(f"--- Starting Projection for {self.security_id} ---")
        
        # --- Phase I: Setup & Compilation ---
        logger.info("[1/5] Loading and filtering inputs...")
        self._load_and_filter_historicals()
        
        logger.info("[2/5] Initializing data grid...")
        self._create_data_grid()
                
        logger.info("[3/5] Sanitizing data...")
        self._sanitize_data_grid()

        logger.info("[4/5] Building and compiling execution graph...")
        self._build_and_compile_graph()
        
        logger.info("[5.5] Pre-calculating projection constants and baselines...")
        self._pre_calculate_constants()

        # --- Phase II: Projection Loop ---
        logger.info("--- Executing Projection Loop ---")
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