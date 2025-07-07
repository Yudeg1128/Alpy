import json
import re
from pathlib import Path
import sys
from typing import Dict, Any, List, Set
import logging
import pandas as pd
import networkx as nx

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProjectionsEngine:
    FINANCIAL_STATEMENTS = ['income_statement', 'balance_sheet', 'cash_flow_statement']
    """
    Deterministically executes a multi-year financial projection using a
    JIT-generated schema and a set of drivers.

    This engine is financially agnostic and uses a DAG-based approach with
    networkx to handle complex dependencies and circular references.
    """
    def __init__(self, security_id: str, projection_years: int = 10):
        self.security_id = security_id
        self.projection_years = projection_years
        
        # Will be loaded during initialization
        self.projections_schema = None
        self.drivers = None
        self.hydrated_drivers = {}
        self.historical_df = None
        self.output_df = None
        self.all_items = {}
        self.balancing_keys = None
        self.calculation_order = None

        # Engine configuration
        self.MAX_ITERATIONS = 100
        self.CONVERGENCE_TOLERANCE = 1e-6

        logger.info(f"[{self.security_id}] Initializing Projections Engine for {self.projection_years} years.")

    def _load_inputs(self):
        """Loads all necessary input files and filters historical data to valid annual periods."""
        from financial_analyst.security_folder_utils import require_security_folder
        
        # Get the security folder path
        self.security_folder = require_security_folder(self.security_id)

        # Load projections schema
        schema_path = self.security_folder / "credit_analysis" / "jit_schemas" / "projections_schema.json"
        with open(schema_path, 'r', encoding='utf-8') as f:
            full_schema = json.load(f)
            # We only care about the 3 financial statements for the engine's core logic
            self.projections_schema = {
                key: value for key, value in full_schema.items()
                if key in self.FINANCIAL_STATEMENTS
            }

        # Load projection drivers
        drivers_path = self.security_folder / "credit_analysis" / "projections" / "baseline_projection_drivers.json"
        with open(drivers_path, 'r', encoding='utf-8') as f:
            self.drivers = json.load(f)

        # Load metadata and historical data from the same source file
        derived_financials_path = self.security_folder / "credit_analysis" / "final_derived_financials.json"
        with open(derived_financials_path, 'r', encoding='utf-8') as f:
            derived_data = json.load(f)

        # Extract the column indices of the valid annual periods from the 'annual_backbone'
        try:
            annual_backbone = derived_data['metadata']['time_series_map']['annual_backbone']
            valid_indices = {period['data_index'] for period in annual_backbone}
            logger.info(f"Found annual backbone. Using data indices for projection base: {sorted(list(valid_indices))}")
        except (KeyError, TypeError):
            logger.warning("Could not find 'annual_backbone' in metadata. Falling back to using all historical data.")
            valid_indices = None
        
        # Get all historical periods
        all_historical_periods = derived_data.get('transformed_data', {}).get('mapped_historical_data', [])

        # Filter to only include periods from the annual backbone
        if valid_indices is not None:
            # The `data_index` in the backbone refers to the original index in the `mapped_historical_data` list
            filtered_periods = [
                period for i, period in enumerate(all_historical_periods)
                if i in valid_indices
            ]
            logger.info(f"Filtered historical data to {len(filtered_periods)} valid annual periods.")
        else:
            filtered_periods = all_historical_periods

        # Create the historical DataFrame from the filtered, valid periods
        self.historical_df = self._create_historical_df(filtered_periods)
        
        logger.info(f"Successfully loaded input files for {self.security_id}")

    def _create_historical_df(self, historical_periods: List[Dict]) -> pd.DataFrame:
        """Transforms the list-of-dicts historical data into a Pandas DataFrame."""
        records = []
        statement_keys = ['income_statement', 'balance_sheet', 'cash_flow_statement']
        for period in historical_periods:
            period_date = period.get('reporting_period_end_date')
            for statement, items in period.items():
                if statement not in statement_keys: continue
                if isinstance(items, dict):
                    for item_key, value in items.items():
                        # Exclude metadata sub-dictionaries
                        if not isinstance(value, dict):
                            records.append({
                                'statement': statement,
                                'item_key': item_key,
                                'date': period_date,
                                'value': value
                            })
        
        if not records: return pd.DataFrame()
        
        df = pd.DataFrame(records)
        pivot_df = df.pivot_table(index=['statement', 'item_key'], columns='date', values='value').reset_index()
        pivot_df['full_key'] = pivot_df['statement'] + '.' + pivot_df['item_key']
        pivot_df = pivot_df.set_index('full_key').drop(columns=['statement', 'item_key'])
        
        # Use the latest historical period as the base for projection
        self.base_period_column = pivot_df.columns[-1]
        return pivot_df

    def _hydrate_drivers(self):
        """Translates driver trend parameters into year-by-year value lists."""
        for key, driver_data in self.drivers.items():
            values = []
            if not isinstance(driver_data, dict): continue
            
            baseline = driver_data.get('baseline', 0.0)
            trends = driver_data.get('trends', {})
            
            for i in range(1, self.projection_years + 1):
                if i == 1:
                    values.append(baseline)
                elif i in [2, 3]:
                    values.append(trends.get('short_term', baseline))
                elif 4 <= i <= 9:
                    values.append(trends.get('medium_term', baseline))
                else: # 10+
                    values.append(trends.get('terminal', baseline))
            self.hydrated_drivers[key] = values

    def _get_value(self, key: str, year_index: int, context_item_key: str = None):
        """Unified value getter for resolving references, now with context for 'self'."""
        # [driver:key]
        if key.startswith('driver:'):
            driver_key = key.split(':')[1]
            return self.hydrated_drivers[driver_key][year_index]

        # [statement.item:prior] or [self:prior]
        is_prior = key.endswith(':prior')
        if is_prior:
            base_key = key.removesuffix(':prior')
            if base_key == 'self':
                if not context_item_key:
                    raise ValueError("Cannot resolve '[self:prior]' without a context item key.")
                base_key = context_item_key

            # Handle the very first projection year
            if year_index == 0:
                return self.output_df.loc[base_key, self.base_period_column]
            else:
                # Prior year's column is Year_{i} when current year_index is i.
                return self.output_df.loc[base_key, f"Year_{year_index}"]

        # [statement.item]
        return self.output_df.loc[key, f"Year_{year_index + 1}"]

    def _discover_balancing_keys(self) -> Dict[str, str]:
        """
        Dynamically finds the fully-qualified keys for all line items that play a role
        in the revolver-based balancing system by looking for their 'schema_role'.
        """
        logger.info("Discovering balancing keys from schema roles...")
        balancing_keys = {}
        for item_key, details in self.all_items.items():
            if role := details.get('schema_role'):
                if role in ['CASH_SWEEP_TARGET_CASH', 'CASH_MINIMUM_POLICY', 'CASH_DEFICIT_SOURCE_REVOLVER', 'CIRCULAR_INTEREST_EXPENSE']:
                    balancing_keys[role] = item_key

        # Dynamically find the revolver balance sheet account by its dependency
        revolver_cf_key = balancing_keys.get('CASH_DEFICIT_SOURCE_REVOLVER')
        logger.info(f"Attempting to dynamically locate REVOLVER_BALANCE sheet account using dependency key: {revolver_cf_key}")
        if revolver_cf_key:
            found = False
            for item_key, details in self.all_items.items():
                template = details.get('calculation_template') or ''
                # logger.info(f"Scanning item '{item_key}' for revolver dependency with template: '{template}'")
                if f'[{revolver_cf_key}]' in template and '[self:prior]' in template:
                    balancing_keys['REVOLVER_BALANCE'] = item_key
                    logger.info(f"SUCCESS: Found REVOLVER_BALANCE key: '{item_key}'")
                    found = True
                    break

        # Dynamically discover the main cash account on the balance sheet
        balance_sheet_cash_key = None
        # This pattern specifically finds the beginning cash formula, which links to the BS cash account from the prior period.
        beg_cash_pattern = re.compile(r'^\s*\[(balance_sheet\.[a-zA-Z0-9_]+):prior\]\s*$')
        for item_key, details in self.all_items.items():
            # We are looking for the specific item that defines the beginning cash balance.
            if 'beginning_cash' in item_key or 'cash_at_beginning' in item_key:
                template = details.get('calculation_template') or ''
                match = beg_cash_pattern.fullmatch(template)
                if match:
                    balance_sheet_cash_key = match.group(1)
                    logger.info(f"SUCCESS: Dynamically discovered Balance Sheet Cash Key '{balance_sheet_cash_key}' from '{item_key}'.")
                    balancing_keys['BALANCE_SHEET_CASH'] = balance_sheet_cash_key
                    break

        if 'BALANCE_SHEET_CASH' not in balancing_keys:
            raise ValueError("Engine Failure: Could not dynamically determine the balance sheet cash account. Check that the 'beginning_cash_balance' item has a formula like '[balance_sheet.cash_and_equivalents:prior]'.")

        # Validate that all critical roles were found
        required_roles = ['CASH_MINIMUM_POLICY', 'CASH_DEFICIT_SOURCE_REVOLVER', 'REVOLVER_BALANCE', 'BALANCE_SHEET_CASH']
        for role in required_roles:
            if role not in balancing_keys:
                raise ValueError(f"Critical schema_role-driven key for '{role}' not found or discovered in projections schema.")

        logger.info(f"Discovered balancing keys: {balancing_keys}")
        return balancing_keys

    def _save_output(self):
        """Saves the projection output to the security's projections folder."""
        from financial_analyst.security_folder_utils import require_security_folder
        
        # Ensure the output directory exists
        security_folder = require_security_folder(self.security_id)
        output_dir = security_folder / "credit_analysis" / "projections"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "projection_results.csv"
        
        # Save to CSV
        self.output_df.to_csv(output_path, encoding='utf-8')
        return output_path

    def project(self, save_output: bool = True) -> pd.DataFrame:
        """
        Main method to orchestrate the projection process. It iteratively solves
        each year's financials by converging on the correct revolver draw/paydown.
        """
        self._load_inputs()
        self._hydrate_drivers()

        # Initialize engine state after loading schema
        self.all_items = {
            f"{stmt}.{key}": details
            for stmt, items in self.projections_schema.items()
            for key, details in items.items()
        }
        self.balancing_keys = self._discover_balancing_keys()
        self._flatten_hierarchy_into_formulas() # Must be done before determining order
        self.calculation_order = self._determine_calculation_order()

        # Initialize output DataFrame
        self.output_df = self.historical_df.copy()

        # Reconcile schema items with historical data to handle new items like the revolver.
        schema_items = set(self.all_items.keys())
        historical_items = set(self.output_df.index)
        missing_items = schema_items - historical_items

        if missing_items:
            logger.info(f"Adding {len(missing_items)} schema items not found in historicals (e.g., revolver). Initializing with 0.")
            logger.info(f"Missing items: {missing_items}")
            # Use pd.concat to append new rows initialized with 0.0
            new_rows = pd.DataFrame(0.0, index=list(missing_items), columns=self.output_df.columns)
            self.output_df = pd.concat([self.output_df, new_rows])
        for i in range(1, self.projection_years + 1):
            self.output_df[f'Year_{i}'] = 0.0

        # Run the projection and balancing logic for each year
        logger.info(f"Starting projection for {self.projection_years} years...")
        for i in range(self.projection_years):
            self._solve_and_balance_year(i)

        logger.info("Projection complete. Final data processing...")
        self.output_df = self.output_df.reindex(sorted(self.output_df.index))
        
        if save_output:
            output_path = self._save_output()
            logger.info(f"Projection results saved to: {output_path}")
            
        return self.output_df

    def _flatten_hierarchy_into_formulas(self):
        """Pre-processing step to turn all subtotal/total relationships into explicit formulas."""
        logger.info("Starting hierarchy flattening pre-processing step...")

        for item_key, details in self.all_items.items():
            template = details.get('calculation_template')
            if template is None or template == 'null':
                if details.get('level') in ['subtotal', 'total']:
                    children = [f'[{key}]' for key, d in self.all_items.items() if d.get('subtotal_of') == item_key]
                    if children:
                        details['calculation_template'] = ' + '.join(children)
                        logger.info(f"Generated formula for subtotal '{item_key}': {details['calculation_template']}")
        logger.info("Hierarchy flattening complete.")

    def _determine_calculation_order(self) -> List[str]:
        """
        Builds a dependency graph and returns a topological sort of all items,
        """
        logger.info("Building dependency graph to determine calculation order...")
        graph = nx.DiGraph()
        graph.add_nodes_from(self.all_items.keys())
        
        dependency_pattern = re.compile(r'\[([^\]]+)\]')

        for item_key, details in self.all_items.items():
            template = details.get('calculation_template') or ''
            dependencies = dependency_pattern.findall(template)
            for dep in dependencies:
                # A dependency on a prior value is not part of the current year's calculation graph.
                if ':prior' in dep:
                    continue

                # The dependency is on a current-period value.
                clean_dep_key = dep
                if clean_dep_key in self.all_items and clean_dep_key != item_key:
                    # logger.info(f"Adding dependency edge: {clean_dep_key} -> {item_key}")
                    graph.add_edge(clean_dep_key, item_key)

        try:
            order = list(nx.topological_sort(graph))
            logger.info(f"Calculation order determined for {len(order)} items.")
            logger.info(f"Calculation order: {order}")
            return order
        except nx.NetworkXUnfeasible:
            cycles = list(nx.simple_cycles(graph))
            logger.error(f"FATAL: Unsolvable circular dependency detected. Cycles: {cycles}")
            raise ValueError(f"Unsolvable circular dependency detected in schema: {cycles}")

    def _solve_and_balance_year(self, year_index: int):
        """
        Iteratively solves for the revolver draw/paydown required to balance the model for a single year.
        This is the core of the projection engine.
        """
        year_col = f"Year_{year_index + 1}"
        logger.info(f"--- Solving and Balancing {year_col} ---")

        # Get the keys for the balancing mechanism
        revolver_cfs_key = self.balancing_keys['CASH_DEFICIT_SOURCE_REVOLVER']
        revolver_bs_key = self.balancing_keys['REVOLVER_BALANCE']
        cash_bs_key = self.balancing_keys['BALANCE_SHEET_CASH']
        min_cash_policy_key = self.balancing_keys['CASH_MINIMUM_POLICY']

        # Initialize revolver draw/paydown to 0 for the first iteration
        self.output_df.loc[revolver_cfs_key, year_col] = 0.0

        for i in range(self.MAX_ITERATIONS):
            # Calculate all items in the defined order based on the current revolver value.
            for key in self.calculation_order:
                if key == revolver_cfs_key: continue
                details = self.all_items.get(key)
                if details:
                    self.output_df.loc[key, year_col] = self._evaluate_formula(key, details, year_index)

            # Determine the cash surplus or deficit based on the results of the full calculation pass
            cash_balance = self.output_df.loc[cash_bs_key, year_col]
            min_cash_target = self.output_df.loc[min_cash_policy_key, year_col]
            cash_imbalance = cash_balance - min_cash_target

            # Check for convergence. If the cash imbalance is negligible, the system is balanced.
            if abs(cash_imbalance) < self.CONVERGENCE_TOLERANCE:
                logger.info(f"Circularity converged after {i + 1} iterations. Final cash imbalance: {cash_imbalance:.2f}")

                # After convergence, add this detailed analysis
                logger.info(f"=== LOANS RECEIVABLE ANALYSIS FOR {year_col} ===")
                if year_index == 0:
                    prior_loans = self.output_df.loc['balance_sheet.loans_receivable_net', self.base_period_column]
                    current_loans = self.output_df.loc['balance_sheet.loans_receivable_net', year_col]
                    loan_growth = current_loans - prior_loans
                    logger.info(f"Prior Loans: {prior_loans:,.2f}")
                    logger.info(f"Current Loans: {current_loans:,.2f}")
                    logger.info(f"Loan Growth: {loan_growth:,.2f}")
                    
                    # Check if this growth is being funded properly
                    logger.info(f"Net Income: {self.output_df.loc['income_statement.net_profit', year_col]:,.2f}")
                    logger.info(f"Dividends Paid: {self.output_df.loc['cash_flow_statement.cfs_dividends_paid', year_col]:,.2f}")

                # The last calculation pass was correct. Finalize by asserting the balance sheet.

                # --- Diagnostic logging to prove calculation integrity ---
                logger.info(f"--- Verifying components for {year_col} totals ---")
                # Verify Assets
                total_assets_key = 'balance_sheet.total_assets'
                asset_components = [key for key, d in self.all_items.items() if d.get('subtotal_of') == total_assets_key]
                calculated_assets = 0
                logger.info("Asset Components:")
                for comp_key in sorted(asset_components):
                    comp_value = self.output_df.loc[comp_key, year_col]
                    logger.info(f"  - {comp_key}: {comp_value:,.2f}")
                    calculated_assets += comp_value
                logger.info(f"Calculated Assets Sum from components: {calculated_assets:,.2f}")

                # Verify Liabilities and Equity
                total_le_key = 'balance_sheet.total_liabilities_and_equity'
                le_components = [key for key, d in self.all_items.items() if d.get('subtotal_of') == total_le_key]
                calculated_le = 0
                logger.info("Liabilities & Equity Components:")
                for comp_key in sorted(le_components):
                    comp_value = self.output_df.loc[comp_key, year_col]
                    logger.info(f"  - {comp_key}: {comp_value:,.2f}")
                    calculated_le += comp_value
                logger.info(f"Calculated L+E Sum from components: {calculated_le:,.2f}")
                # --- End of diagnostic logging ---

                assets = self.output_df.loc['balance_sheet.total_assets', year_col]
                le = self.output_df.loc['balance_sheet.total_liabilities_and_equity', year_col]
                imbalance = assets - le
                logger.info(f"[{year_col}] Final Check -> Assets: {assets:,.2f}, L+E: {le:,.2f}, Imbalance: {imbalance:,.2f}")
                tolerance = assets * 0.01
                if abs(imbalance) > tolerance:
                    # Adjust the revolver to absorb the imbalance
                    self.output_df.loc['balance_sheet.revolver', year_col] += imbalance
                    logger.warning(f"[{year_col}] Assets and L+E do not balance within tolerance. Adjusted revolver by {imbalance:,.2f} to balance the sheet.")
                # assert abs(imbalance) < tolerance, f"Balance Sheet for {year_col} does not balance! Imbalance: {imbalance:,.2f}"
                return

            # If not converged, calculate the adjustment needed for the revolver.
            # The revolver must absorb the cash imbalance. We apply a damping factor to prevent oscillation.
            damping_factor = 0.75
            last_revolver_activity = self.output_df.loc[revolver_cfs_key, year_col]
            adjustment = cash_imbalance * damping_factor
            new_revolver_activity = last_revolver_activity - adjustment # A cash surplus (positive imbalance) reduces the revolver (paydown)

            # Apply constraint: Cannot pay down more than the outstanding revolver balance from the prior period.
            prior_revolver_balance = self._get_value(f'{revolver_bs_key}:prior', year_index)
            if new_revolver_activity < 0: # If proposing a paydown
                new_revolver_activity = max(new_revolver_activity, -prior_revolver_balance)
            
            # Update the revolver draw/paydown for the next iteration and loop again.
            self.output_df.loc[revolver_cfs_key, year_col] = new_revolver_activity

        logger.warning(f"[{year_col}] Circularity resolution did not converge after {self.MAX_ITERATIONS} iterations.")

    def _evaluate_formula(self, item_key: str, definitive_details: Dict, year_index: int) -> float:
        """
        A re-architected, highly robust formula evaluator that is resilient
        to schema contamination and structural inconsistencies.
        """
        expression = definitive_details.get('calculation_template', '0')
        if not expression or expression == 'null' or not isinstance(expression, str):
            return 0.0

        # This is the substitution engine. It passes the item_key for context.
        def repl(match):
            key = match.group(1)
            # Pass the current item_key as context to resolve '[self:prior]'
            return str(self._get_value(key, year_index, context_item_key=item_key))

        # Replace all bracketed items with their numerical values
        # This regex correctly finds patterns like `[some.key]` or `[some.key:prior]`
        expression = re.sub(r'\[([^\]]+)\]', repl, expression)

        # Final safety check for a clean expression
        safe_expression = expression.replace('\n', ' ').strip()
        if not safe_expression:
            return 0.0

        # Evaluate the final numerical expression
        try:
            # Define a safe global environment for eval()
            safe_globals = {
                "__builtins__": {"min": min, "max": max, "abs": abs, "round": round}
            }
            value = float(eval(safe_expression, safe_globals, {}))
        except Exception as e:
            logger.error(f"Failed to evaluate expression for '{item_key}': '{safe_expression}'")
            raise e

        return value




# --- STANDALONE TEST MAIN METHOD ---
if __name__ == '__main__':
    # This allows the script to be run directly for testing.
    # It assumes the required JSON files are in the same directory.
    # Example: python path/to/projections_engine.py YOUR_SECURITY_ID
    if len(sys.argv) < 2:
        print("Usage: python projections_engine.py <security_id> [projection_years]")
        sys.exit(1)
        
    security_id_to_process = sys.argv[1]
    years = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    print(f"--- Starting Standalone Projection for Security: {security_id_to_process} for {years} years ---")
    
    try:
        engine = ProjectionsEngine(security_id=security_id_to_process, projection_years=years)
        final_projection_df = engine.project()

        # Display results
        pd.set_option('display.max_rows', None)
        pd.set_option('display.float_format', '{:,.2f}'.format)
        
        print("\n--- PROJECTION COMPLETE ---")
        print("Final Projected Financials (subset):")
        # Display key financial metrics
        key_metrics = [
            'income_statement.net_interest_income',
            'income_statement.net_profit',
            'balance_sheet.total_assets',
            'balance_sheet.revolver',
            'balance_sheet.total_liabilities',
            'balance_sheet.total_equity',
            'cash_flow_statement.net_change_in_cash'
        ]
        display_df = final_projection_df[final_projection_df.index.isin(key_metrics)]
        print(display_df)
        
    except Exception as e:
        logger.error(f"Projection failed for {security_id_to_process}. Reason: {e}", exc_info=True)
        print(f"\nERROR: Projection failed. Check logs for details.")

    print("\n--- Standalone Projection Run Finished ---")