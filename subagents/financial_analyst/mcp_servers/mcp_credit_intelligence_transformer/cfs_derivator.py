import json
import re
from copy import deepcopy
from pathlib import Path
import sys
from typing import Optional, Dict, Any, List, Tuple
import logging

# Third-party import for robust graph operations
import networkx as nx

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('cfs_derivation.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

class CFSDerivator:
    """
    Derives a Cash Flow Statement by deterministically executing instructions
    from a JIT-generated derivation schema.

    This engine is financially agnostic and leverages the 'networkx' library
    to robustly manage calculation dependencies, ensuring correctness and
    preventing logical errors like circular references.
    """

    def __init__(self, security_id: str):
        """Initializes the derivator by loading the validator's complete output."""
        self.security_id = security_id
        self.validator_output: Optional[Dict[str, Any]] = None
        self.all_periods_data: Optional[List[Dict[str, Any]]] = None
        self.time_series_map: Optional[Dict[str, Any]] = None

        # Per-derivation state variables
        self.schema: Optional[Dict[str, Any]] = None
        self.period_t: Optional[Dict[str, Any]] = None
        self.period_t1: Optional[Dict[str, Any]] = None
        self.results_namespace: Optional[Dict[str, float]] = None

        logger.info(f"[{self.security_id}] Initializing CFS Derivator Engine.")
        try:
            self._load_validated_data()
            self._check_readiness()
            logger.info(f"[{self.security_id}] CFS Derivator initialized successfully.")
        except Exception as e:
            logger.error(f"[{self.security_id}] FAILED to initialize. Error: {e}", exc_info=True)
            raise

    def _load_validated_data(self):
        """Loads the JSON output from the FinancialDataValidator."""
        from financial_analyst.security_folder_utils import get_subfolder
        
        credit_analysis_folder = get_subfolder(self.security_id, "credit_analysis")
        validator_output_path = credit_analysis_folder / "validator_checkpoints" / "validator_final_results.json"

        if not validator_output_path.exists():
            raise FileNotFoundError(f"Validator output file not found at: {validator_output_path}")

        logger.info(f"[{self.security_id}] Loading validated data from: {validator_output_path}")
        with open(validator_output_path, 'r', encoding='utf-8') as f:
            self.validator_output = json.load(f)

        self.time_series_map = self.validator_output.get('metadata', {}).get('time_series_map')
        self.all_periods_data = self.validator_output.get('transformed_data', {}).get('mapped_historical_data')

    def _check_readiness(self):
        """Checks if the loaded data is sufficient for derivation."""
        if not self.time_series_map or not self.all_periods_data:
            raise ValueError("Loaded validator output is missing 'metadata' or 'transformed_data'.")
        
        if len(self.time_series_map.get('annual_backbone', [])) < 2:
            raise ValueError("Annual backbone requires at least 2 periods for delta calculations.")

    def derive_and_save(self) -> Path:
        """
        Orchestrates the CFS derivation process. It iterates through applicable
        periods, loads the corresponding JIT derivation schema, executes the
        derivation, and saves the final, integrated dataset.
        """
        logger.info(f"[{self.security_id}] Starting full derivation and finalization process.")
        if not self.all_periods_data:
            raise RuntimeError("Cannot derive, all_periods_data is not loaded.")

        final_processed_periods = deepcopy(self.all_periods_data)
        backbone = self.time_series_map.get('annual_backbone', [])

        for i in range(1, len(backbone)):
            t_info = backbone[i]
            t1_info = backbone[i-1]
            period_t = final_processed_periods[t_info['data_index']]
            
            quality_assessment = period_t.get('cash_flow_statement', {}).get('cfs_quality_assessment', {})
            if quality_assessment.get('status') == 'RELIABLE':
                logger.info(f"Skipping derivation for {t_info['year']}: Reported CFS is RELIABLE.")
                continue

            try:
                from financial_analyst.security_folder_utils import get_security_file
                schema_path = get_security_file(self.security_id, 'credit_analysis/jit_schemas/cfs_derivation_schema.json')
                with open(schema_path, 'r') as f:
                    derivation_schema = json.load(f)
                logger.info(f"Successfully loaded JIT derivation schema for {t_info['year']}.")
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.warning(f"Could not load/parse JIT derivation schema for {t_info['year']}. Skipping. Error: {e}")
                continue

            logger.info(f"Executing derivation for UNRELIABLE CFS of {t_info['year']}.")
            period_t1 = final_processed_periods[t1_info['data_index']]
            
            derived_cfs, report = self._execute_derivation_from_schema(derivation_schema, period_t, period_t1)

            period_t['cash_flow_statement'] = derived_cfs
            period_t['cfs_derivation_report'] = report
        
        return self._save_final_data(final_processed_periods)

    def _execute_derivation_from_schema(self, schema: dict, period_t: dict, period_t1: dict) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Main orchestrator for the dependency-aware CFS derivation using networkx."""
        self.schema = schema
        self.period_t = period_t
        self.period_t1 = period_t1
        self.results_namespace = {}

        cfs_schema = self.schema.get('cash_flow_statement', {})
        
        # 1. Build dependency graph using networkx
        graph = self._build_dependency_graph(cfs_schema)

        # 2. Validate for circular dependencies using networkx's robust algorithm
        try:
            cycle = nx.find_cycle(graph)
            raise ValueError(f"FATAL: Circular dependency detected in schema. Cycle: {cycle}")
        except nx.NetworkXNoCycle:
            logger.info("Schema is a valid DAG (Directed Acyclic Graph).")

        # 3. Determine calculation order using networkx's topological sort
        calculation_order = list(nx.topological_sort(graph))

        # 4. Execute calculations in the guaranteed correct order
        for item_key in calculation_order:
            item_details = cfs_schema.get(item_key)
            if not item_details or not item_details.get("calculation_template"):
                continue

            value = self._resolve_template(item_details["calculation_template"], cfs_schema)
            self.results_namespace[item_key] = value

        # 5. Calculate subtotals and totals after all components are derived
        self._calculate_subtotals_and_totals(cfs_schema)

        # 6. Assemble final, flattened CFS object in 'key: value' format
        final_cfs_object = {}
        for key in cfs_schema:
            # Assign the calculated value, or 0.0 if it's a structural key without a value (rare)
            final_cfs_object[key] = self.results_namespace.get(key, 0.0)
        
        report = {"status": "DERIVATION_SUCCESS", "message": "CFS derived successfully using networkx."}
        return final_cfs_object, report

    def _build_dependency_graph(self, cfs_schema: dict) -> nx.DiGraph:
        """Constructs a networkx DiGraph from the schema's calculation templates."""
        graph = nx.DiGraph()
        dependency_pattern = re.compile(r'\[([^\]]+)\]')
        all_cfs_keys = set(cfs_schema.keys())

        # Add all items as nodes first
        graph.add_nodes_from(all_cfs_keys)
        
        for item_key, item_details in cfs_schema.items():
            template = item_details.get("calculation_template")
            if template:
                dependencies = dependency_pattern.findall(template)
                for dep in dependencies:
                    # An edge A -> B means A must be calculated before B.
                    if dep in all_cfs_keys:
                        graph.add_edge(dep, item_key)
        return graph

    def _resolve_template(self, template: str, cfs_schema: dict) -> float:
        """Resolves a calculation template by substituting dependencies and evaluating."""
        dependency_pattern = re.compile(r'\[([^\]]+)\]')

        def get_value_for_eval(match):
            key = match.group(1)
            # Distinguish between internal dependencies and source data paths
            if key in cfs_schema:
                if key not in self.results_namespace:
                    raise RuntimeError(f"Dependency '{key}' needed but not yet calculated. This indicates a flaw in the topological sort logic.")
                return str(self.results_namespace[key])
            else:
                return str(self.get_value_from_path(key))

        expression = dependency_pattern.sub(get_value_for_eval, template)
        
        try:
            return eval(expression, {"__builtins__": {}}, {})
        except Exception as e:
            raise ValueError(f"Could not evaluate expression: '{expression}'. Reason: {e}")

    def get_value_from_path(self, path: str) -> float:
        """Resolves a path string to a numerical value from source financial data."""
        is_prior = path.endswith('_prior')
        base_path = path.removesuffix('_prior') if is_prior else path
        
        parts = base_path.split('.')
        if len(parts) < 2:
            raise ValueError(f"Invalid source path format: '{path}'. Expected 'statement.key' or 'statement.sub.key'.")
        
        statement_key = parts[0]
        item_path_parts = parts[1:]

        source_period = self.period_t1 if is_prior else self.period_t
        
        value = source_period.get(statement_key, {})
        for key_part in item_path_parts:
            if not isinstance(value, dict):
                value = None; break
            value = value.get(key_part)

        if value is None:
            logger.warning(f"[{self.security_id}] Path '{path}' resolved to None. Defaulting to 0.0.")
            return 0.0

        return float(value) if isinstance(value, (int, float)) else 0.0
        
    def _calculate_subtotals_and_totals(self, cfs_schema: dict):
        """Calculates all subtotals and totals after component derivation."""
        for _ in range(3): # Iterate to resolve nested subtotals
            for key, details in cfs_schema.items():
                if details.get('level') in ['subtotal', 'total'] and key not in self.results_namespace:
                    children = [
                        item_key for item_key, item_details in cfs_schema.items()
                        if item_details.get('subtotal_of') == key
                    ]
                    if all(child in self.results_namespace for child in children):
                        self.results_namespace[key] = sum(self.results_namespace.get(child, 0) for child in children)

    def _save_final_data(self, final_periods: List[Dict]) -> Path:
        """Saves the final data structure with the new derived CFS data."""
        final_data_to_save = deepcopy(self.validator_output)
        final_data_to_save['transformed_data']['mapped_historical_data'] = final_periods
        
        from financial_analyst.security_folder_utils import require_security_folder
        security_folder = require_security_folder(self.security_id)
        
        output_path = security_folder / "credit_analysis" / "final_derived_financials.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[{self.security_id}] Saving final derived financials to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_data_to_save, f, indent=2, ensure_ascii=False)
            
        return output_path


def derive_financials_for_security(security_id: str):
    """Convenience function to run the entire CFS derivation for a security."""
    try:
        derivator = CFSDerivator(security_id)
        saved_path = derivator.derive_and_save()
        print(f"\nSUCCESS: CFS derivation complete for {security_id}.")
        print(f"Final results saved to: {saved_path}")
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as e:
        print(f"\nERROR: Could not complete derivation for {security_id}.")
        print(f"Reason: {e}")
        # The logger captures the full traceback in the log file.

if __name__ == '__main__':
    # This allows the script to be run directly from the command line for testing.
    # Example: python path/to/cfs_derivator.py YOUR_SECURITY_ID
    if len(sys.argv) != 2:
        print("Usage: python cfs_derivator.py <security_id>")
        sys.exit(1)
        
    security_id_to_process = sys.argv[1]
    print(f"--- Starting Standalone CFS Derivation for Security: {security_id_to_process} ---")
    derive_financials_for_security(security_id_to_process)
    print("--- Standalone Derivation Run Finished ---")