import json
import pandas as pd
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('projection_data_prepper.log')
    ]
)
logger = logging.getLogger(__name__)

def prepare_dataframe_for_projection(security_id: str) -> Path:
    """
    Loads derived financial data and transforms it into a clean Pandas DataFrame.

    This function acts as the first step in the projection pipeline. It takes
    the final JSON output from the CFS Derivator, flattens the nested statements
    into a single structure, and saves it as a time-indexed CSV file, ready for
    projection modeling.

    Args:
        security_id: The ID of the security to process.

    Returns:
        The path to the saved CSV file containing the master DataFrame.

    Raises:
        FileNotFoundError: If the 'final_derived_financials.json' file is not found.
        ValueError: If the loaded data is empty or missing key structures.
    """
    logger.info(f"[{security_id}] Starting data preparation for projection.")

    # 1. Locate and load the input file
    try:
        # Import here to maintain module independence
        from financial_analyst.security_folder_utils import require_security_folder
        security_folder = require_security_folder(security_id)
        input_dir = security_folder / "credit_analysis"
        input_path = input_dir / "final_derived_financials.json"

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found at: {input_path}")

        logger.info(f"[{security_id}] Loading derived data from: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            source_data = json.load(f)

    except Exception as e:
        logger.error(f"[{security_id}] Failed to load source data. Error: {e}")
        raise

    # 2. Extract and flatten the historical data
    historical_periods = source_data.get('transformed_data', {}).get('mapped_historical_data', [])
    if not historical_periods:
        raise ValueError("Source data does not contain 'mapped_historical_data' or it is empty.")

    flattened_data = []
    for period in historical_periods:
        flat_period = {}
        flat_period['reporting_period_end_date'] = period.get('reporting_period_end_date')
        flat_period['reporting_period_type'] = period.get('reporting_period_type')

        for statement_type in ['income_statement', 'balance_sheet', 'cash_flow_statement']:
            statement = period.get(statement_type, {})
            if isinstance(statement, dict):
                total_plug_value = 0.0 # Initialize plug total for the statement
                for key, value in statement.items():
                    if key == 'summation_plugs' and isinstance(value, dict):
                        # Sum up all plug values for this statement
                        total_plug_value = sum(v for v in value.values() if isinstance(v, (int, float)))
                    elif isinstance(value, (int, float)):
                        flat_period[key] = value
                
                # Add the total plug value as a new column
                # e.g., 'bs_summation_plugs_total'
                plug_col_name = f"{statement_type[:2]}_summation_plugs_total"
                flat_period[plug_col_name] = total_plug_value

        flattened_data.append(flat_period)

    logger.info(f"[{security_id}] Successfully flattened {len(flattened_data)} periods.")

    # 3. Create and polish the DataFrame
    df = pd.DataFrame(flattened_data)

    if 'reporting_period_end_date' not in df.columns:
        raise ValueError("Flattened data is missing 'reporting_period_end_date' column.")

    # Convert to datetime and set as index
    df['reporting_period_end_date'] = pd.to_datetime(df['reporting_period_end_date'])
    df = df.set_index('reporting_period_end_date')

    # Sort the dataframe chronologically
    df = df.sort_index()

    logger.info(f"[{security_id}] Created DataFrame with shape: {df.shape}")

    # 4. Save the DataFrame to a CSV file
    output_path = input_dir / "projection_master_dataframe.csv"
    try:
        df.to_csv(output_path)
        logger.info(f"[{security_id}] Successfully saved master DataFrame to: {output_path}")
    except Exception as e:
        logger.error(f"[{security_id}] Failed to save DataFrame to CSV. Error: {e}")
        raise

    return output_path

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m projection.data_prepper <security_id>")
        sys.exit(1)

    security_id_to_process = sys.argv[1]
    try:
        final_path = prepare_dataframe_for_projection(security_id_to_process)
        print(f"\nSuccess! Master DataFrame for '{security_id_to_process}' is ready for projection.")
        print(f"File saved to: {final_path}")
    except Exception as e:
        print(f"\nAn error occurred during data preparation for '{security_id_to_process}': {e}")
        sys.exit(1)