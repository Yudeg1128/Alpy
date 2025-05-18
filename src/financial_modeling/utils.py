import logging
import re
from typing import Tuple, Optional, Any

# Setup a logger for this module
logger = logging.getLogger(__name__)
# Basic configuration if no handlers are set up by the main application
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class FinancialModelingUtils:
    # Regex to find numbers, allowing for commas, decimals, and optional parentheses for negatives
    # Handles cases like: 1,234.56 or (1,234.56) or 1234
    NUMERIC_PATTERN = re.compile(r"^\(?\s*([0-9,]+(?:\.[0-9]+)?)\s*\)?$")
    
    # Common currency symbols/codes and their standard forms
    CURRENCY_MAP = {
        "₮": "MNT",
        "төгрөг": "MNT",
        "төг": "MNT",
        "mnt": "MNT",
        "usd": "USD",
        "$": "USD",
        "доллар": "USD",
        "eur": "EUR",
        "€": "EUR",
        "евро": "EUR",
        # Add more mappings as needed
    }

    # Common unit keywords and their multipliers
    # Order matters: check for longer strings first (e.g., "миллион" before "мян")
    UNIT_MULTIPLIERS = {
        "billion": 1_000_000_000,
        "тэрбум": 1_000_000_000,
        "тэр.": 1_000_000_000,
        "млн.": 1_000_000, # More specific first
        "million": 1_000_000,
        "миллион": 1_000_000,
        "сая": 1_000_000,
        "млн": 1_000_000,   # Less specific after
        "мян.": 1_000,   # More specific
        "thousand": 1_000,
        "мянган": 1_000,
        "k": 1_000, 
        "actuals": 1,
        "units": 1,
        "үндсэн": 1
    }

    EXTRACT_NUMERIC_PATTERN = re.compile(r"^\(?\s*([0-9,]+(?:\.[0-9]+)?)\s*\)?(?:[^0-9.].*)?$")

# In class FinancialModelingUtils:
    # No need for EXTRACT_NUMERIC_PATTERN if we strip symbols first
    # NUMERIC_PATTERN was for validating a "clean" number string
    # NUMERIC_PATTERN = re.compile(r"^\(?\s*([0-9,]+(?:\.[0-9]+)?)\s*\)?$")


    @staticmethod
    def parse_numerical_value(value_str: Optional[Any]) -> Optional[float]:
        if value_str is None:
            return None
        
        s = str(value_str).strip()
        if not s:
            return None

        is_negative = False
        if s.startswith('(') and s.endswith(')'):
            is_negative = True
            s = s[1:-1].strip() # Remove parentheses first

        # Strip common currency symbols from start or end, and all commas
        # More robust symbol stripping might look for them specifically at start/end
        # or use a more comprehensive list.
        s_cleaned_symbols = s
        for sym in ['₮', '$', '€', 'usd', 'eur', 'mnt']: # Add more as needed, case-insensitive
            s_cleaned_symbols = s_cleaned_symbols.replace(sym, '').replace(sym.upper(), '')
        
        s_cleaned_symbols = s_cleaned_symbols.strip()
        s_cleaned_commas = s_cleaned_symbols.replace(',', '')

        # After cleaning symbols and commas, try to extract a leading float
        # This regex will find a number at the start of the string, ignoring any trailing non-numeric text.
        # E.g., "75.2 млн.төг" -> "75.2"
        #       "50.25" -> "50.25"
        match = re.match(r"^\s*(-?[0-9\.]+)\s*.*", s_cleaned_commas)
        numeric_part_to_convert = None

        if match:
            numeric_part_to_convert = match.group(1)
        else:
            # If the initial regex doesn't match (e.g. string was just "123"),
            # s_cleaned_commas might be the number itself.
            numeric_part_to_convert = s_cleaned_commas


        if numeric_part_to_convert:
            try:
                val = float(numeric_part_to_convert)
                return -val if is_negative and val > 0 else val # Apply negative only if not already applied by float() for "-..."
            except ValueError:
                logger.debug(f"Could not convert extracted/cleaned numeric part '{numeric_part_to_convert}' to float from original '{str(value_str)}'.")
                return None
        else:
            logger.debug(f"No numeric part could be isolated from '{str(value_str)}'.")
            return None

    @staticmethod
    def standardize_currency(currency_str: Optional[str]) -> Optional[str]:
        """
        Standardizes common currency symbols/names to their ISO codes.
        """
        if not currency_str or not isinstance(currency_str, str):
            return None
        
        # Attempt direct match first
        curr_lower = currency_str.strip().lower()
        if curr_lower in FinancialModelingUtils.CURRENCY_MAP:
            return FinancialModelingUtils.CURRENCY_MAP[curr_lower]
        
        # If it's already a standard 3-letter code, return it uppercased
        if len(curr_lower) == 3 and curr_lower.isalpha():
            return curr_lower.upper()
            
        logger.debug(f"Could not standardize currency: '{currency_str}'")
        return currency_str # Return original if no mapping found, uppercased if seems like a code

    @staticmethod
    def get_unit_multiplier(unit_str: Optional[str]) -> Tuple[float, Optional[str]]:
        """
        Determines the multiplier based on unit keywords (e.g., "thousands", "сая").
        Returns (multiplier, identified_unit_keyword from UNIT_MULTIPLIERS or original unit_str).
        """
        if not unit_str or not isinstance(unit_str, str):
            return 1.0, unit_str

        unit_lower_stripped = unit_str.strip().lower()
        
        # Iterate in a way that allows longer keys to be checked first if there's ambiguity,
        # though explicit ordering in UNIT_MULTIPLIERS is usually better.
        # For this case, simple iteration should be fine if keys are distinct enough or ordered.
        for keyword_in_map, multiplier in FinancialModelingUtils.UNIT_MULTIPLIERS.items():
            # Check if the keyword_in_map (from our dictionary) is present in the input unit_lower_stripped
            # This is more robust than checking if unit_lower_stripped startswith/endswith keyword
            if keyword_in_map in unit_lower_stripped:
                 # Check for whole word match to avoid "m" in "million" matching "m" if "m" was a unit
                 # This can be done by checking boundaries or using regex.
                 # For simplicity, if `keyword_in_map` is found, we assume it's the intended one.
                 # A more precise match would involve regex word boundaries.
                 # Example: pattern = r'\b' + re.escape(keyword_in_map) + r'\b'
                 # if re.search(pattern, unit_lower_stripped):
                return multiplier, keyword_in_map # Return the KEY from the map

        logger.debug(f"No specific unit multiplier keyword found for: '{unit_str}'. Assuming 1.0.")
        return 1.0, unit_str 

    @staticmethod
    def normalize_financial_value(
        value_input: Optional[Any], 
        unit_input: Optional[str] = None, 
        currency_input: Optional[str] = None
    ) -> Tuple[Optional[float], Optional[str], Optional[str]]:
        """
        Comprehensive normalization for a financial value.
        1. Parses the numerical value string (handles commas, parentheses).
        2. Standardizes the currency.
        3. Applies a multiplier based on the unit string.

        Returns: (normalized_float_value, standardized_currency_code, original_unit_description)
        """
        numerical_value = FinancialModelingUtils.parse_numerical_value(value_input)
        standardized_currency = FinancialModelingUtils.standardize_currency(currency_input)
        
        if numerical_value is None:
            # If value parsing failed, still return standardized currency and original unit
            return None, standardized_currency, unit_input

        multiplier, identified_unit_keyword = FinancialModelingUtils.get_unit_multiplier(unit_input)
        
        final_value = numerical_value * multiplier
        
        return final_value, standardized_currency, unit_input # Return original unit_input for record keeping

# --- Standalone Test Function ---
def run_tests():
    print("--- Testing FinancialModelingUtils ---")

    test_cases_value_parsing = [
        ("1,234.56", 1234.56), ("(1,234.56)", -1234.56), ("1234", 1234.0),
        ("  ( 500 )  ", -500.0), ("-600.0", -600.0), ("abc", None), ("", None), (None, None),
        ("100₮", 100.0), ("(1,000.00 $)", -1000.0), ("€50.25", 50.25)
    ]
    print("\nTesting parse_numerical_value:")
    for i, (inp, expected) in enumerate(test_cases_value_parsing):
        result = FinancialModelingUtils.parse_numerical_value(inp)
        assert result == expected, f"Test {i} failed: input='{inp}', expected={expected}, got={result}"
        print(f"Input: '{inp}', Parsed: {result} (Expected: {expected}) - {'PASS' if result == expected else 'FAIL'}")

    test_cases_currency = [
        ("₮", "MNT"), ("төгрөг", "MNT"), ("USD", "USD"), ("$", "USD"), ("eur", "EUR"),
        ("gbp", "GBP"), ("unknown", "unknown"), (None, None), ("", None), ("  MNT  ", "MNT")
    ]
    print("\nTesting standardize_currency:")
    for i, (inp, expected) in enumerate(test_cases_currency):
        result = FinancialModelingUtils.standardize_currency(inp)
        assert result == expected, f"Test {i} failed: input='{inp}', expected={expected}, got={result}"
        print(f"Input: '{inp}', Standardized: {result} (Expected: {expected}) - {'PASS' if result == expected else 'FAIL'}")

    test_cases_unit_multiplier = [
        ("thousands", (1000.0, "thousand")), (" сая ", (1000000.0, "сая")), ("млн. төгрөг", (1000000.0, "млн.")),
        ("Actuals", (1.0, "actuals")), ("тэрбум USD", (1000000000.0, "тэрбум")),
        ("in MNT millions", (1000000.0, "million")), ("  kEUR  ", (1000.0, "k")),
        ("units", (1.0, "units")), (None, (1.0, None)), ("xyz", (1.0, "xyz"))
    ]
    print("\nTesting get_unit_multiplier:")
    for i, (inp, expected) in enumerate(test_cases_unit_multiplier):
        result_mult, result_unit_kw = FinancialModelingUtils.get_unit_multiplier(inp)
        exp_mult, exp_unit_kw = expected
        assert result_mult == exp_mult and result_unit_kw == exp_unit_kw, \
            f"Test {i} failed: input='{inp}', expected=({exp_mult}, '{exp_unit_kw}'), got=({result_mult}, '{result_unit_kw}')"
        print(f"Input: '{inp}', Multiplier: {result_mult}, ID'd Unit: '{result_unit_kw}' (Expected: {expected}) - {'PASS' if result_mult == exp_mult and result_unit_kw == exp_unit_kw else 'FAIL'}")

# In run_tests():
    test_cases_full_normalization = [
        # Each element is now: ( (input_tuple), (expected_tuple) )
        ( ("1,250", "сая", "₮"), (1250000000.0, "MNT", "сая") ),
        ( ("(50.5)", "thousands", "USD"), (-50500.0, "USD", "thousands") ),
        ( ("100", None, "$"), (100.0, "USD", None) ),
        # For the case "75.2 млн.төг", currency is embedded. Let's assume it's extracted separately for the util.
        # If currency is None, standardize_currency will return None.
        ( ("75.2 млн.төг", "млн.төг", None), (75200000.0, None, "млн.төг") ), 
        ( (200, "тэрбум", "EUR"), (200000000000.0, "EUR", "тэрбум") ),
        ( ("abc", "millions", "USD"), (None, "USD", "millions") ),
        ( (None, "thousands", "MNT"), (None, "MNT", "thousands") ),
        ( ("100", "units", "MNT"), (100.0, "MNT", "units") ),
        ( ("100", "үндсэн", "MNT"), (100.0, "MNT", "үндсэн") )
    ]
    print("\nTesting normalize_financial_value:")
    for i, (inputs_tuple, expecteds_tuple) in enumerate(test_cases_full_normalization): # Corrected unpacking
        val_in, unit_in, curr_in = inputs_tuple # Unpack inputs
        exp_val, exp_curr, exp_orig_unit = expecteds_tuple # Unpack expecteds
        
        res_val, res_curr, res_orig_unit = FinancialModelingUtils.normalize_financial_value(val_in, unit_in, curr_in)
        
        val_match = (res_val == exp_val) or (res_val is None and exp_val is None)
        curr_match = res_curr == exp_curr
        unit_match = res_orig_unit == exp_orig_unit

        assert val_match and curr_match and unit_match, \
            f"Test {i} failed: inputs=({val_in}, '{unit_in}', '{curr_in}'), " \
            f"expected=({exp_val}, '{exp_curr}', '{exp_orig_unit}'), " \
            f"got=({res_val}, '{res_curr}', '{res_orig_unit}')"
        print(f"Inputs: ({val_in}, '{unit_in}', '{curr_in}'), Result: ({res_val}, '{res_curr}', '{res_orig_unit}') (Expected: {expecteds_tuple}) - {'PASS' if val_match and curr_match and unit_match else 'FAIL'}")
    print("\n--- All tests finished ---")


if __name__ == "__main__":
    # To run tests: python src/financial_modeling/utils.py
    run_tests()