import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, TypeVar, Generic
from dataclasses import dataclass, field, asdict
import logging
import os
from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional, Dict, Any, List, TypeVar, Generic, Callable
from copy import deepcopy
from pathlib import Path
import json
from datetime import datetime, timedelta

# Type variable for validator input/output
T = TypeVar('T')

# Set up logging with detailed output
logging.basicConfig(
    level=logging.DEBUG,  # Show all logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('financial_data_validator.log')
    ]
)
logger = logging.getLogger(__name__)


class FinancialDataPreprocessor:
    """
    Preprocesses raw financial data into the standardized format expected by validators.
    Transforms flat financial data into structured statements (income_statement, balance_sheet, etc.)
    based on the provided schema.
    """
    
    def __init__(self, schema: Dict[str, Any]):
        """
        Initialize the preprocessor with the validation schema.
        
        Args:
            schema: The validation schema containing field definitions for each statement type
        """
        self.schema = schema
        self.schema_helper = SchemaHelper(schema)
        
    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform raw financial data into the standardized format expected by validators.
        
        Args:
            data: Raw financial data containing historical_financial_statements
            
        Returns:
            Dict with mapped_historical_data in the standardized format
        """
        if not isinstance(data, dict):
            logger.error("Input data must be a dictionary")
            return data
            
        result = data.copy()
        
        # Skip if no historical data or already processed
        if 'historical_financial_statements' not in data:
            logger.warning("No historical_financial_statements found in data")
            return result
            
        # Process each period in the historical data
        mapped_periods = []
        for period in data['historical_financial_statements']:
            if not isinstance(period, dict) or 'financials' not in period:
                continue
                
            mapped_period = {
                'reporting_period_end_date': period.get('reporting_period_end_date'),
                'reporting_period_type': period.get('reporting_period_type'),
                'income_statement': {},
                'balance_sheet': {},
                'cash_flow_statement': {}
            }
            
            # Map each field to its appropriate statement based on the schema
            for field_name, value in period['financials'].items():
                self._map_field(field_name, value, mapped_period)
                
            mapped_periods.append(mapped_period)
            
        result['mapped_historical_data'] = mapped_periods
        # Remove the original raw data to ensure only preprocessed data is passed forward
        if 'historical_financial_statements' in result:
            del result['historical_financial_statements']
        logger.debug(f"Preprocessing complete. Mapped {len(mapped_periods)} periods")
        return result
    
    def _map_field(self, field_name: str, value: Any, period_data: Dict[str, Any]) -> None:
        """
        Map a single field to its appropriate statement based on the schema.
        
        Args:
            field_name: Name of the field to map
            value: Field value
            period_data: The period data to update with the mapped field
        """
        # Check each statement type in the schema
        for statement_type in ['income_statement', 'balance_sheet', 'cash_flow_statement']:
            if field_name in self.schema.get(statement_type, {}):
                period_data[statement_type][field_name] = value
                return
                
        # If field not found in any statement, log a debug message
        logger.debug(f"Field '{field_name}' not found in any statement schema")


@dataclass
class Transformation:
    """Represents a single transformation made during validation"""
    field_path: str
    original_value: Any
    new_value: Any
    reason: str
    validator: str

@dataclass
class ValidationResult:
    """Represents the result of a validation operation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    transformations: List[Transformation] = field(default_factory=list)
    transformed_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None # To hold non-data results like the time series map

    def add_error(self, error: str) -> None:
        self.errors.append(error)
        self.is_valid = False
        
    def add_warning(self, warning: str) -> None:
        self.warnings.append(warning)
    
    def add_transformation(
        self, 
        field_path: str, 
        original_value: Any, 
        new_value: Any, 
        reason: str,
        validator: str
    ) -> None:
        """Record a transformation made during validation"""
        self.transformations.append(
            Transformation(
                field_path=field_path,
                original_value=original_value,
                new_value=new_value,
                reason=reason,
                validator=validator
            )
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary, including transformation details"""
        return {
            "valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "transformations": [
                {
                    "field": t.field_path,
                    "original_value": t.original_value,
                    "new_value": t.new_value,
                    "reason": t.reason,
                    "validator": t.validator
                } for t in self.transformations
            ],
            "transformed_data": self.transformed_data,
            "metadata": self.metadata
        }

class SchemaHelper:
    """Helper class for accessing schema fields consistently"""
    
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
    
    def get_statement_schema(self, statement_type: str) -> Dict[str, Any]:
        """Get schema for a specific statement type (e.g., 'income_statement')"""
        return self.schema.get(statement_type, {})
    
    def get_field_definition(self, statement_type: str, field_name: str) -> Dict[str, Any]:
        """Get definition for a specific field in a statement"""
        statement_schema = self.get_statement_schema(statement_type)
        return statement_schema.get(field_name, {})
    
    def get_field_property(self, statement_type: str, field_name: str, prop_name: str, default: Any = None) -> Any:
        """Get a specific property for a field"""
        field_def = self.get_field_definition(statement_type, field_name)
        return field_def.get(prop_name, default)


class BaseValidator(ABC, Generic[T]):
    """Base class for all validators with transformation support"""
    
    def __init__(self, name: str, schema: Dict[str, Any]):
        self.name = name
        self.schema = schema
        self.schema_helper = SchemaHelper(schema)
        self.checkpoint_dir = None  # Will be set in get_checkpoint_path() when security_id is available
    
    def get_checkpoint_path(self, security_id: str) -> Path:
        """Get the path to save the checkpoint file for this validator"""
        from financial_analyst.security_folder_utils import require_security_folder
        
        # Get the security folder using the helper
        security_folder = require_security_folder(security_id)
        
        # Create the checkpoint directory within the security folder structure
        self.checkpoint_dir = security_folder / "credit_analysis" / "validator_checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        return self.checkpoint_dir / f"validator_state_{self.name.lower()}.json"
    
    def save_checkpoint(self, security_id: str, original_data: Dict[str, Any], result: ValidationResult) -> None:
        """
        Save the validation result to a checkpoint file.
        
        Args:
            security_id: The ID of the security being validated
            original_data: The data as it was before this validator made any changes
            result: The validation result
        """
        checkpoint_path = self.get_checkpoint_path(security_id)
        
        # Create a copy of the original data and include transformed data
        result_dict = result.to_dict()
        checkpoint_data = {
            'security_id': security_id,
            'validator': self.name,
            'original_data': original_data,
            'result': result_dict
        }
        
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    
    # Data preprocessing is now handled by FinancialDataPreprocessor class
    # to avoid code duplication and ensure consistent behavior across all validators

    def validate(self, data: T, security_id: str) -> Tuple[Dict[str, Any], ValidationResult]:
        """
        Validate and transform the data, then save a checkpoint.
        """
        # Create a deep copy to avoid modifying the original
        data_copy = deepcopy(dict(data))
        original_data = deepcopy(data_copy)
        
        # Data should already be preprocessed by FinancialDataPreprocessor
        if 'mapped_historical_data' not in data_copy or not data_copy['mapped_historical_data']:
            raise ValueError("Data must be preprocessed before validation")
        
        # Run the actual validation
        result = self._validate(data_copy)
        
        # Save checkpoint
        # Pass the transformed data to the result object before saving the checkpoint
        result.transformed_data = data_copy
        self.save_checkpoint(security_id, original_data, result)
        
        # Return transformed data and result
        return data_copy, result
    
    @abstractmethod
    def _validate(self, data: T) -> ValidationResult:
        """
        Internal validation method to be implemented by subclasses.
        Should modify the data in-place if transformations are needed.
        
        Returns:
            ValidationResult with any errors/warnings/transformations
        """
        pass


class HierarchicalCompletenessValidator(BaseValidator[Dict[str, Any]]):
    """
    Validates that all three primary financial statements in a period meet a
    minimum completeness score.

    This validator scores each statement ('income_statement', 'balance_sheet',
    'cash_flow_statement') based on the structural importance of its fields.
    A period is only kept if *all three* statements individually achieve a
    score greater than or equal to the defined threshold.
    """

    def __init__(self, schema: Dict[str, Any], score_threshold: int = 15):
        """
        Initializes the validator with scoring weights and a threshold.

        Args:
            schema: The financial data schema.
            score_threshold: The minimum score each statement must achieve.
        """
        super().__init__("hierarchical_completeness_validator", schema)
        self.score_threshold = score_threshold
        self.score_weights = {
            'subtotal': 10,
            'component': 1,
            'total': 10, # Treat total same as subtotal for scoring
            'default': 1
        }
        logger.info(
            f"Initialized HierarchicalCompletenessValidator with threshold: {self.score_threshold}"
        )

    def _validate(self, data: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(True)
        original_periods = data.get('mapped_historical_data', [])
        if not isinstance(original_periods, list):
            return result

        valid_periods = []
        statements_to_check = ['income_statement', 'balance_sheet']

        for i, period in enumerate(original_periods):
            if not isinstance(period, dict):
                continue

            all_statements_sufficient = True
            for statement_type in statements_to_check:
                statement_data = period.get(statement_type)
                statement_schema = self.schema_helper.get_statement_schema(statement_type)

                # A statement is insufficient if its data is missing or its schema is empty
                if not isinstance(statement_data, dict) or not statement_schema:
                    all_statements_sufficient = False
                    break

                current_score = 0
                # CORRECTED: Iterate directly over the schema's fields, not a 'properties' key.
                for field, schema_def in statement_schema.items():
                    # Skip metadata fields that are not dictionaries (e.g., a 'description' string)
                    if not isinstance(schema_def, dict):
                        continue
                    
                    # Score the field if it exists and is not null in the actual data
                    if statement_data.get(field) is not None:
                        level = schema_def.get('level', 'default')
                        current_score += self.score_weights.get(level, self.score_weights['default'])

                # If any single statement fails to meet the threshold, the entire period is invalid.
                if current_score < self.score_threshold:
                    all_statements_sufficient = False
                    break

            if all_statements_sufficient:
                # Ensure summation_plugs is initialized as a dictionary for each statement
                for stmt_type in statements_to_check:
                    if stmt_type in period and isinstance(period[stmt_type], dict):
                        if 'summation_plugs' not in period[stmt_type] or not isinstance(period[stmt_type]['summation_plugs'], dict):
                            period[stmt_type]['summation_plugs'] = {}
                valid_periods.append(period)
            else:
                result.add_transformation(
                    field_path=f"mapped_historical_data[{i}]",
                    original_value=period,
                    new_value=None,
                    reason=f"Period removed. 2 statements did not meet completeness score threshold of {self.score_threshold}.",
                    validator=self.name
                )

        # Replace original list with the filtered list of valid periods.
        data['mapped_historical_data'] = valid_periods
        return result


class DataTypeValidator(BaseValidator[Dict[str, Any]]):
    """
    Validates data types against schema and performs automatic type conversion.
    Records all transformations in the validation result.
    """
    def __init__(self, schema: Dict[str, Any]):
        """Initializes the validator and its type handlers."""
        super().__init__("data_type_validator", schema)
        self.type_handlers = {
            'number': {'check': self._is_number, 'convert': self._convert_to_number},
            'integer': {'check': self._is_integer, 'convert': self._convert_to_integer},
            'string': {'check': self._is_string, 'convert': self._convert_to_string},
            'boolean': {'check': self._is_boolean, 'convert': self._convert_to_boolean},
            'object': {'check': self._is_object, 'convert': None},
            'array': {'check': self._is_array, 'convert': None}
        }

    # This is the new, corrected method.
    def _validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validates and converts data types against the schema for direct values.
        Modifies data in-place and logs summary of actions.
        Ensures all schema fields are present in the output, setting missing fields to 0.0.
        """
        result = ValidationResult(True)
        if not isinstance(data, dict) or 'mapped_historical_data' not in data:
            error_msg = "Invalid data format: missing 'mapped_historical_data'"
            result.add_error(error_msg)
            return result

        statements = data.get('mapped_historical_data', [])
        if not isinstance(statements, list):
            statements = [statements]

        for i, statement_period in enumerate(statements):
            if not isinstance(statement_period, dict):
                continue

            for statement_type in ['income_statement', 'balance_sheet', 'cash_flow_statement']:
                if statement_type not in statement_period:
                    statement_period[statement_type] = {}
                elif not isinstance(statement_period[statement_type], dict):
                    statement_period[statement_type] = {}
                
                # Initialize missing schema fields with 0.0
                statement_schema = self.schema_helper.get_statement_schema(statement_type)
                for field_name in statement_schema.keys():
                    if field_name not in statement_period[statement_type]:
                        statement_period[statement_type][field_name] = 0.0
                        result.add_transformation(
                            field_path=f"mapped_historical_data[{i}].{statement_type}.{field_name}",
                            original_value=None,
                            new_value=0.0,
                            reason=f"Added missing schema field with default value 0.0",
                            validator=self.name
                        )
                
                # Process existing fields
                for field_name, original_value in list(statement_period[statement_type].items()):
                    # Skip summation_plugs as it's added later in the pipeline
                    if field_name == 'summation_plugs':
                        continue
                        
                    field_path = f"mapped_historical_data[{i}].{statement_type}.{field_name}"
                    field_schema = self.schema_helper.get_field_definition(statement_type, field_name)

                    if not field_schema:
                        result.add_warning(f"No schema found for field: {field_path}")
                        continue

                    expected_types = field_schema.get('type', [])
                    if not isinstance(expected_types, list):
                        expected_types = [expected_types]

                    if original_value is None:
                        if 'null' in expected_types:
                            # Transform null to 0.0 if schema allows
                            new_value = 0.0
                            statement_period[statement_type][field_name] = new_value
                            transform_msg = f"Transformed null to 0.0 for {field_path} as allowed by schema"
                            result.add_transformation(
                                field_path=field_path, 
                                original_value=None, 
                                new_value=new_value,
                                reason=transform_msg, 
                                validator=self.name
                            )
                        else:
                            error_msg = f"Field '{field_path}' is null, but schema does not permit. Expected: {expected_types}"
                            result.add_error(error_msg)
                            logger.error(f"VALIDATION ERROR: {error_msg}")
                        continue

                    current_type_str = self._get_value_type_str(original_value)
                    if current_type_str in expected_types:
                        continue

                    # Find a valid target type to convert to
                    target_type = next((t for t in expected_types if t != 'null' and t in self.type_handlers), None)
                    if not target_type:
                        error_msg = f"Field '{field_path}' has invalid type '{current_type_str}'. Schema expects {expected_types} and no valid converter was found."
                        result.add_error(error_msg)
                        logger.error(f"VALIDATION ERROR: {error_msg}")
                        continue

                    converter = self.type_handlers[target_type].get('convert')
                    if not converter:
                        error_msg = f"Field '{field_path}' requires conversion to '{target_type}', but no converter is defined."
                        result.add_error(error_msg)
                        logger.error(f"VALIDATION ERROR: {error_msg}")
                        continue

                    try:
                        converted_value = converter(original_value)
                        statement_period[statement_type][field_name] = converted_value
                        result.add_transformation(
                            field_path=field_path, original_value=original_value, new_value=converted_value,
                            reason=f"Converted from {current_type_str} to fit schema type '{target_type}'", validator=self.name
                        )
                        result.add_warning(f"Field '{field_path}' was auto-converted from {current_type_str} to {target_type}.")
                    except (ValueError, TypeError, AttributeError) as e:
                        # Convert invalid number strings to 0.0 instead of erroring
                        if target_type in ['number', 'integer']:
                            converted_value = 0.0 if target_type == 'number' else 0
                            statement_period[statement_type][field_name] = converted_value
                            result.add_transformation(
                                field_path=field_path, original_value=original_value, new_value=converted_value,
                                reason=f"Converted invalid number string to {converted_value} (original: {original_value})", 
                                validator=self.name
                            )
                            result.add_warning(f"Field '{field_path}' with invalid number string was converted to {converted_value}.")
                        else:
                            result.add_error(f"Field '{field_path}' with value '{original_value}' could not be converted to required type '{target_type}'. Reason: {e}")

        # Add success message if no issues were found
        if result.is_valid and not result.warnings and not result.transformations:
            result.add_warning("All fields passed type validation with no issues or transformations required.")
        
        return result

    def _get_value_type_str(self, value: Any) -> Optional[str]:
        """Maps a Python value to its corresponding schema type string."""
        if self._is_integer(value): return 'integer'
        if self._is_number(value): return 'number'
        if self._is_string(value): return 'string'
        if self._is_boolean(value): return 'boolean'
        if self._is_object(value): return 'object'
        if self._is_array(value): return 'array'
        return type(value).__name__

    # --- Static Helper Methods ---

    @staticmethod
    def _convert_to_number(value: Any) -> float:
        if isinstance(value, bool): raise ValueError("Cannot convert boolean to number")
        return float(value)

    @staticmethod
    def _convert_to_integer(value: Any) -> int:
        if isinstance(value, bool): raise ValueError("Cannot convert boolean to integer")
        if isinstance(value, float) and not value.is_integer(): raise ValueError("Cannot convert float with decimal to integer")
        return int(float(value))

    @staticmethod
    def _convert_to_string(value: Any) -> str:
        return str(value)

    @staticmethod
    def _convert_to_boolean(value: Any) -> bool:
        if isinstance(value, str):
            value = value.lower().strip()
            if value in ('true', 't', 'yes', 'y', '1'): return True
            if value in ('false', 'f', 'no', 'n', '0'): return False
        return bool(value)

    @staticmethod
    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    @staticmethod
    def _is_integer(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    @staticmethod
    def _is_string(value: Any) -> bool:
        return isinstance(value, str)

    @staticmethod
    def _is_boolean(value: Any) -> bool:
        return isinstance(value, bool)

    @staticmethod
    def _is_object(value: Any) -> bool:
        return isinstance(value, dict)

    @staticmethod
    def _is_array(value: Any) -> bool:
        return isinstance(value, list)


class SignValidator(BaseValidator[Dict[str, Any]]):
    """
    Validates that numeric values have the correct sign (positive/negative)
    as defined by the schema. Automatically corrects inverted signs.
    """

    def __init__(self, schema: Dict[str, Any]):
        """Initializes the sign validator."""
        super().__init__("sign_validator", schema)

    def _validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validates and corrects the sign of numeric data against the schema.
        Modifies data in-place if a sign is inverted.
        """
        result = ValidationResult(True)
        statements = data.get('mapped_historical_data', [])
        if not isinstance(statements, list):
            statements = [statements]

        for statement in statements:
            if not isinstance(statement, dict):
                continue
            
            # Find the statement type key (e.g., 'income_statement')
            statement_type = next((k for k in statement if k.endswith('_statement')), None)
            if not statement_type or not isinstance(statement.get(statement_type), dict):
                continue

            for field_name, value in statement[statement_type].items():
                # Skip non-numeric values and None
                if value is None or not isinstance(value, (int, float)):
                    continue

                # Get sign requirement from schema
                expected_sign = self.schema_helper.get_field_property(
                    statement_type, field_name, 'sign'
                )

                if not expected_sign or expected_sign == 'both':
                    continue

                field_path = f"{statement_type}.{field_name}"
                is_correct = False
                if expected_sign == 'positive' and value >= 0:
                    is_correct = True
                elif expected_sign == 'negative' and value <= 0:
                    is_correct = True

                if not is_correct:
                    new_value = -value
                    statement[statement_type][field_name] = new_value

                    reason = f"Corrected sign to match schema rule: '{expected_sign}'"
                    result.add_transformation(
                        field_path=field_path,
                        original_value=value,
                        new_value=new_value,
                        reason=reason,
                        validator=self.name
                    )
                    result.add_warning(
                        f"Field '{field_path}' had incorrect sign. "
                        f"Value '{value}' was automatically changed to '{new_value}'."
                    )
        return result


class SummationIntegrityValidator(BaseValidator[Dict[str, Any]]):
    """
    Validates that component items sum to their subtotals and totals.

    This validator uses the 'subtotal_of' property in the schema.
    Materiality base fields are HARDCODED to 'total_assets' and 'profit_before_tax'
    to avoid schema modifications.
    """

    def __init__(self, schema: Dict[str, Any]):
        super().__init__("summation_integrity_validator", schema)
        self.float_tolerance = 1e-9
        # Materiality thresholds by statement type
        self.materiality_thresholds = {
            'income_statement': 0.40,  # 40% of materiality base
            'balance_sheet': 0.15      # 15% of materiality base
        }
        self._materiality_anchors = self._find_materiality_anchors()
        
    def _find_materiality_anchors(self) -> Dict[str, str]:
        """
        Find materiality anchor fields for each statement type.
        """
        anchors = {}
        
        # Find materiality anchors for each statement type
        for stmt_type, role in [
            ('income_statement', 'IS_PROFIT_ANCHOR'),
            ('balance_sheet', 'BS_TOTAL_ASSETS_ANCHOR')
        ]:
            stmt_schema = self.schema_helper.get_statement_schema(stmt_type)
            for field_name, field_def in stmt_schema.items():
                if field_def.get('schema_role') == role:
                    anchors[stmt_type] = field_name
                    break
        
        return anchors

    def _set_value(self, statement_data: Dict, field: str, value: float) -> None:
        """
        Safely sets a numeric value in statement data.
        Ensures values are stored directly (not nested in a 'value' object).
        """
        if field in statement_data and isinstance(statement_data[field], dict) and 'value' in statement_data[field]:
            # If the field exists and is a value object, update its value
            statement_data[field]['value'] = value
        else:
            # Otherwise, store the value directly
            statement_data[field] = value

    def _get_value(self, statement_data: Dict, field: str) -> float:
        """Safely retrieves and converts a numeric value from statement data."""
        # Check if field exists directly
        if field not in statement_data:
            # The field itself does not exist in the data for this period
            return 0.0
            
        val = statement_data[field]
        
        # Handle nested value object
        if isinstance(val, dict) and 'value' in val:
            val = val['value']
            
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            try:
                # Handle numbers with commas
                return float(val.replace(',', ''))
            except (ValueError, TypeError):
                # The string is not a valid number
                return 0.0
        
        # Value is None, a bool, or some other non-numeric type
        return 0.0

    def _validate(self, data: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        statement_periods = data.get('mapped_historical_data', [])
        if not statement_periods:
            return result

        invalid_period_indices = set()

        for i, period_data in enumerate(statement_periods):
            if not isinstance(period_data, dict):
                continue

            period_is_currently_valid = True
            for statement_type in ['income_statement', 'balance_sheet']:
                if not period_is_currently_valid:
                    break  # Stop checking this period if it's already marked invalid

                if statement_type not in period_data:
                    continue

                statement_data = period_data.get(statement_type, {})
                if not isinstance(statement_data, dict):
                    continue

                # --- Dependency Sorting Logic ---
                all_rules = self._get_summation_rules(statement_type)
                if not all_rules:
                    continue

                processed_subtotals = set(
                    field for field, val in statement_data.items()
                    if field != 'summation_plugs' and val is not None
                )
                
                sorted_rules = []
                rules_to_process = all_rules.copy()
                for _ in range(len(all_rules) + 1):
                    runnable_rules = [
                        rule for rule in rules_to_process
                        if all(comp in processed_subtotals for comp in rule['components'])
                    ]
                    if not runnable_rules:
                        if rules_to_process:
                            unresolved = [r['subtotal'] for r in rules_to_process]
                            result.add_warning(f"Could not resolve summation dependency for: {unresolved}. Check schema.")
                        break
                    for rule in runnable_rules:
                        sorted_rules.append(rule)
                        processed_subtotals.add(rule['subtotal'])
                        rules_to_process.remove(rule)
                # --- End of Sorting Logic ---

                summation_rules = sorted_rules
                materiality_base_field = self._materiality_anchors.get(statement_type)
                threshold = self.materiality_thresholds.get(statement_type, 0.1)  # Default to 10% if not found
                base_value = self._get_value(statement_data, materiality_base_field) if materiality_base_field else 0
                materiality_limit = abs(base_value * threshold) if base_value else float('inf')
                
                if not materiality_base_field:
                    logger.warning(f"No materiality anchor field found for {statement_type}. Using default threshold.")
                    
                plugs_for_this_statement = {}
                field_path_prefix = f"mapped_historical_data[{i}].{statement_type}"

                for rule in summation_rules:
                    subtotal_field = rule['subtotal']
                    reported_subtotal = self._get_value(statement_data, subtotal_field)
                    calculated_sum = sum(self._get_value(statement_data, comp) for comp in rule['components'])
                    
                    # Handle imputation of the subtotal itself if it's zero
                    if abs(reported_subtotal) <= self.float_tolerance and abs(calculated_sum) > self.float_tolerance:
                        statement_data[subtotal_field] = calculated_sum
                        result.add_transformation(
                            field_path=f"{field_path_prefix}.{subtotal_field}",
                            original_value=0.0, new_value=calculated_sum,
                            reason="Imputed zero-value subtotal from sum of its components.",
                            validator=self.name
                        )
                        continue

                    discrepancy = reported_subtotal - calculated_sum

                    if abs(discrepancy) <= self.float_tolerance:
                        continue
                    
                    missing_components = [c for c in rule['components'] if abs(self._get_value(statement_data, c)) <= self.float_tolerance]

                    # CASE 1: Impute a single missing component.
                    if len(missing_components) == 1:
                        missing_field = missing_components[0]
                        statement_data[missing_field] = discrepancy
                        result.add_transformation(
                            field_path=f"{field_path_prefix}.{missing_field}",
                            original_value=0.0, new_value=discrepancy,
                            reason=f"Imputed single missing component to satisfy total '{subtotal_field}'.",
                            validator=self.name
                        )
                        continue
                    
                    # CASE 2: Flag an "Orphaned Total" for downstream logic.
                    elif abs(calculated_sum) <= self.float_tolerance and len(missing_components) > 1:
                        result.add_warning(f"ORPHANED_TOTAL_FLAG::{field_path_prefix}.{subtotal_field}")
                        continue

                    # CASE 3: Unresolvable breach or minor discrepancy to be plugged.
                    if abs(discrepancy) > materiality_limit:
                        result.add_error(
                            f"Unresolvable Material Breach in '{field_path_prefix}.{subtotal_field}': "
                            f"Discrepancy of {discrepancy:,.0f} exceeds materiality limit of {materiality_limit:,.0f} "
                            f"({threshold:.0%} of {materiality_base_field} = {base_value:,.0f}). "
                            f"Reported: {reported_subtotal:,.0f}, Calculated: {calculated_sum:,.0f}. "
                            f"Marking period {i} for removal."
                        )
                        period_is_currently_valid = False
                        break # Stop checking rules for this statement
                    else:
                        plugs_for_this_statement[subtotal_field] = -discrepancy
                        result.add_warning(f"Reconciliation plug created for '{field_path_prefix}.{subtotal_field}'.")

                if not period_is_currently_valid:
                    break # Stop checking other statements for this period

                if plugs_for_this_statement:
                    # Logic to process and add plugs to the statement_data
                    if 'summation_plugs' not in statement_data:
                        statement_data['summation_plugs'] = {}
                    statement_data['summation_plugs'].update(plugs_for_this_statement)

            if not period_is_currently_valid:
                invalid_period_indices.add(i)

        # After checking all periods, perform the "drop" transformation if necessary
        if invalid_period_indices:
            original_data_list = deepcopy(data['mapped_historical_data'])
            
            valid_periods = [
                period for i, period in enumerate(original_data_list) 
                if i not in invalid_period_indices
            ]
            
            data['mapped_historical_data'] = valid_periods
            
            result.add_transformation(
                field_path="mapped_historical_data",
                original_value=[p for i, p in enumerate(original_data_list) if i in invalid_period_indices],
                new_value=None,
                reason=f"Removed {len(invalid_period_indices)} period(s) due to unresolvable material breaches.",
                validator=self.name
            )
            result.is_valid = False

        return result

    def _get_summation_rules(self, statement_type: str) -> List[Dict[str, Any]]:
        """
        Scans the schema and inverts the 'subtotal_of' relationship to
        dynamically build a list of summation rules.
        """
        statement_schema = self.schema_helper.get_statement_schema(statement_type)
        parent_to_children_map = {}

        for field_name, definition in statement_schema.items():
            parent_field = definition.get('subtotal_of')
            if parent_field:
                if parent_field not in parent_to_children_map:
                    parent_to_children_map[parent_field] = []
                parent_to_children_map[parent_field].append(field_name)
        rules = []
        for parent, children in parent_to_children_map.items():
            if parent in statement_schema:
                rules.append({'subtotal': parent, 'components': children})
            else:
                logger.warning(
                    f"Schema inconsistency: Field '{parent}' is listed as a 'subtotal_of' "
                    f"target but is not defined in the '{statement_type}' schema."
                )
        if not rules:
             logger.debug(f"No 'subtotal_of' relationships found to build summation rules for {statement_type}.")

        logger.info(f"[{statement_type}] Found {len(rules)} summation rules to check.")

        return rules


class AccountingEquationValidator(BaseValidator[Dict[str, Any]]):
    """
    Validates that Assets = Liabilities + Equity for each period.

    This validator identifies and records discrepancies from A=L+E.
    - If the discrepancy is material, it raises a hard error.
    - If the discrepancy is non-material, it records a plug in the
      'summation_plugs' dictionary for use by downstream processes.
      It DOES NOT alter the original historical data.
    """
    def __init__(self, schema: Dict[str, Any]):
        super().__init__("accounting_equation_validator", schema)
        self.bs_materiality_threshold = 0.01  # 1% as specified
        self.float_tolerance = 1e-9 # Tolerance for floating point comparisons

    def _get_value(self, statement_data: Dict, field: str) -> float:
        """Safely retrieves a numeric value from statement data."""
        if field not in statement_data:
            return 0.0
            
        val = statement_data.get(field)
        if isinstance(val, (int, float)):
            return float(val)
        return 0.0

    def _validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validates A=L+E for all periods and records discrepancies as plugs.
        """
        result = ValidationResult(True)
        statement_periods = data.get('mapped_historical_data', [])

        for i, period_data in enumerate(statement_periods):
            if not isinstance(period_data, dict) or 'balance_sheet' not in period_data:
                continue

            bs_data = period_data['balance_sheet']
            if not isinstance(bs_data, dict):
                continue

            total_assets = self._get_value(bs_data, 'total_assets')
            total_liabilities = self._get_value(bs_data, 'total_liabilities')
            total_equity = self._get_value(bs_data, 'total_equity')

            if abs(total_assets) < self.float_tolerance:
                continue

            discrepancy = total_assets - (abs(total_liabilities) + total_equity)
            materiality_limit = abs(total_assets * self.bs_materiality_threshold)
            
            field_path_prefix = f"mapped_historical_data[{i}].balance_sheet"

            if abs(discrepancy) <= self.float_tolerance:
                continue

            if abs(discrepancy) > materiality_limit:
                error_msg = (
                    f"Material accounting equation breach in period {i}: "
                    f"Assets ({total_assets:,.2f}) != L+E ({total_liabilities+total_equity:,.2f}). "
                    f"Discrepancy of {discrepancy:,.2f} exceeds limit of {materiality_limit:,.2f}."
                )
                result.add_error(error_msg)
                continue

            # Non-material discrepancy found. Record it as a plug without altering source data.
            if 'summation_plugs' not in bs_data:
                bs_data['summation_plugs'] = {}

            plug_key = "__accounting_equation__"
            # The only data modification is adding the plug itself.
            bs_data['summation_plugs'][plug_key] = discrepancy
            
            result.add_warning(
                f"Accounting equation discrepancy found in period {i}. "
                f"A plug of {discrepancy:,.2f} was recorded for reconciliation."
            )

            # Record the creation of the plug as the sole transformation.
            result.add_transformation(
                field_path=f"{field_path_prefix}.summation_plugs.{plug_key}",
                original_value=None,
                new_value=discrepancy,
                reason="Plug generated to reconcile A=L+E for downstream use.",
                validator=self.name
            )

        return result


class CFS_QualityAssessor(BaseValidator[Dict[str, Any]]):
    """
    Assesses the quality of a reported Cash Flow Statement.

    This validator performs a comprehensive check to determine if a CFS is
    trustworthy enough to be used for learning a company's financial DNA.
    It does not halt on failure, but instead annotates each period's CFS
    with a structured quality assessment object.

    Checks Performed:
    1.  Internal Consistency (Sum of Flows): Verifies that Operating + Investing
        + Financing cash flows sum to the reported Net Change in Cash.
    2.  Internal Consistency (Cash Roll-Forward): Verifies that Beginning Cash
        + Net Change equals the reported Ending Cash.
    3.  Articulation: Verifies that Net Profit from the Income Statement
        materially matches the Net Profit at the start of the CFS.

    Output:
    - Annotates each CFS with a 'cfs_quality_assessment' object containing:
      - status: 'RELIABLE' or 'UNRELIABLE'
      - failure_reasons: A list of specific failure codes.
      - discrepancies: A dictionary of the monetary discrepancy values.
    """
    def __init__(self, schema: Dict[str, Any]):
        super().__init__("cfs_quality_assessor", schema)
        self.cf_materiality_threshold = 0.05  # 5% as specified
        self.float_tolerance = 1e-9

    def _get_value(self, statement_data: Dict, role: str) -> float:
        """Safely retrieves a numeric value using schema role from statement data."""
        if not isinstance(statement_data, dict):
            return 0.0
            
        # First try to find the field by schema_role in the statement data
        for field, props in statement_data.items():
            if not isinstance(props, dict):
                continue
                
            field_role = props.get('schema_role')
            if field_role == role:
                val = props.get('value', props)
                if isinstance(val, (int, float)):
                    return float(val)
        
        # If not found, try direct field access (for backward compatibility)
        field_name = None
        for stmt_type in ['cash_flow_statement', 'income_statement', 'balance_sheet']:
            if stmt_type in self.schema and isinstance(self.schema[stmt_type], dict):
                for field, field_schema in self.schema[stmt_type].items():
                    if field_schema.get('schema_role') == role:
                        field_name = field
                        break
                if field_name:
                    break
                    
        if field_name and field_name in statement_data:
            val = statement_data[field_name]
            if isinstance(val, dict):
                val = val.get('value', 0.0)
            if isinstance(val, (int, float)):
                return float(val)
                
    def _validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Assesses CFS quality for all periods and annotates the data.
        """
        result = ValidationResult(True)
        statement_periods = data.get('mapped_historical_data', [])

        for i, period_data in enumerate(statement_periods):
            # Ensure all required statements exist for a valid assessment
            if not all(k in period_data for k in ['income_statement', 'cash_flow_statement']):
                continue

            cfs_data = period_data['cash_flow_statement']
            is_data = period_data['income_statement']
            if not isinstance(cfs_data, dict) or not isinstance(is_data, dict):
                continue
            
            # Initialize the assessment object for the current period
            assessment = {
                "status": "RELIABLE",
                "failure_reasons": [],
                "discrepancies": {}
            }
            is_reliable = True

            # --- Gather all required values using schema roles ---
            net_op = self._get_value(cfs_data, 'CFS_OPERATING_ANCHOR')
            net_inv = self._get_value(cfs_data, 'CFS_INVESTING_ANCHOR')
            net_fin = self._get_value(cfs_data, 'CFS_FINANCING_ANCHOR')
            reported_net_change = self._get_value(cfs_data, 'CFS_NET_CHANGE_IN_CASH_ANCHOR')
            beg_cash = self._get_value(cfs_data, 'CFS_BEGINNING_CASH_ANCHOR')
            end_cash = self._get_value(cfs_data, 'CFS_ENDING_CASH_ANCHOR')
            cf_net_profit = self._get_value(cfs_data, 'CFS_NET_PROFIT_ARTICULATION')
            is_net_profit = self._get_value(is_data, 'IS_PROFIT_ANCHOR')

            logger.warning(f"Period {i}: net_op={net_op}, net_inv={net_inv}, net_fin={net_fin}, reported_net_change={reported_net_change}, beg_cash={beg_cash}, end_cash={end_cash}, cf_net_profit={cf_net_profit}, is_net_profit={is_net_profit}")
            
            # Check if essential CFS items are zero
            essential_items = [
                ('CFS_OPERATING_ANCHOR', net_op),
                ('CFS_INVESTING_ANCHOR', net_inv),
                ('CFS_FINANCING_ANCHOR', net_fin),
                ('CFS_NET_CHANGE_IN_CASH_ANCHOR', reported_net_change)
            ]
            
            all_zeros = all(abs(val) < self.float_tolerance for _, val in essential_items)
            if all_zeros:
                is_reliable = False
                assessment["failure_reasons"].append("ZERO_VALUES")
                assessment["discrepancies"]["zero_values"] = [item[0] for item in essential_items]
                
            materiality_base = net_op  # Using net operating cash flow as materiality base
            materiality_limit = abs(materiality_base * self.cf_materiality_threshold) if abs(materiality_base) > self.float_tolerance else self.float_tolerance

            # --- Check 1: Sum of flows vs. Net Change ---
            calculated_net_change = net_op + net_inv + net_fin
            discrepancy_sum = calculated_net_change - reported_net_change
            if abs(discrepancy_sum) > materiality_limit:
                is_reliable = False
                assessment["failure_reasons"].append("INTERNAL_MATH_SUM_FAIL")
                assessment["discrepancies"]["sum_discrepancy"] = discrepancy_sum

            # --- Check 2: Cash Roll-Forward ---
            calculated_end_cash = beg_cash + reported_net_change
            discrepancy_rollfwd = calculated_end_cash - end_cash
            if abs(discrepancy_rollfwd) > materiality_limit:
                is_reliable = False
                assessment["failure_reasons"].append("INTERNAL_MATH_ROLLFWD_FAIL")
                assessment["discrepancies"]["rollforward_discrepancy"] = discrepancy_rollfwd
                
            # --- Check 3: Articulation with Income Statement ---
            discrepancy_articulation = is_net_profit - cf_net_profit
            if abs(discrepancy_articulation) > materiality_limit:
                is_reliable = False
                assessment["failure_reasons"].append("ARTICULATION_FAIL")
                assessment["discrepancies"]["articulation_discrepancy"] = discrepancy_articulation

            # Finalize status and annotate the data
            if not is_reliable:
                assessment["status"] = "UNRELIABLE"
                result.add_warning(f"Period {i} CFS was assessed as UNRELIABLE. Reasons: {assessment['failure_reasons']}")

            # Add the assessment object to the CFS data. This is a transformation.
            field_path = f"mapped_historical_data[{i}].cash_flow_statement.cfs_quality_assessment"
            cfs_data['cfs_quality_assessment'] = assessment
            result.add_transformation(
                field_path=field_path,
                original_value=None,
                new_value=assessment,
                reason="Annotated CFS with quality assessment results.",
                validator=self.name
            )
            
        return result


class CFSDerivabilityAssessor(BaseValidator[Dict[str, Any]]):
    """
    Assesses if a Cash Flow Statement can be mathematically derived from the
    available Income Statement and Balance Sheet data for each period.

    This validator does not derive the CFS. It acts as the final gatekeeper,
    checking for the required ingredients for derivation:
    1.  Existence of a prior period (t-1) for calculating deltas.
    2.  Continuity of all necessary balance sheet accounts between t and t-1.
    3.  Presence of key linking items from the Income Statement (e.g., Depreciation).

    It leverages the output of `CFS_QualityAssessor`, only running its checks
    on periods where the reported CFS is marked 'UNRELIABLE'.
    """

    def __init__(self, schema: Dict[str, Any], security_id: str = None):
        super().__init__("cfs_derivability_assessor", schema)
        self.security_id = security_id
        self.float_tolerance = 1e-9

    def _get_value(self, statement_data: Dict, role: str) -> float:
        """Safely retrieves a numeric value using schema role from statement data."""
        if not isinstance(statement_data, dict):
            return 0.0
            
        # First try to find the field by schema_role in the statement data
        for field, props in statement_data.items():
            if not isinstance(props, dict):
                continue
                
            field_role = props.get('schema_role')
            if field_role == role:
                val = props.get('value')
                if isinstance(val, (int, float)):
                    return float(val)
                return 0.0
                
        # Fallback to direct field access if not found via schema_role
        if role in statement_data:
            val = statement_data[role]
            if isinstance(val, dict):
                val = val.get('value', 0.0)
            if isinstance(val, (int, float)):
                return float(val)
                
    def _validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Audits each period to determine CFS derivability, reading prior validation
        results directly from the checkpoint file system.
        """
        from financial_analyst.security_folder_utils import require_security_folder # Local import
        import json

        result = ValidationResult(is_valid=True)
        statement_periods = data.get('mapped_historical_data', [])
        if not statement_periods:
            return result

        # --- Direct File Access Logic ---
        prior_warnings = []
        try:
            # Use the helper to find the security folder
            security_folder = require_security_folder(self.security_id)
            summation_checkpoint_path = security_folder / "credit_analysis" / "validator_checkpoints" / "validator_state_summation_integrity_validator.json"
            
            if summation_checkpoint_path.exists():
                with open(summation_checkpoint_path, 'r', encoding='utf-8') as f:
                    prior_results_data = json.load(f)
                prior_warnings = prior_results_data.get('result', {}).get('warnings', [])
            else:
                logger.warning(f"Could not find summation validator checkpoint for {self.security_id}. Proceeding without prior warnings.")
        except Exception as e:
            logger.error(f"Failed to load prior validator state for {self.security_id}: {e}")
        # --- End of File Access Logic ---
            
        orphaned_total_flags = {
            warning.split('::')[1] for warning in prior_warnings
            if warning.startswith("ORPHANED_TOTAL_FLAG::")
        }

        for i, period_t in enumerate(statement_periods):
            # ... (The rest of the logic from the previous final version is identical) ...
            # ... It will now correctly use the `orphaned_total_flags` set populated from the file ...

            if i == 0:
                # ... (first period logic) ...
                continue

            cfs_data_t = period_t.get('cash_flow_statement', {})
            quality_assessment = cfs_data_t.get('cfs_quality_assessment', {})
            if quality_assessment.get('status') == 'RELIABLE':
                # ... (reliable CFS logic) ...
                continue
            
            period_t1 = statement_periods[i - 1]
            bs_data_t = period_t.get('balance_sheet', {})
            bs_data_t1 = period_t1.get('balance_sheet', {})
            is_data_t = period_t.get('income_statement', {})

            assessment = { "status": "", "checks": {
                    "cfo": {"status": "OK", "missing_items": []},
                    "cfi": {"status": "OK", "missing_items": []},
                    "cff": {"status": "OK", "missing_items": []}
            }}
            all_deltas_available = True
            bs_schema = self.schema_helper.get_statement_schema('balance_sheet')

            for field, definition in bs_schema.items():
                classification = definition.get('cfs_classification')
                if not classification or classification in ['CASH_EQUIVALENT', 'IGNORE']:
                    continue
                if self._get_value(bs_data_t, field) is None or self._get_value(bs_data_t1, field) is None:
                    all_deltas_available = False
                    section = classification.split('_')[0].lower()
                    if section in assessment['checks']:
                        assessment['checks'][section]['status'] = 'FAIL_INCOMPLETE_DELTA'
                        assessment['checks'][section]['missing_items'].append(f"MISSING_DELTA_{field.upper()}")

            # Get depreciation from IS using schema role
            dep_val = self._get_value(is_data_t, 'IS_DEPRECIATION_AMORTIZATION')
            if dep_val is None or abs(dep_val) < self.float_tolerance:
                assessment['checks']['cfi']['missing_items'].append("MISSING_IS_DEPRECIATION_FOR_CAPEX")

            # Get net profit from IS using schema role
            profit_val = self._get_value(is_data_t, 'IS_PROFIT_ANCHOR')
            if profit_val is None or abs(profit_val) < self.float_tolerance:
                assessment['checks']['cff']['missing_items'].append("MISSING_IS_NET_PROFIT_FOR_DIVIDENDS")

            path_prefix = f"mapped_historical_data[{i}].balance_sheet"
            if any(flag.startswith(path_prefix) for flag in orphaned_total_flags):
                for section in ['cfo', 'cfi', 'cff']:
                    assessment['checks'][section]['status'] = 'BROAD_STROKES_ONLY'

            has_missing_deltas = not all_deltas_available
            has_broad_strokes = any(check['status'] == 'BROAD_STROKES_ONLY' for check in assessment['checks'].values())
            has_missing_links = any(check['missing_items'] for check in assessment['checks'].values())

            if has_missing_deltas:
                assessment['status'] = 'NOT_DERIVABLE_INCOMPLETE_DELTAS'
            elif has_broad_strokes:
                assessment['status'] = 'DERIVABLE_BROAD_STROKES_ONLY'
            elif has_missing_links:
                assessment['status'] = 'DERIVABLE_WITH_GAPS'
            else:
                assessment['status'] = 'DERIVABLE_GRANULAR'

            cfs_data_t['cfs_derivation_assessment'] = assessment
            result.add_transformation(
                field_path=f"mapped_historical_data[{i}].cash_flow_statement.cfs_derivation_assessment",
                original_value=None,
                new_value=assessment,
                reason="Annotated period with CFS derivability assessment.",
                validator=self.name
            )

        return result
        

class TimeSeriesCompletenessValidator(BaseValidator[Dict[str, Any]]):
    """
    A final validation step to ensure the dataset is analytically useful.

    This validator runs after all period-level filtering and checks if the
    remaining dataset contains the minimum required data for time-series
    analysis, specifically for CFS derivation.

    It verifies that there are at least two CONSECUTIVE annual periods.
    It does not transform data; it acts as a final gatekeeper.
    """

    def __init__(self, schema: Dict[str, Any]):
        """
        Initializes the time series completeness validator.
        """
        super().__init__("time_series_completeness_validator", schema)
        # Define a tolerance for what "one year" means to account for leap years etc.
        self.one_year_min = timedelta(days=360)
        self.one_year_max = timedelta(days=370)
        logger.info(
            f"Initialized TimeSeriesCompletenessValidator."
        )

    def _validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Checks if at least two consecutive annual periods exist in the data.

        Returns:
            ValidationResult: A pass/fail result. Fails if the structural
                              requirement for consecutive annual data is not met.
        """
        result = ValidationResult(is_valid=True)
        periods = data.get('mapped_historical_data', [])
        if not periods:
            result.add_error("Insufficient Data: No historical periods found after filtering.")
            return result

        # 1. Filter to get only annual periods
        annual_periods = [
            p for p in periods if isinstance(p, dict) and p.get('reporting_period_type') == 'Annual'
        ]

        logger.debug(f"Found {len(annual_periods)} annual periods to check for consecutiveness.")

        # 2. Check if there are at least two annual periods to compare
        if len(annual_periods) < 2:
            result.add_error(
                f"Insufficient Data: Found only {len(annual_periods)} annual period(s). "
                "At least two are required for CFS derivation."
            )
            return result

        # 3. Sort the annual periods by date to ensure correct comparison
        try:
            annual_periods.sort(key=lambda p: p['reporting_period_end_date'])
        except KeyError:
            result.add_error("Data Integrity Error: A period is missing 'reporting_period_end_date'.")
            return result

        # 4. Iterate through the sorted list to find one consecutive pair
        found_consecutive_pair = False
        for i in range(len(annual_periods) - 1):
            try:
                date_t1_str = annual_periods[i+1]['reporting_period_end_date']
                date_t_str = annual_periods[i]['reporting_period_end_date']

                date_t1 = datetime.strptime(date_t1_str, '%Y-%m-%d')
                date_t = datetime.strptime(date_t_str, '%Y-%m-%d')

                delta = date_t1 - date_t
                if self.one_year_min <= delta <= self.one_year_max:
                    found_consecutive_pair = True
                    logger.info(
                        f"Found consecutive annual pair: {date_t_str} and {date_t1_str}. "
                        "Dataset is sufficient for time-series analysis."
                    )
                    break  # Found what we need, no need to check further

            except (ValueError, TypeError, KeyError) as e:
                result.add_warning(
                    f"Could not parse or compare dates for a period due to format error: {e}"
                )
                continue # Skip this pair and check the next one

        # 5. If after checking all pairs, none were consecutive, fail the validation
        if not found_consecutive_pair:
            result.add_error(
                f"Insufficient Data: Found {len(annual_periods)} annual periods, but none are consecutive. "
                "At least two consecutive annual periods are required for CFS derivation."
            )

        # This validator performs no data transformations, so it never modifies the data.
        return result


class PeriodicityAndAnchorMapper(BaseValidator[Dict[str, Any]]):
    """
    Scans the final, clean dataset to create a structural map for downstream tools.

    This validator does not typically "fail" a validation. Instead, its primary
    purpose is to build the 'time_series_map' object which provides an
    explicit "recipe" for the CFS Derivator and Projection Engine.

    It identifies:
    - The consecutive annual periods forming the "backbone" for analysis.
    - Any interim (Quarterly/Semi-Annual) periods that can serve as "anchors".
    """

    def __init__(self, schema: Dict[str, Any]):
        """Initializes the period and anchor mapper."""
        super().__init__("periodicity_and_anchor_mapper", schema)
        logger.info("Initialized PeriodicityAndAnchorMapper.")

    def _validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Builds the time_series_map from the structured financial data.
        """
        result = ValidationResult(is_valid=True)
        periods = data.get('mapped_historical_data', [])
        if not periods:
            # Should be caught by the previous validator, but good to have a safeguard
            result.metadata = {
                'time_series_map': {
                    'analysis_status': 'INSUFFICIENT_DATA',
                    'analysis_message': 'No historical periods found to map.',
                    'annual_backbone': [],
                    'anchor_map': {}
                }
            }
            return result

        annual_backbone = []
        anchor_map = {}

        for i, period in enumerate(periods):
            try:
                end_date_str = period.get('reporting_period_end_date')
                period_type = period.get('reporting_period_type')
                if not (end_date_str and period_type):
                    result.add_warning(f"Period at index {i} is missing date or type information.")
                    continue

                end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
                year = end_date.year

                if period_type == 'Annual':
                    annual_backbone.append({
                        "year": year,
                        "data_index": i
                    })
                else: # It's an anchor (Quarterly, Semi-Annual, etc.)
                    if year not in anchor_map:
                        anchor_map[year] = []
                    anchor_map[year].append({
                        "period_end_date": end_date_str,
                        "period_type": period_type,
                        "data_index": i
                    })

            except (ValueError, TypeError) as e:
                result.add_warning(f"Could not process period at index {i} due to error: {e}")

        # Sort the backbone by year to ensure it's in chronological order
        annual_backbone.sort(key=lambda p: p['year'])

        # Create the final map object
        time_series_map = {
            'analysis_status': 'READY_FOR_DERIVATION',
            'analysis_message': (
                f"Found {len(annual_backbone)} annual period(s) forming the backbone "
                f"and anchor data for {len(anchor_map)} year(s)."
            ),
            'annual_backbone': annual_backbone,
            'anchor_map': anchor_map
        }

        # Attach the map to the result's metadata
        result.metadata = {'time_series_map': time_series_map}
        
        logger.info("Successfully created the time series map.")

        return result


class FinancialDataValidator:
    """
    Validates financial data against industry-specific schemas using a modular validation system.
    """
    
    # Paths
    SCHEMA_DIR = Path("/home/me/CascadeProjects/Alpy/subagents/financial_analyst/mcp_servers/mcp_data_extractor/industry_schemas")
    
    def __init__(self, security_id: str):
        """
        Initialize the validator with a security ID.
        
        Args:
            security_id: The ID of the security to validate
        """
        self.security_id = security_id
        self.mapped_data = None
        self.schema = None
        self.validators: List[BaseValidator] = []
    
    def load_data(self) -> None:
        """
        Load the financial data and metadata for the security.
        """
        # Import here to avoid circular imports
        from financial_analyst.security_folder_utils import require_security_folder
        
        try:
            # Get the security folder path
            security_folder = require_security_folder(self.security_id)
            
            # Load complete data
            complete_data_path = security_folder / "data_extraction" / "complete_data.json"
            if not complete_data_path.exists():
                raise FileNotFoundError(f"Complete data not found at: {complete_data_path}")
                
            with open(complete_data_path, 'r', encoding='utf-8') as f:
                complete_data = json.load(f)
                
            # Extract financial data from complete_data
            self.mapped_data = complete_data.get("bond_financials_historical")
            
            # Load the JIT validation schema
            schema_path = security_folder / "credit_analysis" / "jit_schemas" / "validation_schema.json"
            if not schema_path.exists():
                raise FileNotFoundError(f"JIT validation schema not found at: {schema_path}")
                
            with open(schema_path, 'r', encoding='utf-8') as f:
                self.schema = json.load(f)
                
            if not isinstance(self.schema, dict) or not self.schema:
                raise ValueError("Loaded schema is empty or not a dictionary")
            # Initialize validators
            self._initialize_validators()
            logger.info(f"Successfully loaded data and schema for security {self.security_id}")
            
        except Exception as e:
            logger.error(f"Error loading data for security {self.security_id}: {str(e)}")
            raise
    
    def _initialize_validators(self) -> None:
        """Initialize all validators with the schema"""
        self.validators = [
            HierarchicalCompletenessValidator(self.schema, score_threshold=15),
            DataTypeValidator(self.schema),
            SignValidator(self.schema),
            SummationIntegrityValidator(self.schema),
            AccountingEquationValidator(self.schema),
            CFS_QualityAssessor(self.schema),
            CFSDerivabilityAssessor(self.schema, self.security_id),
            TimeSeriesCompletenessValidator(self.schema),
            PeriodicityAndAnchorMapper(self.schema),
        ]
        
        # Set security_id for all validators that need it
        for validator in self.validators:
            if hasattr(validator, 'security_id'):
                validator.security_id = self.security_id
    
    # In your FinancialDataValidator class

    def validate(self) -> Dict[str, Any]:
        """
        Validate the loaded financial data using all registered validators.
        
        Returns:
            Dict containing combined validation results from all validators
        """
        if not all([self.mapped_data, self.schema]):
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Create a working copy of the data that validators can modify
        working_data = deepcopy(self.mapped_data)
        combined_result = ValidationResult(True)
        
        # Preprocess the raw financial data into the expected format
        try:
            preprocessor = FinancialDataPreprocessor(self.schema)
            working_data = preprocessor.preprocess(working_data)
            
            # Ensure mapped_historical_data exists after preprocessing
            if 'mapped_historical_data' not in working_data or not working_data['mapped_historical_data']:
                raise ValueError("Preprocessing failed - no mapped_historical_data created")
            logger.debug("Successfully preprocessed financial data")
        except Exception as e:
            error_msg = f"Error preprocessing financial data: {str(e)}"
            logger.exception(error_msg)
            combined_result.add_error(error_msg)
            combined_result.is_valid = False
            return combined_result.to_dict()
            
        combined_metadata = {} # To aggregate metadata from all validators

        # Run all validators in sequence, passing the working data
        for validator in self.validators:
            try:
                logger.debug(f"DEBUG: Calling validator {validator.name}.validate()")
                # Each validator receives the latest version of the data
                # and returns the potentially transformed data
                working_data, result = validator.validate(
                    data=working_data,
                    security_id=self.security_id
                )
                
                # Merge results
                if not result.is_valid:
                    combined_result.is_valid = False
                combined_result.errors.extend(result.errors)
                combined_result.warnings.extend(result.warnings)
                combined_result.transformations.extend(result.transformations)
                
                # Merge metadata if it exists
                if result.metadata:
                    combined_metadata.update(result.metadata)
                                
                # If validation failed, we might want to stop early
                if not result.is_valid and result.errors:
                    logger.warning(
                        f"Validator {validator.name} failed with {len(result.errors)} errors. "
                        f"Stopping further validation."
                    )
                    # --- OPTIONAL: uncomment to stop processing on first hard error
                    # break 
                    
            except Exception as e:
                error_msg = f"Error in validator {validator.name}: {str(e)}"
                logger.exception(error_msg)
                combined_result.add_error(error_msg)
                combined_result.is_valid = False
        
        # Save final transformed data and metadata
        combined_result.transformed_data = working_data
        # --- ADD THIS LINE ---
        combined_result.metadata = combined_metadata

        # Save final checkpoint
        self._save_final_checkpoint(working_data, combined_result)
        
        return combined_result.to_dict()

            
    def _save_final_checkpoint(self, data: Dict[str, Any], result: ValidationResult) -> None:
        """Save the final validation results and transformed data"""
        try:
            from financial_analyst.security_folder_utils import require_security_folder
            
            security_folder = require_security_folder(self.security_id)
            checkpoint_dir = security_folder / "credit_analysis" / "validator_checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = checkpoint_dir / "validator_final_results.json"
            
            # Build the dictionary using the newly corrected to_dict method
            # This is a cleaner way to ensure consistency
            final_output_dict = result.to_dict()
            
            # We just need to add the security_id for the checkpoint file
            final_output_dict['security_id'] = self.security_id
            
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                # Dump the entire dictionary, which now includes metadata
                json.dump(final_output_dict, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved final validation results to {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to save final checkpoint: {str(e)}")

def validate_financial_data(security_id: str) -> Dict[str, Any]:
    """
    Convenience function to validate financial data for a given security ID.
    
    Args:
        security_id: The ID of the security to validate
        
    Returns:
        Dict containing validation results
    """
    validator = FinancialDataValidator(security_id)
    validator.load_data()
    return validator.validate()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python financial_data_validator.py <security_id>")
        sys.exit(1)
        
    security_id = sys.argv[1]
    print(f"Starting validation for security ID: {security_id}...")
    result = validate_financial_data(security_id)
    
    # Only show a summary of the validation result
    status = "SUCCESS" if result.get("is_valid", False) else "COMPLETED WITH ISSUES"
    warnings = len(result.get("warnings", []))
    errors = len(result.get("errors", []))
    
    print(f"\nValidation {status}")
    print(f"- Warnings: {warnings}")
    print(f"- Errors: {errors}")
    print(f"- Checkpoint files saved to: {result.get('checkpoint_path', 'N/A')}")
    
    if not result.get("is_valid", True):
        sys.exit(1)