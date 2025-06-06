"""
Value corrector module for applying direct corrections to extracted data.
"""
import copy
import json
import logging
from pathlib import Path
from typing import Dict, Any, Union, List

logger = logging.getLogger("MCPQualityChecker")

def get_value_at_path(data: Dict[str, Any], path: str) -> Any:
    """
    Get a value at a specific path in a nested dictionary.
    
    Args:
        data: The nested dictionary to query
        path: Dot-notation path (e.g., "bond_metadata.principal_amount_issued")
              or array notation (e.g., "bond_financials_historical.historical_financial_statements[0].financials.total_assets")
        
    Returns:
        Any: The value at the path, or None if path is invalid/not found
    """
    try:
        # Split the path into components
        parts = []
        current = ""
        in_brackets = False
        
        for char in path:
            if char == '.' and not in_brackets:
                if current:
                    parts.append(current)
                current = ""
            elif char == '[':
                if current:
                    parts.append(current)
                current = "["
                in_brackets = True
            elif char == ']' and in_brackets:
                current += "]"
                parts.append(current)
                current = ""
                in_brackets = False
            else:
                current += char
                
        if current:
            parts.append(current)
            
        # Navigate to the value
        current_data = data
        for i, part in enumerate(parts):
            if part.startswith('[') and part.endswith(']'):
                idx = int(part[1:-1])
                current_data = current_data[idx]
            elif '[' in part and part.endswith(']') and part.index('[') > 0:
                key, idx_str = part.split('[', 1)
                idx = int(idx_str[:-1])
                current_data = current_data[key][idx]
            else:
                current_data = current_data[part]
        return current_data
    except (KeyError, IndexError, ValueError, TypeError):
        return None

def set_value_at_path(data: Dict[str, Any], path: str, value: Any) -> bool:
    """
    Set a value at a specific path in a nested dictionary.
    
    Args:
        data: The nested dictionary to modify
        path: Dot-notation path (e.g., "bond_metadata.principal_amount_issued")
              or array notation (e.g., "bond_financials_historical.historical_financial_statements[0].financials.total_assets")
        value: The value to set at the path
        
    Returns:
        bool: True if successful, False if path is invalid
    """
    try:
        # Split the path into components
        parts = []
        current = ""
        in_brackets = False
        
        for char in path:
            if char == '.' and not in_brackets:
                if current: # Only append if current is not empty
                    parts.append(current)
                current = ""
            elif char == '[':
                if current: # Only append if there's a key before the bracket
                    parts.append(current)
                current = "["
                in_brackets = True
            elif char == ']' and in_brackets:
                current += "]"
                parts.append(current)
                current = ""
                in_brackets = False
            else:
                current += char
                
        if current:
            parts.append(current)
            
        # Navigate to the parent object
        parent = data
        final_key = parts[-1]
        
        for i, part in enumerate(parts[:-1]):
            if part.startswith('[') and part.endswith(']'):
                # Handle array index
                idx = int(part[1:-1])
                parent = parent[idx]
            elif '[' in part and part.endswith(']'):
                # Handle object key with array index
                key, idx_str = part.split('[', 1)
                idx = int(idx_str[:-1])
                parent = parent[key][idx]
            else:
                # Handle regular object key
                parent = parent[part]
        
        # Set the value in the final key
        if final_key.startswith('[') and final_key.endswith(']'):
            # Handle array index
            idx = int(final_key[1:-1])
            parent[idx] = value
        elif '[' in final_key and final_key.endswith(']'):
            # Handle object key with array index
            key, idx_str = final_key.split('[', 1)
            idx = int(idx_str[:-1])
            parent[key][idx] = value
        else:
            # Handle regular object key
            parent[final_key] = value
            
        return True
    except (KeyError, IndexError, ValueError, TypeError) as e:
        logger.error(f"Error setting value at path '{path}': {type(e).__name__}: {e}")
        return False

def apply_corrections(data: Dict[str, Any], corrections: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply a set of corrections to the data.
    
    Args:
        data: The data to correct
        corrections: Dictionary mapping paths to corrected values
        
    Returns:
        The corrected data
    """
    # CRITICAL: Use deep copy to avoid modifying the original data structure
    corrected_data = copy.deepcopy(data)
    successful_corrections = 0
    failed_corrections = 0
    
    for path, value in corrections.items():
        success = set_value_at_path(corrected_data, path, value)
        if success:
            logger.info(f"Applied correction at path '{path}': {value}")
            successful_corrections += 1
        else:
            logger.warning(f"Failed to apply correction at path '{path}'")
            failed_corrections += 1
    
    logger.info(f"Applied {successful_corrections} corrections, {failed_corrections} failed")
    return corrected_data

def apply_corrections_to_file(file_path: str, corrections: Dict[str, Any]) -> bool:
    """
    Apply corrections to a JSON file.
    
    Args:
        file_path: Path to the JSON file to correct
        corrections: Dictionary mapping paths to corrected values
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load the data
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Apply corrections
        corrected_data = apply_corrections(data, corrections)
        
        # Write the corrected data back
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(corrected_data, f, indent=2, ensure_ascii=False)
            
        return True
    except Exception as e:
        logger.error(f"Error applying corrections to file '{file_path}': {e}")
        return False
