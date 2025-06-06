from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

# List of board directories to search for security folders
CURRENT_ROOT = Path("/home/me/CascadeProjects/Alpy/otcmn_tool_test_output/current")

def list_board_dirs() -> list:
    """
    Return a list of all board directories under CURRENT_ROOT.
    """
    return [d for d in CURRENT_ROOT.iterdir() if d.is_dir()]

# No longer a static list
DEFAULT_BOARD_DIRS = None

def find_security_folder(security_id: str, board_dirs: Optional[List[Path]] = None) -> Optional[Path]:
    """
    Search for the folder corresponding to a security_id in the provided board directories.
    Returns the Path if found, else None.
    """
    if not board_dirs:
        board_dirs = list_board_dirs()
    logger.info(f"[DEBUG] Searching for security_id={security_id} in board_dirs={board_dirs}")
    for board_dir in board_dirs:
        candidate = board_dir / security_id
        logger.info(f"[DEBUG] Checking candidate: {candidate}")
        if candidate.exists() and candidate.is_dir():
            logger.info(f"[DEBUG] Found security folder: {candidate}")
            return candidate
    logger.warning(f"[DEBUG] Security folder for {security_id} not found in any board directory. Checked: {[str(board_dir / security_id) for board_dir in board_dirs]}")
    return None

def require_security_folder(security_id: str, board_dirs: Optional[List[Path]] = None) -> Path:
    """
    Same as find_security_folder, but raises ValueError if not found.
    """
    folder = find_security_folder(security_id, board_dirs)
    if folder is None:
        raise ValueError(f"Security folder for {security_id} not found in any board directory.")
    return folder

def get_subfolder(security_id: str, subfolder: str, board_dirs: Optional[List[Path]] = None) -> Path:
    """
    Get a subfolder (e.g., 'data_extraction', 'parsed_images') inside the security folder.
    Raises ValueError if the security folder is not found.
    """
    folder = require_security_folder(security_id, board_dirs)
    return folder / subfolder

def get_security_file(security_id: str, filename: str, board_dirs: Optional[List[Path]] = None) -> Path:
    """
    Get a file path inside the security folder.
    Raises ValueError if the security folder is not found.
    """
    folder = require_security_folder(security_id, board_dirs)
    return folder / filename
