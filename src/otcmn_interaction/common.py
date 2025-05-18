import logging
import re
from typing import List, Dict, Any, Optional, Literal, Tuple
from pathlib import Path

from playwright.async_api import Page, Locator, TimeoutError as PlaywrightTimeoutError

# --- Constants ---
BASE_URL = "https://otc.mn"
SECURITIES_LISTING_URL_PATH = "/securities"
SECURITY_DETAIL_URL_PATH_TEMPLATE = "/securities/detail?id={}" # id is a UUID string
LOGIN_URL_PATH = "/login" # Path, not full URL
DASHBOARD_URL_PATH = "/dashboard" # Path, not full URL

# This should match the constant used in the calling Tool
# For example, if the tool expects the ISIN column to be named "ISIN"
# in the dictionaries it receives.
ISIN_COLUMN_HEADER_TEXT = "ISIN"


# --- Custom Exceptions ---
class OtcmInteractionError(Exception):
    """Base exception for otcmn_interaction module."""
    pass

class PageNavigationError(OtcmInteractionError):
    """Error during page navigation."""
    pass

class ElementNotFoundError(OtcmInteractionError):
    """Expected HTML element was not found."""
    pass

class DataExtractionError(OtcmInteractionError):
    """Error during data extraction from a page."""
    pass

class PageStateError(OtcmInteractionError):
    """Error due to the page being in an unexpected state for an action."""
    pass


class BasePageHandler:
    """
    (Optional) A base class for common Playwright interaction patterns.
    Can be expanded with more helper methods.
    """
    def __init__(self, page: Page, logger: logging.Logger, default_timeout: float):
        self._page = page
        self._logger = logger.getChild(self.__class__.__name__)
        self._default_timeout = default_timeout

    async def _wait_for_locator(
        self,
        locator: Locator,
        state: Literal["attached", "detached", "visible", "hidden"] = "visible",
        description: str = "element",
        timeout_override: Optional[float] = None
    ) -> None:
        effective_timeout = timeout_override if timeout_override is not None else self._default_timeout
        self._logger.debug(f"Waiting for {description} (locator: {locator}) to be {state} with timeout {effective_timeout}ms.")
        try:
            await locator.wait_for(state=state, timeout=effective_timeout)
            self._logger.debug(f"{description} is {state}.")
        except PlaywrightTimeoutError as pte: # <--- Make sure you are CATCHING the ALIASED version
            msg = f"Timeout waiting for {description} to be {state}."
            self._logger.error(msg)
            # Consider taking a screenshot here for debugging
            # await self._page.screenshot(path=f"debug_wait_fail_{description.replace(' ', '_')}.png")
            raise ElementNotFoundError(msg) from pte # <--- And CHAINING the CAUGHT INSTANCE 'pte'

    async def _safe_click(self, locator: Locator, description: str, wait_for_network_idle_after: bool = False, network_idle_timeout: Optional[float] = None) -> None:
        """Clicks a locator safely, waiting for it to be visible and enabled."""
        self._logger.debug(f"Attempting to click {description} (locator: {locator})")
        await self._wait_for_locator(locator, state="visible", description=f"{description} for click")
        await self._wait_for_locator(locator, state="attached", description=f"{description} for click (attached check)") # Check if it's in DOM
        
        is_enabled = await locator.is_enabled()
        if not is_enabled:
            msg = f"Cannot click {description} because it is not enabled."
            self._logger.error(msg)
            raise PageStateError(msg)
        
        try:
            await locator.click(timeout=self._default_timeout / 2) # Shorter timeout for the click action itself
            self._logger.info(f"Clicked {description}.")
            if wait_for_network_idle_after:
                await self._page.wait_for_load_state("networkidle", timeout=network_idle_timeout or self._default_timeout)
                self._logger.debug(f"Network idle after clicking {description}.")
        except PlaywrightTimeoutError as pte:
            msg = f"Timeout during click action on {description}."
            self._logger.error(msg)
            raise ElementNotFoundError(msg) from pte
        except Exception as e:
            msg = f"Unexpected error clicking {description}: {e}"
            self._logger.error(msg, exc_info=True)
            raise OtcmInteractionError(msg) from e

    async def _get_text_content(self, locator: Locator, description: str, default: str = "") -> str:
        """Safely gets text content from a locator."""
        try:
            await self._wait_for_locator(locator, state="visible", description=f"{description} for text content")
            text = await locator.text_content()
            return text.strip() if text else default
        except ElementNotFoundError: # Catch if wait_for_locator fails
            self._logger.warning(f"{description} not found for text extraction, returning default '{default}'.")
            return default
        except Exception as e:
            self._logger.error(f"Error getting text from {description}: {e}", exc_info=True)
            return default
            
    async def _get_attribute(self, locator: Locator, attr_name: str, description: str, default: Optional[str] = None) -> Optional[str]:
        """Safely gets an attribute from a locator."""
        try:
            await self._wait_for_locator(locator, state="attached", description=f"{description} for attribute '{attr_name}'")
            attr_value = await locator.get_attribute(attr_name)
            return attr_value if attr_value is not None else default
        except ElementNotFoundError:
            self._logger.warning(f"{description} not found for attribute '{attr_name}' extraction, returning default.")
            return default
        except Exception as e:
            self._logger.error(f"Error getting attribute '{attr_name}' from {description}: {e}", exc_info=True)
            return default