import logging
import asyncio # For ensure_logged_in if implemented that way

from playwright.async_api import Page
from typing import List, Dict, Any, Optional, Tuple

from .common import BASE_URL, SECURITIES_LISTING_URL_PATH, SECURITY_DETAIL_URL_PATH_TEMPLATE, LOGIN_URL_PATH, DASHBOARD_URL_PATH, PageNavigationError, OtcmInteractionError
from .listing_page_handler import ListingPageHandler
from .detail_page_handler import DetailPageHandler # Stub for now

class OtcmSiteInteractor:
    def __init__(self, page: Page, logger: logging.Logger, default_page_timeout: float = 30000.0):
        self._page = page
        self._logger = logger.getChild(self.__class__.__name__)
        self._default_timeout = default_page_timeout #ms

        self.listing_handler = ListingPageHandler(page, self._logger, default_page_timeout)
        self.detail_handler = DetailPageHandler(page, self._logger, default_page_timeout) # Stub
        self._logger.info("OtcmSiteInteractor initialized.")

    async def navigate_to_securities_listing_page(self) -> None:
        """Navigates to the main securities listing page and verifies it."""
        full_url = BASE_URL + SECURITIES_LISTING_URL_PATH
        self._logger.info(f"Navigating to securities listing page: {full_url}")
        try:
            await self._page.goto(full_url, wait_until="domcontentloaded", timeout=self._default_timeout)
            current_url = self._page.url
            if LOGIN_URL_PATH in current_url.lower() and SECURITIES_LISTING_URL_PATH not in current_url.lower():
                 raise PageNavigationError(f"Redirected to login page ({current_url}) when trying to access securities list. Manual login might be required.")
            await self.listing_handler.verify_initial_elements()
            self.listing_handler.reset_header_cache() # Important for fresh state
            self._logger.info("Successfully navigated to securities listing page.")
        except Exception as e:
            msg = f"Failed to navigate to or verify securities listing page: {e}"
            self._logger.error(msg, exc_info=True)
            # Consider screenshot on failure: await self._page.screenshot(path="debug_nav_listing_fail.png")
            raise PageNavigationError(msg) from e

    async def navigate_to_security_detail_page(self, security_url_id: str) -> None:
        """Navigates to a specific security detail page using its URL ID."""
        if not security_url_id:
            raise ValueError("security_url_id cannot be empty for detail page navigation.")
            
        detail_path_segment = SECURITY_DETAIL_URL_PATH_TEMPLATE.format(security_url_id)
        full_url = BASE_URL + detail_path_segment
        self._logger.info(f"Navigating to security detail page: {full_url}")
        try:
            await self._page.goto(full_url, wait_until="domcontentloaded", timeout=self._default_timeout)
            current_url = self._page.url
            if LOGIN_URL_PATH in current_url.lower() and detail_path_segment not in current_url.lower():
                 raise PageNavigationError(f"Redirected to login page ({current_url}) when trying to access security detail. Manual login might be required.")
            
            # TODO: Add verification for detail page elements once DetailPageHandler is implemented
            # For now, just log success.
            # await self.detail_handler.verify_initial_elements() # Example
            self._logger.info(f"Successfully navigated to security detail page for ID: {security_url_id}.")
        except Exception as e:
            msg = f"Failed to navigate to security detail page for ID {security_url_id}: {e}"
            self._logger.error(msg, exc_info=True)
            raise PageNavigationError(msg) from e
    
    async def ensure_logged_in(self, login_prompt_message: Optional[str] = None, login_wait_timeout: float = 300.0) -> None:
        """
        Checks if logged in by visiting dashboard. If on login page, prompts for manual login.
        This is primarily for non-CDP scenarios where the tool launches the browser.
        """
        self._logger.info("Checking login status by navigating to dashboard...")
        dashboard_full_url = BASE_URL + DASHBOARD_URL_PATH
        login_full_url_pattern = BASE_URL + LOGIN_URL_PATH # Used for checking

        await self._page.goto(dashboard_full_url, wait_until="domcontentloaded", timeout=self._default_timeout)
        
        # Normalize URLs for comparison (remove trailing slashes, query params if any)
        current_url_normal = self._page.url.split('?')[0].rstrip('/')
        login_url_normal = login_full_url_pattern.split('?')[0].rstrip('/')

        if current_url_normal == login_url_normal:
            prompt_msg = login_prompt_message or "MANUAL LOGIN REQUIRED in the browser. The page is currently on the login screen."
            self._logger.warning(prompt_msg)
            self._logger.info(f"Please log in and then press Enter in the console (timeout: {login_wait_timeout}s).")
            
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(input, "Press Enter in this console after completing login in the browser: "),
                    timeout=login_wait_timeout
                )
                self._logger.info("Login confirmation received from console.")
                # Re-check by navigating to dashboard again
                await self._page.goto(dashboard_full_url, wait_until="domcontentloaded", timeout=self._default_timeout)
                current_url_normal = self._page.url.split('?')[0].rstrip('/')
                if current_url_normal == login_url_normal:
                    raise OtcmInteractionError("Login attempt failed or was not completed. Still on login page after prompt.")
                self._logger.info("Login confirmed: Successfully accessed dashboard after prompt.")
            except asyncio.TimeoutError:
                raise OtcmInteractionError(f"Manual login confirmation timed out after {login_wait_timeout}s.")
        else:
            self._logger.info("Login status OK: Dashboard accessed without redirection to login page.")