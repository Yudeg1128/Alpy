import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Set

import pandas as pd
from playwright.async_api import async_playwright, Page, Browser, TimeoutError as PlaywrightTimeoutError

from langchain_core.tools import BaseTool, ToolException
from langchain_core.callbacks.manager import AsyncCallbackManagerForToolRun
from pydantic import BaseModel, Field, model_validator, PrivateAttr

# Module logger
logger = logging.getLogger(__name__)

class OTCMNScraperToolError(ToolException):
    """Custom exception for the OTCMNScraperTool."""
    pass

class OTCMNScraperInput(BaseModel):
    output_directory: str = Field(description="Base directory to save scraped Parquet files and downloaded documents. Will be created if it doesn't exist.")
    max_listing_pages_per_board: Optional[int] = Field(default=None, description="Maximum number of listing pages to scrape per board category (e.g., Primary-A, Secondary-C). If None or 0, scrapes all available pages for each category. Default: None (all pages).")
    max_securities_to_process: Optional[int] = Field(default=None, description="Maximum number of unique securities to process from the listings for detailed scraping. If None or 0, processes all found securities. Default: None (all securities).")
    cdp_endpoint_url: Optional[str] = Field(default=None, description="Optional: CDP endpoint URL for Playwright to connect to an existing browser session. If None, the tool will launch its own browser instance.")


class OTCMNScraperTool(BaseTool, BaseModel):
    name: str = "OTCMN_Scraper"
    description: str = (
        "Automates the scraping of financial securities data from otc.mn. "
        "It systematically navigates listing pages, extracts summary tables, then visits individual security detail pages "
        "to extract detailed information tables and download associated documents (e.g., PDFs, DOCs).\n\n"
        "**IMPORTANT: MANUAL LOGIN REQUIRED!**\n"
        "When this tool starts, it will open a web browser. You MUST manually log into otc.mn in that browser. "
        "After successful login, you need to return to the console where Alpy is running and press Enter to allow the tool to proceed.\n\n"
        "**Tool `action_input` Structure (JSON Object):**\n"
        "The `action_input` must be a JSON object with the following fields:\n"
        "1. `output_directory` (string, required): The local filesystem path where all scraped data (Parquet tables and downloaded documents) will be stored. The tool will create this directory if it doesn't exist. Example: \"/mnt/data/otc_mn_data\".\n"
        "2. `max_listing_pages_per_board` (integer, optional): Limits how many pages of listings are scraped for each board category (e.g., Primary-Board1, Primary-Board2, etc.). If not provided, 0, or null, the tool attempts to scrape all available pages for each category. Example: `5` (scrape up to 5 pages per board).\n"
        "3. `max_securities_to_process` (integer, optional): Limits the total number of unique securities for which detailed information and documents are fetched. Securities are collected from the listing pages first. If not provided, 0, or null, the tool attempts to process all unique securities found. Example: `10` (process details for up to 10 securities).\n\n"
        "**Operational Flow:**\n"
        "1.  Launches a Playwright-controlled browser and navigates to `https://otc.mn/login`.\n"
        "2.  Prompts the human user (via console `input()`) to manually log in.\n"
        "3.  Once login is confirmed by the user (pressing Enter in console), the tool proceeds.\n"
        "4.  Iterates through predefined listing pages (combinations of 'Primary'/'Secondary' tabs and 'Board 1'/'Board 2'/'Board 3' sub-filters).\n"
        "    *   For each listing category (e.g., Primary-Board1):\n"
        "        *   Scrapes the main table of securities. Extracts all columns and rows.\n"
        "        *   Handles pagination, clicking 'Next Page' up to `max_listing_pages_per_board` or until no more pages.\n"
        "**Example `action_input`:**\n"
        "```json\n"
        "{\n"
        "  \"output_directory\": \"/tmp/alpy_otc_scrape\",\n"
        "  \"max_listing_pages_per_board\": 2,\n"
        "  \"max_securities_to_process\": 3,\n"
        "  \"cdp_endpoint_url\": \"ws://127.0.0.1:9222/devtools/browser/xxxx\"\n"
        "}\n"
        "```\n\n"
        "If `cdp_endpoint_url` is not provided, the tool will launch its own browser and prompt for manual login.\n\n"
        "**Example Observation (Output from tool):**\n"
        "```json\n"
        "{\n"
        "  \"status\": \"success\",\n"
        "  \"output_directory\": \"/tmp/alpy_otc_scrape\",\n"
        "  \"listing_tables_scraped\": 12,\n"
        "  \"security_details_scraped\": 3,\n"
        "  \"documents_downloaded\": 7,\n"
        "  \"errors\": []\n"
        "}\n"
        "```\n\n"
        "**Notes & Cautions:**\n"
        "*   Successful manual login by the human user is critical for the tool's operation unless using a pre-authenticated CDP session.\n"
        "*   The scraping process can be time-consuming, especially if not limited by `max_listing_pages_per_board` or `max_securities_to_process`.\n"
        "*   Ensure the specified `output_directory` is writable by the Alpy application.\n"
        "*   Requires Playwright and its browser drivers to be installed in the environment where Alpy runs (e.g., `playwright install`)."
    )
    args_schema: Type[BaseModel] = OTCMNScraperInput
    return_direct: bool = False
    handle_tool_error: bool = True

    playwright_channel: str = Field(default="chromium", description="Browser channel: 'chromium', 'firefox', 'webkit'.")
    login_prompt_timeout: float = Field(default=300.0, description="Seconds to wait for user to confirm manual login via console input.")
    page_timeout: float = Field(default=60000, description="Default page/navigation/action timeout in milliseconds for Playwright.")
    # brave-browser --remote-debugging-port=9222

    _logger_instance: logging.Logger = PrivateAttr(default=None)

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def _tool_post_init_validator(self) -> 'OTCMNScraperTool':
        if self._logger_instance is None:
            self._logger_instance = logger.getChild(f"{self.name}.{id(self)}")
            if not self._logger_instance.hasHandlers() and not logging.getLogger().hasHandlers():
                _handler = logging.StreamHandler(sys.stdout)
                _formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s')
                _handler.setFormatter(_formatter)
                self._logger_instance.addHandler(_handler)
                self._logger_instance.propagate = False 
                if not self._logger_instance.level: 
                    self._logger_instance.setLevel(logging.INFO) 

        self._logger_instance.info(f"Initialized {self.name} for Playwright channel '{self.playwright_channel}'. Manual login will be required.")
        self._logger_instance.debug(f"Config: login_prompt_timeout={self.login_prompt_timeout}s, page_timeout={self.page_timeout}ms")
        return self

    async def _arun(
        self,
        output_directory: str,
        cdp_endpoint_url: Optional[str] = None, # Added CDP endpoint parameter
        max_listing_pages_per_board: Optional[int] = None,
        max_securities_to_process: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any
    ) -> str:
        self._logger_instance.info(f"Starting OTCMN scraper. Output directory: {output_directory}")
        self._logger_instance.info(f"--- _arun CALLED ---") # New log
        self._logger_instance.info(f"Received cdp_endpoint_url: {cdp_endpoint_url}") # New log
        self._logger_instance.info(f"Tool's configured playwright_channel: {self.playwright_channel}") # New log

        if cdp_endpoint_url:
            self._logger_instance.info(f"CDP endpoint provided: {cdp_endpoint_url}")
        
        base_output_path = Path(output_directory)
        tables_path = base_output_path / "tables"
        documents_path = base_output_path / "documents"
        
        try:
            base_output_path.mkdir(parents=True, exist_ok=True)
            tables_path.mkdir(parents=True, exist_ok=True)
            documents_path.mkdir(parents=True, exist_ok=True)
            self._logger_instance.debug(f"Ensured output directories exist: {tables_path}, {documents_path}")
        except Exception as e:
            self._logger_instance.error(f"Failed to create output directories: {e}")
            raise OTCMNScraperToolError(f"Failed to create output directories: {e}")

        summary_log = {
            "status": "success",
            "output_directory": str(base_output_path.resolve()),
            "listing_tables_scraped": 0,
            "security_details_scraped": 0,
            "documents_downloaded": 0,
            "errors": []
        }
        all_security_detail_urls: Set[str] = set()
        
        browser_connected_via_cdp = False # Flag to manage browser closing behavior

        async with async_playwright() as p_context:
            browser: Optional[Browser] = None
            context: Optional[PlaywrightContext] = None
            page: Optional[Page] = None
            try:
                if cdp_endpoint_url:
                    self._logger_instance.info(f"Attempting to connect to existing browser via CDP: {cdp_endpoint_url}")
                    try:
                        browser = await p_context.chromium.connect_over_cdp(cdp_endpoint_url, timeout=self.page_timeout)
                        browser_connected_via_cdp = True
                        self._logger_instance.info(f"Successfully connected to browser via CDP. Assuming manual login is complete.")
                        
                        if not browser.contexts:
                            self._logger_instance.error("No browser contexts found after CDP connect. Cannot proceed.")
                            raise OTCMNScraperToolError("No browser contexts found via CDP. Ensure the target browser has at least one window/tab open to otc.mn or a related page.")
                        context = browser.contexts[0] 
                        
                        # Try to find an existing otc.mn page, otherwise create new or use first available.
                        otc_page_found = False
                        for p_item in context.pages:
                            if "otc.mn" in p_item.url:
                                page = p_item
                                await page.bring_to_front()
                                self._logger_instance.info(f"Using existing otc.mn page from CDP browser: {page.url}")
                                otc_page_found = True
                                break
                        if not page: # If no otc.mn page, use the first one or create one
                             if context.pages:
                                page = context.pages[0]
                                await page.bring_to_front()
                                self._logger_instance.info(f"Using first available existing page from CDP browser: {page.url}. Will navigate to otc.mn.")
                             else:
                                self._logger_instance.info("No pages in context, creating a new one.")
                                page = await context.new_page() 
                        
                        context.set_default_timeout(self.page_timeout)
                        context.set_default_navigation_timeout(self.page_timeout)

                        self._logger_instance.info("Checking session by navigating to otc.mn dashboard...")
                        await page.goto("https://otc.mn/dashboard", wait_until="domcontentloaded")
                        if "/login" in page.url.lower() or "login" in (await page.title()).lower():
                            self._logger_instance.error("CDP connection successful, but not logged into otc.mn or session invalid. Please log in via the externally launched browser and re-run.")
                            raise OTCMNScraperToolError("Not logged in or session invalid in CDP browser.")
                        self._logger_instance.info(f"Session confirmed via CDP. Current page: {page.url}")

                    except Exception as e_cdp:
                        self._logger_instance.error(f"Failed to connect or use CDP endpoint '{cdp_endpoint_url}': {e_cdp}", exc_info=True)
                        summary_log["status"] = "error"
                        summary_log["errors"].append(f"CDP connection/usage failed: {str(e_cdp)}")
                        # browser might be None or partially connected, let finally handle it.
                        raise OTCMNScraperToolError(f"CDP connection/usage failed: {str(e_cdp)}") # Re-raise to stop further execution
                else:
                    # --- Original Playwright launch and manual login prompt logic ---
                    self._logger_instance.info(f"No CDP endpoint provided. Launching new Playwright browser (channel: {self.playwright_channel})...")
                    browser = await getattr(p_context, self.playwright_channel).launch(headless=False)
                    context = await browser.new_context(
                        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
                        viewport={'width': 1920, 'height': 1080},
                        ignore_https_errors=True
                    )
                    context.set_default_timeout(self.page_timeout)
                    context.set_default_navigation_timeout(self.page_timeout)
                    page = await context.new_page()
                    self._logger_instance.info("Browser launched. Navigating to login page...")

                    await page.goto("https://otc.mn/login", wait_until="domcontentloaded")
                    self._logger_instance.info(
                        f"Browser directed to https://otc.mn/login. Please log in manually in the Playwright browser window. "
                        f"After successful login, press Enter in THIS CONSOLE to continue. Timeout: {self.login_prompt_timeout} seconds."
                    )
                    try:
                        await asyncio.wait_for(
                            asyncio.to_thread(input, "Press Enter in this console after logging in: "),
                            timeout=self.login_prompt_timeout
                        )
                    except asyncio.TimeoutError:
                        self._logger_instance.error("Timeout waiting for manual login confirmation.")
                        summary_log["status"] = "error"
                        summary_log["errors"].append("Manual login confirmation timed out.")
                        raise OTCMNScraperToolError("Manual login confirmation timed out.")

                    if "/login" in page.url.lower() or "login" in (await page.title()).lower():
                        self._logger_instance.warning(f"Login may not have been successful. Current URL: {page.url}, Title: {await page.title()}")
                        # Depending on policy, you might raise an error here or allow continuation.
                        # For now, raising error if manual login path seems to have failed.
                        raise OTCMNScraperToolError(f"Manual login attempt failed or not confirmed. URL: {page.url}")
                    else:
                        self._logger_instance.info(f"Login presumed successful. Current URL: {page.url}, Title: {await page.title()}")
                # --- End of browser/login setup ---

                # Ensure page is not None before proceeding
                if page is None:
                    self._logger_instance.error("Page object is None, cannot proceed with scraping.")
                    raise OTCMNScraperToolError("Failed to initialize page for scraping.")

                # --- Scraping Logic (Common for both paths if login/CDP is successful) ---
                tabs = ["primary", "secondary"]
                boards = ["1", "2", "3"] 

                for tab_name in tabs:
                    for board_id in boards:
                        # The listing_url is now more for logging/identification
                        listing_url_stub = f"otc.mn/securities (Tab: {tab_name}, Board: {board_id})"
                        self._logger_instance.info(f"Processing listing: {listing_url_stub}")
                        try:
                            extracted_urls, num_tables = await self._scrape_listing_page_otc(
                                page, 
                                listing_url_stub, # For logging
                                tables_path, 
                                tab_name, 
                                board_id, 
                                max_listing_pages_per_board or 0 
                            )
                            all_security_detail_urls.update(extracted_urls)
                            summary_log["listing_tables_scraped"] += num_tables
                        except Exception as e:
                            self._logger_instance.error(f"Error scraping listing {listing_url_stub}: {e}", exc_info=True)
                            summary_log["errors"].append(f"Failed listing: {listing_url_stub} - {str(e)}")
                
                self._logger_instance.info(f"Found {len(all_security_detail_urls)} unique security detail URLs to process.")

                processed_securities_count = 0
                detail_urls_to_process = list(all_security_detail_urls)
                if max_securities_to_process and max_securities_to_process > 0:
                     detail_urls_to_process = detail_urls_to_process[:max_securities_to_process]
                
                self._logger_instance.info(f"Will process up to {len(detail_urls_to_process)} security detail pages.")

                for detail_url_path in detail_urls_to_process:
                    full_detail_url = f"https://otc.mn{detail_url_path}" 
                    self._logger_instance.info(f"Processing security detail page: {full_detail_url}")
                    try:
                        security_id_match = re.search(r"id=([a-f0-9-]+)", detail_url_path, re.IGNORECASE)
                        security_id = security_id_match.group(1) if security_id_match else "unknown_id_" + str(processed_securities_count).zfill(3)
                        
                        num_docs = await self._scrape_security_detail_page_otc(
                            page, full_detail_url, tables_path, documents_path, security_id
                        )
                        summary_log["security_details_scraped"] += 1 
                        summary_log["documents_downloaded"] += num_docs
                        processed_securities_count += 1
                    except Exception as e:
                        self._logger_instance.error(f"Error scraping detail page {full_detail_url}: {e}", exc_info=True)
                        summary_log["errors"].append(f"Failed detail: {full_detail_url} - {str(e)}")
                
            except PlaywrightTimeoutError as pte:
                self._logger_instance.error(f"Playwright operation timed out: {pte}", exc_info=True)
                summary_log["status"] = "error"
                summary_log["errors"].append(f"Playwright timeout: {str(pte)}")
            except OTCMNScraperToolError: # Re-raise tool-specific errors from login/CDP blocks
                raise
            except Exception as e:
                self._logger_instance.error(f"An unexpected error occurred in _arun: {e}", exc_info=True)
                summary_log["status"] = "error"
                summary_log["errors"].append(f"Unexpected error: {str(e)}")
            finally:
                if browser and browser.is_connected():
                    if browser_connected_via_cdp:
                        self._logger_instance.info("Disconnecting from CDP browser (user's browser will remain open)...")
                        await browser.close() # Disconnects, doesn't close user's browser
                        self._logger_instance.info("Disconnected from CDP browser.")
                    else:
                        self._logger_instance.info("Closing Playwright-launched browser...")
                        await browser.close()
                        self._logger_instance.info("Playwright-launched browser closed.")
        
        self._logger_instance.info(f"Scraping process finished. Summary: {json.dumps(summary_log, indent=2)}")
        return json.dumps(summary_log)

    async def _scrape_listing_page_otc(
        self, page: Page, list_url_stub_for_logging: str,
        tables_path: Path, 
        tab_name: str, board_id: str, max_pages: int 
    ) -> (List[str], int):
        detail_urls: Set[str] = set()
        page_num = 1
        tables_saved_count = 0

        self._logger_instance.info(f"[LISTING PAGE START] Tab: '{tab_name}', Board ID: '{board_id}'. Target: {list_url_stub_for_logging}")
        
        self._logger_instance.debug(f"[NAV BASE] Navigating to base securities page: https://otc.mn/securities")
        try:
            await page.goto("https://otc.mn/securities", wait_until="domcontentloaded", timeout=self.page_timeout)
            self._logger_instance.debug(f"[NAV BASE SUCCESS] Current URL: {page.url}")
            await page.wait_for_timeout(3000) 
        except Exception as e_nav_base:
            self._logger_instance.error(f"[NAV BASE FAIL] Failed to navigate to base /securities: {e_nav_base}")
            return list(detail_urls), tables_saved_count

        capitalized_tab_name = tab_name.capitalize()
        tab_selector_by_text = f"//div[@role='tablist']//div[@role='tab' and normalize-space(.)='{capitalized_tab_name}']"
        
        self._logger_instance.debug(f"[CLICK TAB] Attempting to click tab: '{capitalized_tab_name}' using XPath: {tab_selector_by_text}")
        try:
            tab_element = page.locator(f"xpath={tab_selector_by_text}")
            await tab_element.wait_for(state="visible", timeout=self.page_timeout / 2)
            await tab_element.click(timeout=self.page_timeout / 3)
            self._logger_instance.info(f"[CLICK TAB SUCCESS] Clicked tab: '{capitalized_tab_name}'. Current URL: {page.url}")
            
            active_tab_pane_selector_for_confirm = f"div.ant-tabs-tabpane-active[id*='{tab_name}']"
            await page.wait_for_selector(active_tab_pane_selector_for_confirm, state="visible", timeout=self.page_timeout / 2)
            self._logger_instance.debug(f"[CLICK TAB CONFIRM] Active tab pane for '{tab_name}' confirmed by selector: {active_tab_pane_selector_for_confirm}.")
            await page.wait_for_timeout(3000) 
        except Exception as e_tab_click:
            self._logger_instance.error(f"[CLICK TAB FAIL] Failed to click/confirm tab '{capitalized_tab_name}': {e_tab_click}. Current URL: {page.url}")
            return list(detail_urls), tables_saved_count 

        board_map = {"1": "A", "2": "B", "3": "C", "all": "All Boards"}
        board_filter_text = board_map.get(board_id)

        if board_filter_text:
            board_filter_selector = f"//ul[contains(@class, 'sub-menu') and contains(@class, 'custom-menu')]//li[.//span[@class='ant-menu-title-content' and normalize-space(text())='{board_filter_text}']]"
            self._logger_instance.debug(f"[CLICK BOARD] Attempting to click board filter: '{board_filter_text}' using XPath: {board_filter_selector}")
            try:
                board_element = page.locator(f"xpath={board_filter_selector}")
                await board_element.wait_for(state="visible", timeout=self.page_timeout / 2)
                await board_element.click(timeout=self.page_timeout / 3)
                self._logger_instance.info(f"[CLICK BOARD SUCCESS] Clicked board filter: '{board_filter_text}'. Current URL: {page.url}")
                
                expected_url_pattern_part_board = f"board={board_id}"
                expected_url_pattern_part_tab = f"tab={tab_name}"
                try:
                    await page.wait_for_function(
                        f"document.location.href.includes('{expected_url_pattern_part_tab}') && document.location.href.includes('{expected_url_pattern_part_board}')",
                        timeout=self.page_timeout / 4 
                    )
                    self._logger_instance.debug(f"[CLICK BOARD URL CHECK] URL contains '{expected_url_pattern_part_tab}' and '{expected_url_pattern_part_board}'. Current URL: {page.url}")
                except PlaywrightTimeoutError:
                     self._logger_instance.warning(f"[CLICK BOARD URL CHECK FAIL] URL did not update with tab & board quickly. Current URL: {page.url}. Relying on content.")
                
                self._logger_instance.debug(f"[POST BOARD CLICK] Waiting for potential data load. Initial 5s hard pause then networkidle.")
                await page.wait_for_timeout(5000) 
                
                self._logger_instance.debug(f"[POST BOARD CLICK] Now waiting for network idle.")
                await page.wait_for_load_state("networkidle", timeout=self.page_timeout) 
                self._logger_instance.debug(f"[POST BOARD CLICK] Network idle confirmed after extended wait. Current URL: {page.url}")

            except Exception as e_board_click:
                self._logger_instance.error(f"[CLICK BOARD FAIL] Failed to click board filter '{board_filter_text}': {e_board_click}. Current URL: {page.url}")
                return list(detail_urls), tables_saved_count
        else:
            self._logger_instance.warning(f"[CLICK BOARD SKIP] No mapping for board_id '{board_id}'. Assuming 'All Boards' or current state. Waiting for network idle.")
            await page.wait_for_load_state("networkidle", timeout=self.page_timeout)
            self._logger_instance.debug(f"[CLICK BOARD SKIP NET IDLE] Network idle confirmed. Current URL: {page.url}")
        
        while True:
            if max_pages > 0 and page_num > max_pages:
                self._logger_instance.info(f"[PAGINATION LIMIT] Reached max_listing_pages ({max_pages}) for Tab: {tab_name}, Board: {board_id}.")
                break
            
            self._logger_instance.info(f"[PAGINATION START PAGE {page_num}] Tab: {tab_name}, Board: {board_id}. Current URL: {page.url}")
            
            active_tab_pane_locator_str = f"div.ant-tabs-tabpane-active[id*='{tab_name}']"
            try:
                await page.locator(active_tab_pane_locator_str).wait_for(state="visible", timeout=self.page_timeout / 2)
                self._logger_instance.debug(f"[WAIT ACTIVE TABPANE {page_num}] Active tab pane for '{tab_name}' is visible.")
            except PlaywrightTimeoutError:
                self._logger_instance.error(f"[WAIT ACTIVE TABPANE FAIL {page_num}] Timeout waiting for active tab pane for '{tab_name}'.")
                break 

            # More specific path to the tbody, reflecting the Ant Design structure:
            table_body_selector_primary = (
                f"{active_tab_pane_locator_str} "
                "div.ant-table-wrapper " 
                "div.ant-spin-container " 
                "div.ant-table-container "
                "div.ant-table-content "
                "table > " 
                "tbody.ant-table-tbody"
            )
            table_body_selector_fallback = ( # Omitting spin-container
                 f"{active_tab_pane_locator_str} "
                "div.ant-table-wrapper " 
                "div.ant-table-container "
                "div.ant-table-content "
                "table > " 
                "tbody.ant-table-tbody"
            )
            
            actual_table_body_handle = None
            used_selector_for_body = ""

            self._logger_instance.debug(f"[WAIT TABLE BODY {page_num}] Waiting for table body using primary selector: {table_body_selector_primary}")
            try:
                table_body_locator = page.locator(table_body_selector_primary)
                await table_body_locator.first.wait_for(state="attached", timeout=self.page_timeout / 2)
                count = await table_body_locator.count()
                self._logger_instance.debug(f"Found {count} element(s) for primary table body selector.")
                if count == 0: raise PlaywrightTimeoutError("Primary table body selector found 0 elements.")
                await table_body_locator.first.wait_for(state="visible", timeout=self.page_timeout)
                actual_table_body_handle = table_body_locator.first
                used_selector_for_body = "primary"
            except PlaywrightTimeoutError:
                self._logger_instance.warning(f"[WAIT TABLE BODY {page_num}] Timeout with primary selector. Trying fallback: {table_body_selector_fallback}")
                try:
                    table_body_locator = page.locator(table_body_selector_fallback)
                    await table_body_locator.first.wait_for(state="attached", timeout=self.page_timeout / 2)
                    count = await table_body_locator.count()
                    self._logger_instance.debug(f"Found {count} element(s) for fallback table body selector.")
                    if count == 0: raise PlaywrightTimeoutError("Fallback table body selector found 0 elements.")
                    await table_body_locator.first.wait_for(state="visible", timeout=self.page_timeout)
                    actual_table_body_handle = table_body_locator.first
                    used_selector_for_body = "fallback"
                except PlaywrightTimeoutError:
                    self._logger_instance.error(f"[WAIT TABLE BODY FAIL {page_num}] Timeout waiting for table body (both selectors) for Tab: {tab_name}, Board: {board_id}. Page URL: {page.url}")
                    # --- Start Diagnostic Dump ---
                    self._logger_instance.info(f"--- DIAGNOSTIC DUMP for {page.url} (Tab: {tab_name}, Board: {board_id}, Page: {page_num}) ---")
                    screenshot_path = tables_path.parent / f"debug_screenshot_tab_{tab_name}_board_{board_id}_page_{page_num}.png"
                    try:
                        await page.screenshot(path=screenshot_path, full_page=True)
                        self._logger_instance.info(f"Saved debug screenshot: {screenshot_path}")
                    except Exception as e_ss: self._logger_instance.error(f"Failed to save debug screenshot: {e_ss}")
                    html_path = tables_path.parent / f"debug_page_html_tab_{tab_name}_board_{board_id}_page_{page_num}.html"
                    try:
                        full_html = await page.content()
                        with open(html_path, "w", encoding="utf-8") as f: f.write(full_html)
                        self._logger_instance.info(f"Saved debug HTML: {html_path}")
                    except Exception as e_html: self._logger_instance.error(f"Failed to save debug HTML: {e_html}")
                    self._logger_instance.info("Checking for 'div.ant-tabs-tabpane-active' elements...")
                    active_panes = await page.query_selector_all("div.ant-tabs-tabpane-active")
                    self._logger_instance.info(f"Found {len(active_panes)} element(s) matching 'div.ant-tabs-tabpane-active'.")
                    for i, pane_handle_diag in enumerate(active_panes): # Renamed to avoid conflict
                        pane_id = await pane_handle_diag.get_attribute("id")
                        self._logger_instance.info(f"  Active Pane {i} ID: {pane_id}")
                        if tab_name in (pane_id or ""):
                            self._logger_instance.info(f"    Pane ID {pane_id} matches current tab_name '{tab_name}'. Checking for table body with simple query...")
                            tb_handle_in_pane = await pane_handle_diag.query_selector("div.ant-table-tbody") # Simple query for diagnostics
                            if tb_handle_in_pane:
                                self._logger_instance.info(f"    Found 'div.ant-table-tbody' (simple query) within pane ID {pane_id}.")
                                row_count_in_tbody = await tb_handle_in_pane.eval_on_selector_all("tr.ant-table-row", "nodes => nodes.length")
                                self._logger_instance.info(f"    Count of 'tr.ant-table-row' in this tbody (simple query): {row_count_in_tbody}")
                            else: self._logger_instance.warning(f"    'div.ant-table-tbody' (simple query) NOT FOUND within pane ID {pane_id}.")
                        else: self._logger_instance.info(f"    Pane (ID: {pane_id}) does NOT match tab_name '{tab_name}'.")
                    self._logger_instance.info(f"--- END DIAGNOSTIC DUMP ---")
                    # --- End Diagnostic Dump ---
                    break # Exit pagination loop

            self._logger_instance.debug(f"[WAIT TABLE BODY SUCCESS {page_num}] Table body container is visible using {used_selector_for_body} selector.")
            
            populated_row_locator = actual_table_body_handle.locator("tr.ant-table-row:has(td)")
            
            self._logger_instance.debug(f"[WAIT POPULATED ROW {page_num}] Waiting for populated data row within identified table body.")
            try:
                await populated_row_locator.first.wait_for(state="visible", timeout=self.page_timeout)
                self._logger_instance.info(f"[WAIT POPULATED ROW SUCCESS {page_num}] At least one populated data row found.")
            except PlaywrightTimeoutError:
                self._logger_instance.warning(f"[WAIT POPULATED ROW FAIL {page_num}] Timeout waiting for populated data row. Table might be empty/failed to load. Page URL: {page.url}")
                no_data_placeholder_selector = ".ant-table-placeholder .ant-empty-description"
                no_data_element = await actual_table_body_handle.query_selector(no_data_placeholder_selector)
                if no_data_element and (await no_data_element.is_visible()):
                    no_data_text = await no_data_element.text_content()
                    self._logger_instance.info(f"[TABLE EMPTY {page_num}] Table shows 'No data' placeholder (text: '{no_data_text}'). Ending pagination.")
                else:
                    self._logger_instance.warning(f"[TABLE EMPTY UNKNOWN {page_num}] No populated rows and no clear 'No data' placeholder. Ending pagination.")
                break 
            
            rows_handles = await populated_row_locator.all()
            if not rows_handles:
                self._logger_instance.info(f"[NO ROWS {page_num}] No actual data rows found using populated_row_locator. Ending pagination.")
                break

            self._logger_instance.debug(f"[EXTRACT ROWS {page_num}] Found {len(rows_handles)} data rows to extract.")
            page_data = []
            for i, row_handle_iter in enumerate(rows_handles): # Renamed to avoid conflict
                cell_locators = row_handle_iter.locator("td")
                cell_texts = await cell_locators.all_inner_texts() # Gets texts from all matching 'td'
                row_data = [text.strip() for text in cell_texts]
                page_data.append(row_data)
                self._logger_instance.debug(f"[EXTRACT ROW {page_num}-{i+1}] Data: {row_data}")
                
                link_locator = row_handle_iter.locator('a[href^="/securities/detail?id="]')
                if await link_locator.count() > 0: # Check if the locator finds at least one element
                    href = await link_locator.first.get_attribute("href")
                    if href:
                        detail_urls.add(href)
                        self._logger_instance.debug(f"[EXTRACT URL {page_num}-{i+1}] Found detail URL: {href}")
            
            if page_data:
                try:
                    columns = []
                    header_th_elements_loc = page.locator(f"{active_tab_pane_locator_str} div.ant-table-thead tr th") # Target all th in thead row
                    all_th_handles = await header_th_elements_loc.all()
                    for idx, th_handle in enumerate(all_th_handles):
                        # Try to get text from .ant-table-column-title first if it exists
                        title_span = await th_handle.query_selector(".ant-table-column-title")
                        text_content = ""
                        if title_span:
                            text_content = (await title_span.text_content() or "").strip()
                        if not text_content: # Fallback to th's direct text if title_span is empty or not found
                            text_content = (await th_handle.text_content() or "").strip()
                        
                        # Only add non-empty headers, or provide a placeholder
                        if text_content:
                            columns.append(text_content)
                        else:
                            # This column might be for checkboxes or actions, assign a generic name or skip
                            # For now, let's assign a generic one if it's likely a data column based on data length later
                            # Or, if we know the number of data columns, we can be smarter.
                            # For simplicity here, we'll let the mismatch logic handle it if some headers are truly blank.
                            columns.append(f"HeaderCol_Empty_{idx}") # Placeholder for truly empty headers
                    
                    columns = [col for col in columns if col] # Final cleanup
                    self._logger_instance.debug(f"[SAVE TABLE HEADERS {page_num}] Tentative Headers: {columns}")
                except Exception as he:
                    self._logger_instance.warning(f"[SAVE TABLE HEADERS FAIL {page_num}] (Tab: {tab_name}, Board: {board_id}): {he}. Using generic.")
                    columns = [f"Column_{i_col+1}" for i_col in range(len(page_data[0]))] if page_data else [] # Corrected generic columns

                if columns and len(columns) == len(page_data[0]):
                    df = pd.DataFrame(page_data, columns=columns)
                else:
                    self._logger_instance.warning(f"[SAVE TABLE COL MISMATCH {page_num}] Header count ({len(columns)}) vs data ({len(page_data[0]) if page_data else 0}). Saving with pandas-inferred/generic headers.")
                    df = pd.DataFrame(page_data, columns=columns if columns and len(columns) == len(page_data[0]) else None) # Use columns if valid, else None
                
                parquet_file = tables_path / f"listings_tab_{tab_name}_board_{board_id}_page_{page_num}.parquet"
                df.to_parquet(parquet_file, index=False)
                tables_saved_count += 1
                self._logger_instance.info(f"[SAVE TABLE SUCCESS {page_num}] Saved: {parquet_file} ({len(df)} rows)")
            else:
                self._logger_instance.info(f"[SAVE TABLE SKIP {page_num}] No data extracted from rows to save.")

            next_button_selector = "ul.ant-pagination li.ant-pagination-next:not(.ant-pagination-disabled) button"
            self._logger_instance.debug(f"[PAGINATION NEXT CHECK {page_num}] Looking for next button: {next_button_selector}")
            next_button_handle = await page.query_selector(next_button_selector) # Renamed
            
            if next_button_handle: # Check if handle is not None
                self._logger_instance.debug(f"[PAGINATION NEXT CLICK {page_num}] Found 'Next Page' button. Clicking...")
                await next_button_handle.click(timeout=self.page_timeout / 3)
                self._logger_instance.info(f"[PAGINATION NEXT CLICK SUCCESS {page_num}] Clicked 'Next Page'.")
                page_num += 1
                try:
                    self._logger_instance.debug(f"[PAGINATION WAIT NEW PAGE {page_num}] Waiting for content of page {page_num}...")
                    # Use the more specific selector for the new page's table body
                    table_body_locator_for_next_page = page.locator(table_body_selector_primary if used_selector_for_body == "primary" else table_body_selector_fallback)
                    await table_body_locator_for_next_page.locator("tr.ant-table-row:has(td)").first.wait_for(state="visible", timeout=self.page_timeout)
                    self._logger_instance.debug(f"[PAGINATION WAIT NEW PAGE SUCCESS {page_num}] Content for page {page_num} loaded.")
                except PlaywrightTimeoutError:
                    self._logger_instance.warning(f"[PAGINATION WAIT NEW PAGE FAIL {page_num}] Timeout for page {page_num} content (Tab: {tab_name}, Board: {board_id}). Assuming end of pages.")
                    break
            else:
                self._logger_instance.info(f"[PAGINATION END {page_num}] No 'Next Page' button. End of listings for (Tab: {tab_name}, Board: {board_id}).")
                break
        
        self._logger_instance.info(f"[LISTING PAGE END] Finished Tab: '{tab_name}', Board ID: '{board_id}'. Found {len(detail_urls)} URLs, saved {tables_saved_count} tables.")
        return list(detail_urls), tables_saved_count

    async def _extract_table_data_from_page(
        self,
        page: Page,
        # table_container_xpath: str, # We will use more specific locators now
        section_locator_strategy: Dict[str, str], # e.g., {"xpath": "//div[header text check]"}
        table_name: str,
        security_id: str,
        tables_path: Path,
        is_key_value_style: bool = False,
        # New parameter to define how to find the actual table(s) within the section
        # 'columns': implies looking for md:w-1/2 divs, then a table in each
        # 'direct_wrapper': implies looking for .ant-table-wrapper directly in the section
        table_layout_type: str = 'direct_wrapper' # 'columns' or 'direct_wrapper'
    ):
        self._logger_instance.info(f"[EXTRACT TABLE START] Security ID: {security_id}, Table Name: '{table_name}', Layout: '{table_layout_type}'.")

        # 1. Locate the main section container
        section_container_loc = None
        if "xpath" in section_locator_strategy:
            xpath_str = section_locator_strategy["xpath"]
            self._logger_instance.debug(f"Attempting to locate section container for '{table_name}' using XPath: {xpath_str}")
            section_container_loc = page.locator(f"xpath={xpath_str}")
        # Add other strategies like "css" if needed
        else:
            self._logger_instance.error(f"No valid locator strategy provided for section '{table_name}'.")
            return

        try:
            await section_container_loc.first.wait_for(state="visible", timeout=self.page_timeout / 2)
            self._logger_instance.info(f"Section container for '{table_name}' is visible.")
        except PlaywrightTimeoutError:
            self._logger_instance.error(f"Timeout waiting for section container for '{table_name}' (ID: {security_id}). Strategy: {section_locator_strategy}")
            return
        except Exception as e_sec_vis:
            self._logger_instance.error(f"Error ensuring section visibility for '{table_name}': {e_sec_vis}")
            return

        # 2. Identify the "table contexts" based on layout_type
        table_contexts = []
        if table_layout_type == 'columns':
            # This layout is specific to "Security Details" type where content is in side-by-side columns
            # The section_container_loc should point to the div.rounded-lg... holding the flex-row
            row_layout_div_locator = section_container_loc.locator("div.flex.flex-col.md\\:flex-row") # Escaped : for CSS if needed
            try:
                await row_layout_div_locator.first.wait_for(state="visible", timeout=self.page_timeout / 3)
                # Get direct children that are columns: div.w-full.md:w-1/2
                column_div_locators = row_layout_div_locator.locator("> div.w-full.md\\:w-1\\/2") # Direct children, escaped / and :
                
                count = await column_div_locators.count()
                if count == 0:
                    self._logger_instance.warning(f"Layout type is 'columns' for '{table_name}', but no 'md:w-1/2' column divs found within its row layout.")
                else:
                    self._logger_instance.debug(f"Found {count} column(s) for '{table_name}'.")
                    all_cols = await column_div_locators.all()
                    for col_loc in all_cols:
                        # Each column div is a context that should contain one ant-table-wrapper
                        table_wrapper_in_col = col_loc.locator("div.ant-table-wrapper")
                        if await table_wrapper_in_col.count() > 0:
                            table_contexts.append(table_wrapper_in_col.first) # Add the Locator
                        else:
                            self._logger_instance.warning(f"Column found for '{table_name}', but no 'div.ant-table-wrapper' within it.")
            except PlaywrightTimeoutError:
                self._logger_instance.warning(f"Timeout finding column layout structure for '{table_name}' of type 'columns'.")
            except Exception as e_cols:
                self._logger_instance.error(f"Error processing 'columns' layout for '{table_name}': {e_cols}")

        elif table_layout_type == 'direct_wrapper':
            # For "File List", "Transaction History", "Underwriter Info" (single table directly in section wrapper)
            # section_container_loc points to the div.ant-col... or div.rounded-lg...
            wrapper_locator = section_container_loc.locator("div.ant-table-wrapper")
            try:
                await wrapper_locator.first.wait_for(state="attached", timeout=self.page_timeout / 3)
                count = await wrapper_locator.count()
                if count > 0:
                    self._logger_instance.debug(f"Found {count} 'div.ant-table-wrapper'(s) for '{table_name}' with 'direct_wrapper' layout. Processing first/primary.")
                    # If multiple, this assumes the first one is the target or that the section_locator_strategy was specific enough.
                    # For truly multiple distinct tables not in columns, the strategy might need adjustment or multiple calls.
                    table_contexts.append(wrapper_locator.first) # Add the Locator
                else:
                    self._logger_instance.warning(f"No 'div.ant-table-wrapper' found for '{table_name}' with 'direct_wrapper' layout.")
            except PlaywrightTimeoutError:
                self._logger_instance.warning(f"Timeout finding 'div.ant-table-wrapper' for '{table_name}' of type 'direct_wrapper'.")
            except Exception as e_direct:
                self._logger_instance.error(f"Error processing 'direct_wrapper' layout for '{table_name}': {e_direct}")
        else:
            self._logger_instance.error(f"Unknown table_layout_type: '{table_layout_type}' for table '{table_name}'.")
            return

        if not table_contexts:
            self._logger_instance.warning(f"No valid table contexts found to process for '{table_name}' (ID: {security_id}).")
            return
        
        self._logger_instance.info(f"Identified {len(table_contexts)} table context(s) to process for '{table_name}'.")

        # 3. Process each identified table context
        all_key_value_data_rows = []
        tables_successfully_parsed_count = 0

        for context_index, table_wrapper_handle_loc in enumerate(table_contexts): # table_wrapper_handle_loc is a Locator
            self._logger_instance.info(f"[PROCESS CONTEXT #{context_index+1}/{len(table_contexts)}] For '{table_name}', Security ID: {security_id}")
            
            current_instance_data_rows = []
            try:
                # Ensure the table wrapper itself (which is our context) is visible
                await table_wrapper_handle_loc.wait_for(state="visible", timeout=self.page_timeout / 2)
                self._logger_instance.debug(f"Table context (wrapper) #{context_index+1} is visible.")

                # Hierarchically target tbody.ant-table-tbody
                # Path: .ant-table-wrapper -> .ant-spin-container (optional but good to be aware of) -> .ant-table -> .ant-table-container -> .ant-table-content -> table -> tbody.ant-table-tbody
                # A more direct locator from the wrapper:
                target_tbody_loc = table_wrapper_handle_loc.locator("table > tbody.ant-table-tbody")

                if await target_tbody_loc.count() == 0:
                    self._logger_instance.warning(f"No 'tbody.ant-table-tbody' found within table context #{context_index+1} for '{table_name}'. Skipping this context.")
                    continue
                
                self._logger_instance.debug(f"Waiting for populated rows (tr.ant-table-row-level-0:has(td)) in tbody of context #{context_index+1}...")
                # Use :has(td) to ensure it's a data row, not an empty/measure row.
                # 'ant-table-row-level-0' is common, but 'ant-table-row' might be more general if level changes.
                # For File List example, it's 'ant-table-row ant-table-row-level-0'
                populated_row_loc = target_tbody_loc.locator("tr.ant-table-row:has(td)") # General for any row with cells

                try:
                    await populated_row_loc.first.wait_for(state="visible", timeout=self.page_timeout / 2)
                    self._logger_instance.info(f"Populated data rows found in tbody of context #{context_index+1}.")
                except PlaywrightTimeoutError:
                    no_data_placeholder_loc = table_wrapper_handle_loc.locator(".ant-empty-description")
                    if await no_data_placeholder_loc.count() > 0 and await no_data_placeholder_loc.is_visible():
                        no_data_text = (await no_data_placeholder_loc.text_content() or "").strip()
                        self._logger_instance.info(f"Table context #{context_index+1} shows 'No data' (text: '{no_data_text}'). Skipping.")
                    else:
                        self._logger_instance.warning(f"Timeout waiting for populated rows in table context #{context_index+1} and no clear 'No data' placeholder. Skipping.")
                    continue

                all_tr_handles_in_instance = await populated_row_loc.all()
                if not all_tr_handles_in_instance: # Should not happen if wait_for succeeded, but as a safeguard
                    self._logger_instance.info(f"No data rows to extract from table context #{context_index+1} despite visibility. Skipping.")
                    continue

                self._logger_instance.debug(f"Extracting {len(all_tr_handles_in_instance)} rows from table context #{context_index+1}.")
                for r_idx, row_handle_loc in enumerate(all_tr_handles_in_instance): # row_handle_loc is a Locator
                    cell_locators = row_handle_loc.locator("td")
                    if await cell_locators.count() == 0:
                        self._logger_instance.debug(f"Row {r_idx+1} in context #{context_index+1} has no 'td' cells. Skipping.")
                        continue
                    
                    cell_texts = await cell_locators.all_inner_texts()
                    cleaned_row_data = [text.strip() for text in cell_texts]
                    
                    if any(cleaned_row_data):
                        current_instance_data_rows.append(cleaned_row_data)
                    else:
                        self._logger_instance.debug(f"Row {r_idx+1} in context #{context_index+1} was all empty after strip.")
                
                if not current_instance_data_rows:
                    self._logger_instance.info(f"No actual data extracted from rows of table context #{context_index+1}.")
                    continue

                if is_key_value_style:
                    all_key_value_data_rows.extend(current_instance_data_rows)
                    self._logger_instance.debug(f"Added {len(current_instance_data_rows)} K/V rows from table context #{context_index+1}.")
                else: # Regular table, save it individually
                    headers = []
                    # Headers are in the thead of the table within this context
                    header_th_locators = table_wrapper_handle_loc.locator("table > thead tr th")
                    all_th_handles_loc = await header_th_locators.all() # List of Locators
                    self._logger_instance.debug(f"Found {len(all_th_handles_loc)} 'th' elements for regular table context #{context_index+1}.")

                    for th_idx, th_loc in enumerate(all_th_handles_loc):
                        title_span_loc = th_loc.locator(".ant-table-column-title")
                        text_content = ""
                        if await title_span_loc.count() > 0:
                            text_content = (await title_span_loc.first.text_content() or "").strip()
                        if not text_content: # Fallback
                            text_content = (await th_loc.text_content() or "").strip()
                        headers.append(text_content if text_content else f"Header_Empty_{th_idx}")
                    
                    df = pd.DataFrame(current_instance_data_rows)
                    if headers and len(headers) == df.shape[1]:
                        df.columns = headers
                    elif headers:
                        self._logger_instance.warning(f"Header count ({len(headers)}) mismatch with data columns ({df.shape[1]}) for regular table context #{context_index+1}. Pandas will infer/adjust.")
                        try: df.columns = headers[:df.shape[1]] if df.shape[1] <= len(headers) else headers + [f"Unnamed_{c}" for c in range(df.shape[1] - len(headers))]
                        except: self._logger_instance.error(f"Failed to reconcile headers for regular table context #{context_index+1}.")
                    
                    # For regular tables, if there are multiple contexts (unlikely for File List/Transaction History but possible if 'columns' was misused)
                    # we save them as parts.
                    file_suffix = f"_part{context_index}" if len(table_contexts) > 1 and table_layout_type == 'columns' else ""
                    parquet_file = tables_path / f"{table_name.lower().replace(' ', '_')}_{security_id}{file_suffix}.parquet"
                    try:
                        df.to_parquet(parquet_file, index=False)
                        self._logger_instance.info(f"[SAVE SUCCESS] Saved regular table data: {parquet_file} ({len(df)} rows)")
                        tables_successfully_parsed_count += 1
                    except Exception as e_save_reg:
                        self._logger_instance.error(f"Failed to save regular table {parquet_file}: {e_save_reg}")
            
            except PlaywrightTimeoutError as pte_context:
                self._logger_instance.error(f"Timeout processing table context #{context_index+1} for '{table_name}': {pte_context}")
            except Exception as e_context:
                self._logger_instance.error(f"Unexpected error processing table context #{context_index+1} for '{table_name}': {e_context}", exc_info=True)

        # 4. Final save for accumulated key-value style table
        if is_key_value_style:
            if all_key_value_data_rows:
                self._logger_instance.info(f"Processing {len(all_key_value_data_rows)} accumulated K/V rows for '{table_name}'.")
                valid_kv_data = [row for row in all_key_value_data_rows if len(row) >= 2]
                if valid_kv_data:
                    max_cols = max(len(row) for row in valid_kv_data)
                    kv_columns = ["Key", "Value"] + [f"ExtraCol_{i+1}" for i in range(max_cols - 2)]
                    padded_kv_data = [row + [None] * (max_cols - len(row)) for row in valid_kv_data]

                    df_kv = pd.DataFrame(padded_kv_data, columns=kv_columns)
                    parquet_file_kv = tables_path / f"{table_name.lower().replace(' ', '_')}_{security_id}.parquet"
                    try:
                        df_kv.to_parquet(parquet_file_kv, index=False)
                        self._logger_instance.info(f"[SAVE SUCCESS] Saved combined K/V table data: {parquet_file_kv} ({len(df_kv)} rows)")
                        tables_successfully_parsed_count += 1
                    except Exception as e_save_kv:
                        self._logger_instance.error(f"Failed to save combined K/V table {parquet_file_kv}: {e_save_kv}")
                else:
                    self._logger_instance.warning(f"No valid K/V data (min 2 columns per row) after processing all contexts for '{table_name}', Security ID: {security_id}.")
            else:
                self._logger_instance.warning(f"No K/V data accumulated after processing all contexts for '{table_name}', Security ID: {security_id}.")

        if tables_successfully_parsed_count == 0 and len(table_contexts) > 0 : # Check if any contexts were identified
            self._logger_instance.warning(f"[EXTRACT TABLE - NO DATA SAVED] No table data was ultimately saved for '{table_name}' (ID: {security_id}) despite identifying {len(table_contexts)} table context(s).")
        
        self._logger_instance.info(f"[EXTRACT TABLE END] Finished processing for '{table_name}', Security ID: {security_id}. Saved {tables_successfully_parsed_count} Parquet file(s).")

    async def _scrape_security_detail_page_otc(
        self, page: Page, detail_url: str, tables_path: Path, documents_path: Path, security_id: str
    ) -> int:
        documents_downloaded_count = 0
        self._logger_instance.debug(f"Navigating to detail page: {detail_url} for ID: {security_id}")
        # INCREASE PAGE LOAD TIMEOUT HERE if initial page elements are slow
        await page.goto(detail_url, wait_until="domcontentloaded", timeout=self.page_timeout * 1.5) # e.g., 90s
        await page.wait_for_timeout(2000) # Small static wait for JS to settle after DOM load

        # --- Security Details ---
        sec_details_header_xpath_base = "//div[contains(@class,'text-lg') and normalize-space(.)='Security Details']"
        sec_details_content_block_xpath = f"{sec_details_header_xpath_base}/ancestor::div[contains(@class, 'ant-space-item')][1]/following-sibling::div[contains(@class, 'ant-space-item')][1]/div[contains(@class, 'rounded-lg')]"
        await self._extract_table_data_from_page(
            page,
            section_locator_strategy={"xpath": sec_details_content_block_xpath},
            table_name="Security_Details",
            security_id=security_id,
            tables_path=tables_path,
            is_key_value_style=True,
            table_layout_type='columns'
        )

        # --- Underwriter Info ---
        underwriter_header_xpath_base = "//div[contains(@class,'text-lg') and normalize-space(.)='Underwriter Info']"
        underwriter_content_block_xpath = f"{underwriter_header_xpath_base}/ancestor::div[contains(@class, 'ant-space-item')][1]/following-sibling::div[contains(@class, 'ant-space-item')][1]/div[contains(@class, 'rounded-lg')]"
        await self._extract_table_data_from_page(
            page,
            section_locator_strategy={"xpath": underwriter_content_block_xpath},
            table_name="Underwriter_Info",
            security_id=security_id,
            tables_path=tables_path,
            is_key_value_style=True,
            table_layout_type='direct_wrapper'
        )

        # --- Transaction History ---
        tx_history_header_xpath_base = "//div[contains(@class,'text-lg') and normalize-space(.)='Transaction History']"
        tx_history_content_container_xpath = f"{tx_history_header_xpath_base}/ancestor::div[contains(@class, 'ant-space-item')][1]/following-sibling::div[contains(@class, 'ant-space-item')][1]"
        await self._extract_table_data_from_page(
            page,
            section_locator_strategy={"xpath": tx_history_content_container_xpath},
            table_name="Transaction_History",
            security_id=security_id,
            tables_path=tables_path,
            is_key_value_style=False,
            table_layout_type='direct_wrapper'
        )
        
        # --- File List --- (Using the previously working XPath for its section container)
        # This XPath points to the ant-col that contains the File List table wrapper.
        file_list_section_container_xpath = "//div[div/div[contains(text(), 'File List')]]/ancestor::div[contains(@class, 'ant-col')][1]"
        await self._extract_table_data_from_page(
            page,
            section_locator_strategy={"xpath": file_list_section_container_xpath},
            table_name="File_List",
            security_id=security_id,
            tables_path=tables_path,
            is_key_value_style=False,
            table_layout_type='direct_wrapper'
        )

        # --- Document Downloading using HTTP GET ---
        self._logger_instance.info(f"Processing 'File List' for document downloads for security {security_id}")
        # XPath to the ant-col that contains the File List table wrapper
        file_list_links_section_xpath = "//div[div/div[contains(text(), 'File List')]]/ancestor::div[contains(@class, 'ant-col')][1]"
        
        try:
            file_list_links_container_loc = page.locator(f"xpath={file_list_links_section_xpath}")
            # We expect the table wrapper to be inside this ant-col
            actual_table_wrapper_for_links = file_list_links_container_loc.locator("div.ant-table-wrapper")
            
            await actual_table_wrapper_for_links.wait_for(state="visible", timeout=self.page_timeout/2)
            
            file_link_locators = actual_table_wrapper_for_links.locator("table > tbody tr td a[href]")
            link_count = await file_link_locators.count()
            self._logger_instance.info(f"Found {link_count} file links for document download for security {security_id}.")
            
            if link_count > 0:
                security_docs_path = documents_path / security_id
                security_docs_path.mkdir(parents=True, exist_ok=True)
                all_link_locs = await file_link_locators.all() # Get all Locator objects

                for link_loc in all_link_locs: # Iterate through Locators
                    file_href = await link_loc.get_attribute("href")
                    file_name_text = (await link_loc.inner_text() or "unknown_file").strip()
                    
                    if not file_href:
                        self._logger_instance.warning(f"Found a file link element without an href: {file_name_text}")
                        continue
                    
                    # Sanitize the filename obtained from link text or a potential query parameter
                    # This part might need adjustment if filename isn't in link_text
                    # For S3 URLs like this, the filename is usually in response-content-disposition
                    # or as the last part of the path if not for a CDN.
                    # Here, we'll use file_name_text as a base and sanitize it.
                    safe_filename = re.sub(r'[^\w\.\- ()]', '_', file_name_text)
                    if not Path(safe_filename).suffix and ".pdf" in file_href.lower(): # Basic suffix guess
                        safe_filename += ".pdf"
                    # A more robust way might be to parse filename from query params if available
                    # from urllib.parse import urlparse, parse_qs
                    # parsed_url = urlparse(file_href)
                    # query_params = parse_qs(parsed_url.query)
                    # if 'filename' in query_params.get('response-content-disposition', [''])[0]:
                    #     # Complex parsing needed here for "inline; filename=..."
                    #     pass


                    save_path = security_docs_path / safe_filename
                    
                    self._logger_instance.debug(f"Attempting to download file: '{safe_filename}' via HTTP GET from {file_href}")
                    try:
                        # Use Playwright's APIRequestContext for the GET request
                        # It will use the browser's cookies and context if page.request is used.
                        # For external S3, context.request might be cleaner if cookies aren't needed.
                        api_request_context = page.request 
                        # Or, for a new context without page's cookies:
                        # playwright_instance = page.context.browser.playwright
                        # api_request_context = playwright_instance.request.new_context(ignore_https_errors=True)
                        
                        response = await api_request_context.get(file_href, timeout=self.page_timeout * 3) # Increased timeout for download

                        if response.ok:
                            file_bytes = await response.body()
                            with open(save_path, "wb") as f:
                                f.write(file_bytes)
                            self._logger_instance.info(f"Successfully downloaded and saved: {save_path} ({len(file_bytes)} bytes)")
                            documents_downloaded_count += 1
                        else:
                            self._logger_instance.error(f"Failed to download file '{safe_filename}'. Status: {response.status} {response.status_text}. URL: {file_href}")
                            self._logger_instance.debug(f"Response headers: {response.headers_array()}")
                            # try:
                            #     self._logger_instance.debug(f"Response text: {await response.text()}")
                            # except: pass


                    except PlaywrightTimeoutError: # This timeout is for the HTTP request itself
                        self._logger_instance.error(f"Timeout during HTTP GET for file '{safe_filename}' from {file_href}", exc_info=True)
                    except Exception as e_download:
                        self._logger_instance.error(f"Failed to download file '{safe_filename}' via HTTP GET from {file_href}: {e_download}", exc_info=True)
            else:
                self._logger_instance.info(f"No file links found to download for security {security_id} after processing locators.")

        except PlaywrightTimeoutError:
             self._logger_instance.info(f"Timeout waiting for 'File List' links container for security {security_id}. XPath: {file_list_links_section_xpath}")
        except Exception as e_file_list_dl:
            self._logger_instance.error(f"Error processing 'File List' (for downloads) for security {security_id}: {e_file_list_dl}", exc_info=True)
            
        return documents_downloaded_count

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError(f"{self.name} is async-native. Use its `_arun` method.")

    async def close(self):
        self._logger_instance.debug(f"{self.name} close() called. Playwright resources are managed within _arun per call.")

    def __del__(self):
         if hasattr(self, '_logger_instance') and self._logger_instance: 
              self._logger_instance.debug(f"{self.name} (ID: {id(self)}) instance deleted.")

async def main_test_otcmn_scraper_tool():
    log_level_str = os.getenv("LOG_LEVEL_TOOL_TEST", "DEBUG").upper()
    log_level = getattr(logging, log_level_str, logging.DEBUG)
    
    logging.basicConfig(
        level=log_level, 
        format='%(asctime)s - %(name)s - %(levelname)s - [%(process)d:%(threadName)s] - %(message)s', # Added process ID
        stream=sys.stdout
    )
    logging.getLogger("playwright").setLevel(logging.INFO) # Quieter Playwright logs
    
    test_logger = logging.getLogger(__name__ + ".OTCMNScraperTool_Test")
    test_logger.info(f"Starting OTCMN_Scraper standalone async test with log level {log_level_str}...")

    test_output_dir = Path("./test_otcmn_scraper_output_tool") # Changed dir name
    test_logger.info(f"Test output will be saved to: {test_output_dir.resolve()}")
    if test_output_dir.exists():
        test_logger.warning(f"Test output directory {test_output_dir} already exists. Files may be overwritten.")
    
    tool_instance: Optional[OTCMNScraperTool] = None
    try:
        tool_instance = OTCMNScraperTool()
        if tool_instance._logger_instance:
             tool_instance._logger_instance.setLevel(log_level) # Ensure tool instance logger respects test level

        test_logger.info("\n--- [Test Case: Full Scrape (limited pages/securities for test)] ---")
        result_scrape = await tool_instance._arun(
            output_directory=str(test_output_dir),
            cdp_endpoint_url=tool_instance.cdp_endpoint_url,
            max_listing_pages_per_board=1, 
            max_securities_to_process=1    # Further reduced for quicker test
        )
        
        test_logger.info(f"Scraping Result (JSON):")
        try:
            parsed_result = json.loads(result_scrape)
            print(json.dumps(parsed_result, indent=2)) 
            if parsed_result.get("status") != "success" or (parsed_result.get("errors") and len(parsed_result["errors"]) > 0) : # Check if errors list is non-empty
                 test_logger.error("Scraping process reported errors or did not complete successfully.")
        except json.JSONDecodeError:
            test_logger.error(f"Failed to parse result as JSON: {result_scrape}")

    except Exception as e:
        test_logger.error(f"An error occurred during the OTCMN_Scraper test: {e}", exc_info=True)
    finally:
        if tool_instance:
            test_logger.info(f"Closing OTCMNScraperTool (if applicable)...")
            await tool_instance.close() 
            test_logger.info(f"OTCMNScraperTool close called.")
        test_logger.info(f"OTCMNScraperTool standalone async test finished.")

if __name__ == "__main__":
    asyncio.run(main_test_otcmn_scraper_tool())