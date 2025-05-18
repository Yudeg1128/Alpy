import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, Literal, Tuple

# Playwright imports
from playwright.async_api import async_playwright, Page, Locator, Browser, TimeoutError as PlaywrightTimeoutError, BrowserContext

# Langchain and Pydantic imports
from langchain_core.tools import BaseTool, ToolException
from langchain_core.callbacks.manager import AsyncCallbackManagerForToolRun
from pydantic import BaseModel, Field, model_validator, PrivateAttr, field_validator

# Import from your new otcmn_interaction module
# Assuming your src directory is in PYTHONPATH or you're running from a place where Python can find 'src'
try:
    from src.otcmn_interaction.interactor import OtcmSiteInteractor
    from src.otcmn_interaction.common import (
        OtcmInteractionError,
        PageNavigationError,
        ElementNotFoundError,
        DataExtractionError,
        PageStateError, # <--- ADD THIS
        ISIN_COLUMN_HEADER_TEXT
    )
    from src.otcmn_interaction.listing_page_handler import ListingPageHandler
except ImportError as e:
    # This allows the file to be parsed even if the module isn't immediately available during linting/CI in some environments
    # but it will fail at runtime if the module isn't truly found.
    print(f"Warning: Could not import from src.otcmn_interaction: {e}. Ensure it's in PYTHONPATH.")
    OtcmSiteInteractor = None 
    OtcmInteractionError = ToolException # Fallback to generic ToolException
    PageNavigationError = ToolException
    ElementNotFoundError = ToolException
    DataExtractionError = ToolException
    ISIN_COLUMN_HEADER_TEXT = "ISIN" # Default fallback
    ListingPageHandler = None


# Module logger
logger = logging.getLogger(__name__)

class OTCMNScraperToolError(ToolException):
    """Custom exception for the OTCMNScraperTool."""
    pass

# --- Pydantic Models for Action Parameters (Unchanged) ---
class ScrapeBoardsParams(BaseModel):
    output_directory: str = Field(description="Base directory, e.g., 'otcmn_data'. 'current/' subdir will be used.")
    max_listing_pages_per_board: Optional[int] = Field(default=None, description="Max listing pages per board. Default: all.")

class FilterSecuritiesParams(BaseModel):
    output_directory: str = Field(description="Base directory, e.g., 'otcmn_data'. 'filters/' subdir will be used.")
    filter_output_filename: str = Field(description="Filename for the filter results in 'otcmn_data/filters/', e.g., 'my_filter.json'.")
    filter_currency: Optional[str] = Field(default=None)
    filter_interest_rate_min: Optional[float] = Field(default=None)
    filter_interest_rate_max: Optional[float] = Field(default=None)
    filter_maturity_cutoff_date: Optional[str] = Field(default=None, description="YYYY-MM-DD format. Securities maturing on or after this date.")
    filter_underwriter: Optional[str] = Field(default=None)

class ScrapeFilteredDetailsParams(BaseModel):
    output_directory: str = Field(description="Base directory, e.g., 'otcmn_data'. Details saved under 'current/'.")
    filter_input_filename: str = Field(description="Filter filename from 'otcmn_data/filters/' to process, e.g., 'my_filter.json'.")
    max_securities_to_process_from_filter: Optional[int] = Field(default=None, description="Max securities from the filter to scrape details for. Default: all in filter.")

class OTCMNScraperActionInput(BaseModel):
    action: Literal["scrape_boards", "filter_securities", "scrape_filtered_details"] = Field(description="The operation to perform.")
    parameters: Union[ScrapeBoardsParams, FilterSecuritiesParams, ScrapeFilteredDetailsParams] = Field(description="Parameters specific to the chosen action.")
    cdp_endpoint_url: Optional[str] = Field(default=None, description="Optional CDP endpoint for an existing browser.")


class OTCMNScraperTool(BaseTool, BaseModel): # type: ignore[misc] # For Pydantic v2 + Langchain if any issues
    name: str = "OTCMN_Data_Manager"
    description: str = (
        "Manages data scraping and filtering for financial securities from otc.mn.\n"
        "The `action_input` MUST be a JSON object containing 'action' and 'parameters'.\n\n"
        # ... (rest of description is unchanged) ...
        "Example `action_input` for 'filter_securities':\n"
        "```json\n"
        "{\n"
        "  \"action\": \"filter_securities\",\n"
        "  \"parameters\": {\n"
        "    \"output_directory\": \"./otcmn_scrape_output\",\n"
        "    \"filter_output_filename\": \"usd_bonds_filter.json\",\n"
        "    \"filter_currency\": \"USD\"\n"
        "  }\n"
        "}\n"
        "```"
    )
    args_schema: Type[BaseModel] = OTCMNScraperActionInput
    return_direct: bool = False # Standard Langchain Tool behavior
    handle_tool_error: bool = True # Let Langchain handle ToolException

    # Configuration for Playwright interactions
    playwright_channel: str = Field(default="chromium")
    login_prompt_timeout: float = Field(default=300.0) # Timeout for user to press Enter after manual login
    page_timeout: float = Field(default=60000) # Default timeout for Playwright operations in ms

    _logger_instance: logging.Logger = PrivateAttr(default=None)
    # Changed: _site_interactor now holds an instance of the new OtcmSiteInteractor
    _site_interactor: Optional['OtcmSiteInteractor'] = PrivateAttr(default=None) # type: ignore

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def _tool_post_init_validator(self) -> 'OTCMNScraperTool':
        if self._logger_instance is None:
            self._logger_instance = logger.getChild(f"{self.name}.{id(self)}")
            if not self._logger_instance.hasHandlers() and not logging.getLogger().hasHandlers():
                # Basic stdout handler if no root logger is configured
                _handler = logging.StreamHandler(sys.stdout)
                _formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s')
                _handler.setFormatter(_formatter)
                self._logger_instance.addHandler(_handler)
                self._logger_instance.propagate = False 
                # Set level for this tool's logger if not set by higher config
                if not self._logger_instance.level or self._logger_instance.level == logging.NOTSET:
                     self._logger_instance.setLevel(logging.INFO)
        
        # Check if the interaction module was loaded
        if OtcmSiteInteractor is None:
            self._logger_instance.critical("OtcmSiteInteractor from src.otcmn_interaction module was not loaded. Tool will not function.")
            # No need to raise here, as arun will fail if it's None.
        
        self._logger_instance.info(f"Initialized {self.name}. ISIN Header expected: '{ISIN_COLUMN_HEADER_TEXT}'")
        return self

    async def _save_json(self, data: Any, file_path: Path):
        """Saves data to a JSON file asynchronously."""
        self._logger_instance.debug(f"Saving JSON to: {file_path}")
        try:
            def _write_sync(): # Synchronous part to run in thread
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            
            await asyncio.to_thread(_write_sync)
            # self._logger_instance.info(f"Successfully saved JSON: {file_path}")
        except Exception as e:
            self._logger_instance.error(f"Failed to save JSON to {file_path}: {e}", exc_info=True)
            # Let this exception propagate to be caught by the main error handler in _arun
            # and added to the summary_log["errors"]
            raise OTCMNScraperToolError(f"Failed to save JSON to {file_path}: {e}") from e

    async def _arun(
        self,
        action: str, 
        parameters: Union[ScrapeBoardsParams, FilterSecuritiesParams, ScrapeFilteredDetailsParams],
        cdp_endpoint_url: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None, # Langchain callback manager
        **kwargs: Any # Catches any other args Langchain might pass
    ) -> str:
        if OtcmSiteInteractor is None: # Critical check
            self._logger_instance.error("OtcmSiteInteractor is not available. Cannot run scraping actions.")
            return json.dumps({"status": "error", "action": action, "errors": ["Tool dependency OtcmSiteInteractor not loaded."]})

        output_dir_param = getattr(parameters, 'output_directory', None)
        self._logger_instance.info(f"Action '{action}' received. Output base: '{output_dir_param if output_dir_param else 'N/A for this action'}'")
        
        summary_log: Dict[str, Any] = {"status": "success", "action": action, "parameters_used": parameters.model_dump(mode='json'), "errors": []}
        
        # Browser setup only for scraping actions
        if action in ["scrape_boards", "scrape_filtered_details"]:
            if not output_dir_param:
                summary_log["status"] = "error"
                summary_log["errors"].append(f"Action '{action}' requires 'output_directory' parameter.")
                return json.dumps(summary_log, default=str)

            async with async_playwright() as p_context:
                browser: Optional[Browser] = None
                context: Optional[BrowserContext] = None
                page: Optional[Page] = None
                browser_connected_via_cdp = False
                try:
                    if cdp_endpoint_url:
                        self._logger_instance.info(f"Attempting to connect to existing browser via CDP: {cdp_endpoint_url}")
                        browser = await p_context.chromium.connect_over_cdp(cdp_endpoint_url, timeout=self.page_timeout)
                        browser_connected_via_cdp = True
                        self._logger_instance.info("Successfully connected to browser via CDP.")
                        
                        if not browser.contexts:
                            # If no contexts, try to create one? Or is this an error state for CDP?
                            # For now, assume first context or new one.
                            self._logger_instance.warning("No existing browser contexts found via CDP. Using default or creating new one.")
                            context = await browser.new_context(user_agent="Mozilla/5.0", viewport={'width': 1920, 'height': 1080})
                        else:
                            context = browser.contexts[0] # Use the first available context
                        
                        # Try to find a relevant page or create one
                        page_found = False
                        for p_item in context.pages:
                             # Check if page URL is relevant (otc.mn domain)
                            if "otc.mn" in p_item.url:
                                page = p_item
                                await page.bring_to_front()
                                page_found = True
                                self._logger_instance.info(f"Reusing existing page from CDP: {page.url}")
                                break
                        if not page:
                            page = await context.new_page()
                            self._logger_instance.info(f"Created new page in CDP browser context.")
                        
                        self._site_interactor = OtcmSiteInteractor(page, self._logger_instance, self.page_timeout)
                        # With CDP, login is assumed to be handled externally.
                        # The interactor's navigation methods will check for login redirection.
                        self._logger_instance.info("CDP setup: Login is assumed to be handled in the external browser.")

                    else: # Launch new browser
                        self._logger_instance.info(f"Launching new browser instance (channel: {self.playwright_channel}). Headless=False for manual login.")
                        browser = await getattr(p_context, self.playwright_channel).launch(headless=False) # Must be False for manual login
                        context = await browser.new_context(user_agent="Mozilla/5.0", viewport={'width': 1920, 'height': 1080})
                        page = await context.new_page()
                        
                        # Instantiate interactor before login prompt, as it might handle the prompt
                        self._site_interactor = OtcmSiteInteractor(page, self._logger_instance, self.page_timeout)
                        await self._site_interactor.ensure_logged_in(
                            login_prompt_message="MANUAL LOGIN REQUIRED in the newly launched browser.",
                            login_wait_timeout=self.login_prompt_timeout
                        )
                        self._logger_instance.info("Manual login process completed (or skipped if already logged in).")
                    
                    if not page: raise OTCMNScraperToolError("Playwright Page object not initialized after browser setup.")
                    if not self._site_interactor : raise OTCMNScraperToolError("OtcmSiteInteractor not initialized.")

                    # Set default timeout on the context for all subsequent page operations
                    context.set_default_timeout(self.page_timeout)

                    # --- Dispatch to action-specific handlers that need Playwright ---
                    if action == "scrape_boards":
                        if not isinstance(parameters, ScrapeBoardsParams): # Should be caught by Pydantic earlier
                            raise OTCMNScraperToolError("Internal error: Parameters type mismatch for scrape_boards.")
                        # Pass summary_log to be updated directly by the handler
                        await self._handle_scrape_boards_action(parameters, summary_log) 
                    
                    elif action == "scrape_filtered_details":
                        if not isinstance(parameters, ScrapeFilteredDetailsParams):
                             raise OTCMNScraperToolError("Internal error: Parameters type mismatch for scrape_filtered_details.")
                        # This handler returns its part of the summary
                        details_summary = await self._handle_scrape_filtered_details_action(parameters)
                        summary_log.update(details_summary)

                except (OtcmInteractionError, PageNavigationError, ElementNotFoundError, DataExtractionError, PageStateError) as otc_e:
                    self._logger_instance.error(f"otcmn_interaction module error during '{action}': {otc_e}", exc_info=True)
                    summary_log["status"] = "error"
                    summary_log["errors"].append(f"Interaction module error in '{action}': {str(otc_e)}")
                except PlaywrightTimeoutError as pwt_e:
                    self._logger_instance.error(f"Playwright timeout error during '{action}': {str(pwt_e).splitlines()[0]}", exc_info=False) # Keep log concise
                    summary_log["status"] = "error"
                    summary_log["errors"].append(f"Playwright timeout in '{action}': {str(pwt_e).splitlines()[0]}")
                except OTCMNScraperToolError as tool_e: # Catch custom tool errors (like failed save)
                    self._logger_instance.error(f"Tool-specific error during '{action}': {tool_e}", exc_info=True)
                    summary_log["status"] = "error"
                    summary_log["errors"].append(f"Tool error in '{action}': {str(tool_e)}")
                except Exception as e: # Catch-all for unexpected issues
                    self._logger_instance.error(f"Unexpected error during Playwright action '{action}': {e}", exc_info=True)
                    summary_log["status"] = "error"
                    summary_log["errors"].append(f"Unexpected error in '{action}': {str(e)}")
                finally:
                    if browser and browser.is_connected():
                        if browser_connected_via_cdp:
                            # For CDP, we typically don't close the browser itself, just our connection to the page/context if managed that way.
                            # If we created the context, we could close it.
                            # For now, if connecting to existing browser, just log disconnection attempt.
                            self._logger_instance.info("CDP connection: Browser assumed to be managed externally. Not closing browser.")
                            # If page was created by us: await page.close() if page and not page.is_closed()
                            # If context was created by us: await context.close() if context
                        else: # Browser was launched by tool
                            self._logger_instance.info("Closing browser launched by the tool.")
                            await browser.close()
                    self._site_interactor = None # Clear interactor instance

        elif action == "filter_securities": # No Playwright needed
            try:
                if not isinstance(parameters, FilterSecuritiesParams):
                    raise OTCMNScraperToolError("Internal error: Parameters type mismatch for filter_securities.")
                filter_summary = await self._handle_filter_securities_action(parameters) # Returns its part
                summary_log.update(filter_summary)
            except OTCMNScraperToolError as tool_e: # Catch custom tool errors (like failed save)
                self._logger_instance.error(f"Tool-specific error during 'filter_securities': {tool_e}", exc_info=True)
                summary_log["status"] = "error"
                summary_log["errors"].append(f"Tool error in 'filter_securities': {str(tool_e)}")
            except Exception as e:
                self._logger_instance.error(f"Error during 'filter_securities' action: {e}", exc_info=True)
                summary_log["status"] = "error"
                summary_log["errors"].append(f"Filter action failed: {str(e)}")
        else:
            summary_log["status"] = "error"
            summary_log["errors"].append(f"Unknown action: {action}")
            self._logger_instance.error(f"Unknown action received by tool: {action}")

        # If there were any errors, ensure overall status reflects it
        if summary_log["errors"]:
            summary_log["status"] = "error"

        self._logger_instance.info(f"Action '{action}' finished. Overall status: {summary_log['status']}.")
        if summary_log["errors"]:
             self._logger_instance.warning(f"Errors occurred during '{action}': {summary_log['errors']}")
        # self._logger_instance.debug(f"Full summary for '{action}': {json.dumps(summary_log, indent=2, default=str)}") # For debugging, can be verbose
        return json.dumps(summary_log, default=str) # Ensure all parts are serializable

    async def _handle_scrape_boards_action(self, params: ScrapeBoardsParams, summary_log: Dict[str, Any]) -> None:
        """
        Handles the 'scrape_boards' action using the OtcmSiteInteractor.
        Attempts to reset to page 1 after each board and tab is processed.
        Modifies summary_log in place.
        """
        self._logger_instance.info(f"--- Starting _handle_scrape_boards_action (Output: {params.output_directory}) ---")
        if not self._site_interactor or not self._site_interactor.listing_handler:
            self._logger_instance.error("Site interactor or listing handler not initialized. Cannot execute scrape_boards.")
            summary_log["errors"].append("Critical: Site interactor/listing_handler not available for scrape_boards.")
            summary_log["status"] = "error"
            return

        listing_h = self._site_interactor.listing_handler

        summary_log.update({
            "board_metadata_files_saved": 0,
            "unique_securities_identified_on_boards": 0,
            "total_rows_scraped_for_boards": 0,
            "boards_processed_counts": {}, # E.g., {"primary_board_A": {"pages_scraped": X, "rows": Y}, ...}
            "pagination_reset_failures": [] # Track if _go_to_first_page_of_listing fails
        })

        base_output_path = Path(params.output_directory) / "current"
        all_securities_globally_this_run: Dict[str, Dict[str, Any]] = {} # To count unique ISINs

        try:
            await self._site_interactor.navigate_to_securities_listing_page()
        except PageNavigationError as nav_e:
            self._logger_instance.error(f"Initial navigation to securities page failed: {nav_e}")
            summary_log["errors"].append(f"Initial navigation failed: {str(nav_e)}")
            summary_log["status"] = "error"
            return

        tab_keys_to_process: List[Literal["primary", "secondary"]] = ["primary", "secondary"]
        board_chars_to_process: List[Literal["A", "B", "C"]] = ["A", "B", "C"]

        for tab_key in tab_keys_to_process:
            self._logger_instance.info(f"===== Processing Tab Group: '{listing_h.MAIN_TAB_KEY_TO_SITE_TEXT_MAP.get(tab_key, tab_key)}' ({tab_key}) =====")
            try:
                await listing_h.select_main_tab(tab_key) # This should reset headers and wait for table
            except Exception as e_tab:
                self._logger_instance.error(f"Failed to select main tab '{tab_key}': {e_tab}. Skipping this tab group.")
                summary_log["errors"].append(f"Tab select fail: {tab_key} - {str(e_tab)}")
                continue
            
            # Explicitly go to page 1 after tab selection is complete
            if not await listing_h._go_to_first_page_of_listing(f"after selecting tab {tab_key}"):
                self._logger_instance.warning(f"Failed to confirm/reset to Page 1 after selecting tab {tab_key}. Proceeding, but pagination might be off for the first board.")
                summary_log["pagination_reset_failures"].append(f"Tab: {tab_key}")
            listing_h.reset_header_cache() # Ensure headers are fresh for the first board of new tab

            for board_char in board_chars_to_process:
                board_category_name = f"{tab_key}_board_{board_char}"
                self._logger_instance.info(f"--- Processing Board: '{board_char}' (Category: {board_category_name}) ---")
                summary_log["boards_processed_counts"][board_category_name] = {"pages_scraped": 0, "rows_on_board": 0}
                board_output_path = base_output_path / board_category_name
                
                try:
                    await listing_h.select_board_filter_exclusively(board_char) # This should reset headers and wait for table
                except Exception as e_board_filter:
                    self._logger_instance.error(f"Failed to set filter to '{board_char}' for tab '{tab_key}': {e_board_filter}. Skipping this board.")
                    summary_log["errors"].append(f"Board filter set fail: {board_category_name} - {str(e_board_filter)}")
                    continue
                
                # Explicitly go to page 1 after board filter selection
                if not await listing_h._go_to_first_page_of_listing(f"after selecting board {board_char} on tab {tab_key}"):
                    self._logger_instance.warning(f"Failed to confirm/reset to Page 1 for {board_category_name}. Proceeding, but data might start from wrong page.")
                    summary_log["pagination_reset_failures"].append(f"Board: {board_category_name}")
                listing_h.reset_header_cache() # Ensure headers are fresh for this specific board

                current_board_all_rows_for_json: List[Dict[str, Any]] = []
                page_scrape_count_for_this_board = 0

                while True:
                    page_scrape_count_for_this_board += 1
                    summary_log["boards_processed_counts"][board_category_name]["pages_scraped"] = page_scrape_count_for_this_board

                    if params.max_listing_pages_per_board and page_scrape_count_for_this_board > params.max_listing_pages_per_board:
                        self._logger_instance.info(f"Reached max_listing_pages ({params.max_listing_pages_per_board}) for {board_category_name}.")
                        break
                    
                    self._logger_instance.info(f"Scraping page {page_scrape_count_for_this_board} for {board_category_name}...")
                    
                    try:
                        headers_from_page, rows_as_dicts_from_page = await listing_h.extract_current_page_data()
                        if not headers_from_page and rows_as_dicts_from_page: # Headers failed but rows came? Unlikely with current logic.
                             self._logger_instance.error(f"Header extraction failed but rows were returned for {board_category_name} P{page_scrape_count_for_this_board}. Critical error.")
                             summary_log["errors"].append(f"Header fail but rows found: {board_category_name} P{page_scrape_count_for_this_board}")
                             break 
                        if not headers_from_page and not rows_as_dicts_from_page and page_scrape_count_for_this_board == 1:
                             self._logger_instance.info(f"No headers and no rows on first page attempt for {board_category_name}. Assuming board is empty.")
                             break


                    except DataExtractionError as e_extract:
                        self._logger_instance.error(f"Data extraction error for {board_category_name}, page {page_scrape_count_for_this_board}: {e_extract}. Stopping this board.")
                        summary_log["errors"].append(f"Data extraction error: {board_category_name} P{page_scrape_count_for_this_board} - {str(e_extract)}")
                        break 
                    except Exception as e_unexp_extract: # Catch any other error during extraction
                        self._logger_instance.error(f"Unexpected error during data extraction for {board_category_name}, P{page_scrape_count_for_this_board}: {e_unexp_extract}", exc_info=True)
                        summary_log["errors"].append(f"Unexpected extraction error: {board_category_name} P{page_scrape_count_for_this_board} - {str(e_unexp_extract)}")
                        break

                    if not rows_as_dicts_from_page and page_scrape_count_for_this_board > 1 : # No more rows on subsequent pages
                        self._logger_instance.info(f"No more data rows found for {board_category_name} after page {page_scrape_count_for_this_board -1}.")
                        break 
                    if not rows_as_dicts_from_page and page_scrape_count_for_this_board == 1 : # No rows on the first page itself
                        self._logger_instance.info(f"No data rows found on the very first page of {board_category_name}.")
                        break


                    self._logger_instance.debug(f"Processing {len(rows_as_dicts_from_page)} rows from page {page_scrape_count_for_this_board} of {board_category_name}.")
                    for row_dict in rows_as_dicts_from_page:
                        current_board_all_rows_for_json.append(row_dict)
                        summary_log["total_rows_scraped_for_boards"] += 1
                        summary_log["boards_processed_counts"][board_category_name]["rows_on_board"] +=1

                        isin_value = row_dict.get(ISIN_COLUMN_HEADER_TEXT) 
                        if isin_value and isinstance(isin_value, str) and isin_value.strip() and isin_value != "0": # Added check for ISIN != "0"
                            isin_value = isin_value.strip()
                            security_output_path_isin_dir = board_output_path / isin_value
                            
                            security_metadata_content = {**row_dict}
                            if "_url_id" in security_metadata_content:
                                security_metadata_content["url_id"] = security_metadata_content.pop("_url_id")
                            if "_detail_url_path" in security_metadata_content:
                                security_metadata_content["detail_url_path"] = security_metadata_content.pop("_detail_url_path")

                            try:
                                await self._save_json(security_metadata_content, security_output_path_isin_dir / "security_metadata.json")
                                if isin_value not in all_securities_globally_this_run:
                                     all_securities_globally_this_run[isin_value] = security_metadata_content
                                     summary_log["unique_securities_identified_on_boards"] += 1
                            except OTCMNScraperToolError as e_save_sec_meta:
                                self._logger_instance.error(f"Failed to save security_metadata for ISIN {isin_value} in {board_category_name}: {e_save_sec_meta}")
                                summary_log["errors"].append(f"Save fail sec_meta ISIN {isin_value} ({board_category_name}): {str(e_save_sec_meta)}")
                        else:
                            company_name_guess = row_dict.get(headers_from_page[0] if headers_from_page else "col_0", "Unknown Security")
                            self._logger_instance.warning(f"No valid ISIN ('{isin_value}') for row '{company_name_guess}' in {board_category_name}, P{page_scrape_count_for_this_board}. Skipping security_metadata save.")
                    
                    try:
                        can_go_next = await listing_h.go_to_next_page()
                        if not can_go_next:
                            self._logger_instance.info(f"No more pages for {board_category_name} after page {page_scrape_count_for_this_board}.")
                            break 
                    except Exception as e_paginate: # Catch any error from go_to_next_page
                        self._logger_instance.error(f"Error during pagination for {board_category_name} after P{page_scrape_count_for_this_board}: {e_paginate}", exc_info=True)
                        summary_log["errors"].append(f"Pagination error: {board_category_name} P{page_scrape_count_for_this_board} - {str(e_paginate)}")
                        break 
                
                # After processing all pages for a board
                if current_board_all_rows_for_json:
                    try:
                        await self._save_json(current_board_all_rows_for_json, board_output_path / "board_metadata.json")
                        summary_log["board_metadata_files_saved"] += 1
                    except OTCMNScraperToolError as e_save_board_meta:
                        summary_log["errors"].append(f"Save fail board_meta for {board_category_name}: {str(e_save_board_meta)}")
                else:
                    self._logger_instance.info(f"No data collected for {board_category_name} to save in board_metadata.json.")

                # "Blindly" attempt to go to page 1 after finishing a board, to prepare for the next board filter.
                # This is done even if max_listing_pages_per_board was hit.
                self._logger_instance.info(f"Finished processing board {board_category_name}. Attempting reset to page 1.")
                if not await listing_h._go_to_first_page_of_listing(f"cleanup after board {board_category_name}"):
                    self._logger_instance.warning(f"Could not confirm Page 1 after board {board_category_name}. Next board might start on an unexpected page if tab remains same.")
                    summary_log["pagination_reset_failures"].append(f"End of board: {board_category_name}")
                listing_h.reset_header_cache() # Reset headers before next board on same tab
            
            # After processing all boards for a given tab
            self._logger_instance.info(f"Finished all boards for tab {tab_key}. Attempting reset to page 1 before next tab.")
            if not await listing_h._go_to_first_page_of_listing(f"cleanup after tab {tab_key}"):
                 self._logger_instance.warning(f"Could not confirm Page 1 after finishing tab {tab_key}. Next tab might start on an unexpected page.")
                 summary_log["pagination_reset_failures"].append(f"End of tab: {tab_key}")
            # Headers will be reset by select_main_tab when it's called for the next tab.

        # Final summary updates
        summary_log["boards_processed"] = list(summary_log["boards_processed_counts"].keys()) # Keep original for compatibility
        # pages_scraped_per_board is now within boards_processed_counts
        
        if summary_log["errors"] or summary_log["pagination_reset_failures"]:
            summary_log["status"] = "warning" if not summary_log["errors"] else "error" # Mark as warning if only pagination reset issues

        self._logger_instance.info(f"--- Finished _handle_scrape_boards_action. Status: {summary_log['status']} ---")


    async def _handle_filter_securities_action(self, params: FilterSecuritiesParams) -> Dict[str, Any]:
        # This method does not use Playwright and can remain largely unchanged
        # except for ensuring it uses the correct ISIN_COLUMN_HEADER_TEXT from common
        action_summary = {"filters_applied": [], "matching_securities_count": 0, "filter_file_saved_to": ""}
        
        # Ensure output_directory exists for "filters" subdir
        output_dir = Path(params.output_directory)
        if not output_dir.exists():
             self._logger_instance.error(f"Base output directory '{output_dir}' does not exist for filtering.")
             action_summary["error"] = f"Output directory '{params.output_directory}' not found."
             return action_summary

        current_data_path = output_dir / "current"
        filters_path = output_dir / "filters"
        filters_path.mkdir(parents=True, exist_ok=True)

        matching_securities: List[Dict[str, Any]] = []

        if not current_data_path.exists() or not current_data_path.is_dir():
            self._logger_instance.warning(f"'current' data directory not found or not a directory at {current_data_path}. Cannot filter.")
            action_summary["error"] = f"Source data directory 'current/' not found at {current_data_path}."
            return action_summary

        self._logger_instance.info(f"Starting filter operation. Reading from: {current_data_path}")
        
        # Iterate through board_category folders, then ISIN folders
        for board_category_dir in current_data_path.iterdir():
            if not board_category_dir.is_dir(): continue
            self._logger_instance.debug(f"Scanning board category: {board_category_dir.name}")
            for isin_dir in board_category_dir.iterdir():
                if not isin_dir.is_dir(): continue 
                
                sec_meta_file = isin_dir / "security_metadata.json"
                if sec_meta_file.exists() and sec_meta_file.is_file():
                    try:
                        with open(sec_meta_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        
                        match = True # Assume match initially
                        applied_for_this_security = []

                        # Currency Filter
                        if params.filter_currency:
                            # The key for currency in security_metadata.json should be consistent
                            # It's derived from table headers. Let's assume it's "Currency".
                            # If the table header changes, this key lookup needs to adapt or be robust.
                            meta_currency = metadata.get("Currency", "").strip()
                            if not meta_currency or meta_currency.upper() != params.filter_currency.upper():
                                match = False
                            else:
                                applied_for_this_security.append(f"currency={params.filter_currency}")
                        
                        # Interest Rate Min Filter
                        if match and params.filter_interest_rate_min is not None:
                            # Assuming "Interest Rate" is the header. Value is like "10.00%"
                            rate_str = metadata.get("Interest Rate", "")
                            try:
                                rate_val = float(rate_str.replace('%', '').strip())
                                if rate_val < params.filter_interest_rate_min:
                                    match = False
                                else:
                                    applied_for_this_security.append(f"interest_rate_min={params.filter_interest_rate_min}")
                            except ValueError: # Cannot parse rate
                                self._logger_instance.debug(f"Could not parse interest rate '{rate_str}' for ISIN {metadata.get(ISIN_COLUMN_HEADER_TEXT)}. Filter skipped for this criterion.")
                                # Decide if unparseable rate means it doesn't match, or skip filter.
                                # For now, if it's critical, it implies no match.
                                match = False 
                        
                        # Interest Rate Max Filter
                        if match and params.filter_interest_rate_max is not None:
                            rate_str = metadata.get("Interest Rate", "")
                            try:
                                rate_val = float(rate_str.replace('%', '').strip())
                                if rate_val > params.filter_interest_rate_max:
                                    match = False
                                else:
                                    applied_for_this_security.append(f"interest_rate_max={params.filter_interest_rate_max}")
                            except ValueError:
                                match = False
                        
                        # Maturity Cutoff Date Filter
                        if match and params.filter_maturity_cutoff_date:
                            # Assuming "Maturity Date" is "YYYY-MM-DD"
                            maturity_date_str = metadata.get("Maturity Date", "")
                            try:
                                # Basic YYYY-MM-DD string comparison works here
                                if maturity_date_str < params.filter_maturity_cutoff_date:
                                    match = False
                                else:
                                    applied_for_this_security.append(f"maturity_cutoff={params.filter_maturity_cutoff_date}")
                            except TypeError: # if maturity_date_str is not string
                                match = False

                        # Underwriter Filter (case-insensitive substring match)
                        if match and params.filter_underwriter:
                            # Assuming "Underwriter" is the header
                            underwriter_str = metadata.get("Underwriter", "").strip()
                            if params.filter_underwriter.lower() not in underwriter_str.lower():
                                match = False
                            else:
                                 applied_for_this_security.append(f"underwriter_contains='{params.filter_underwriter}'")


                        if match:
                            action_summary["filters_applied"].extend(applied_for_this_security)
                            # Ensure 'detail_url_path' and 'url_id' were saved by scrape_boards
                            # The scrape_boards now saves 'detail_url_path' (renamed from _detail_url_path)
                            # and 'url_id' (renamed from _url_id)
                            security_isin = metadata.get(ISIN_COLUMN_HEADER_TEXT) # Use the tool's constant
                            detail_url = metadata.get("detail_url_path") # This should be present
                            url_id_val = metadata.get("url_id")
                            
                            if not security_isin:
                                self._logger_instance.warning(f"Security metadata file {sec_meta_file} is missing '{ISIN_COLUMN_HEADER_TEXT}'. Cannot reliably include in filter.")
                                continue
                            if not detail_url and url_id_val : # Fallback if somehow detail_url_path was missing but url_id is there
                                from src.otcmn_interaction.common import SECURITY_DETAIL_URL_PATH_TEMPLATE
                                detail_url = SECURITY_DETAIL_URL_PATH_TEMPLATE.format(url_id_val)
                                self._logger_instance.debug(f"Reconstructed detail_url_path for ISIN {security_isin} using url_id.")
                            elif not detail_url and not url_id_val:
                                self._logger_instance.warning(f"Security ISIN {security_isin} matched filters but missing 'detail_url_path' and 'url_id' in metadata. Cannot include for detail scraping.")
                                continue


                            matching_securities.append({
                                "isin": security_isin,
                                "detail_url_path": detail_url, # Essential for scrape_filtered_details
                                "board_category": board_category_dir.name, # From folder structure
                                "original_metadata_summary": { # For quick review in the filter file
                                    # Pick a few key fields. These keys should match actual headers.
                                    "Security": metadata.get("Security"), 
                                    "Currency": metadata.get("Currency"),
                                    "Interest Rate": metadata.get("Interest Rate"),
                                    "Maturity Date": metadata.get("Maturity Date"),
                                    ISIN_COLUMN_HEADER_TEXT: security_isin # Ensure this is also here
                                }
                            })
                    except json.JSONDecodeError as e_json:
                        self._logger_instance.error(f"Error decoding JSON from metadata file {sec_meta_file}: {e_json}")
                        action_summary.setdefault("file_errors", []).append(f"JSONDecodeError: {sec_meta_file}")
                    except Exception as e:
                        self._logger_instance.error(f"Unexpected error processing metadata file {sec_meta_file}: {e}", exc_info=True)
                        action_summary.setdefault("file_errors", []).append(f"ProcessingError: {sec_meta_file} - {str(e)}")
        
        filter_file_path = filters_path / params.filter_output_filename
        try:
            await self._save_json(matching_securities, filter_file_path)
            action_summary["matching_securities_count"] = len(matching_securities)
            action_summary["filter_file_saved_to"] = str(filter_file_path.resolve())
            action_summary["filters_applied"] = sorted(list(set(action_summary["filters_applied"]))) # Unique and sorted
            self._logger_instance.info(f"Filter results saved to {filter_file_path}. Matched {len(matching_securities)} securities.")
        except OTCMNScraperToolError as e_save: # Error from _save_json
            self._logger_instance.error(f"Failed to save filter output file {filter_file_path}: {e_save}")
            action_summary["error"] = f"Failed to save filter output: {str(e_save)}"
            action_summary["filter_file_saved_to"] = "SAVE_FAILED"

        return action_summary


    async def _handle_scrape_filtered_details_action(self, params: ScrapeFilteredDetailsParams) -> Dict[str, Any]:
        """
        Handles scraping details for securities listed in a filter file.
        Uses OtcmSiteInteractor and its DetailPageHandler.
        """
        self._logger_instance.info("--- Starting _handle_scrape_filtered_details_action ---")
        if not self._site_interactor or not self._site_interactor.detail_handler:
            self._logger_instance.error("Site interactor or detail handler not initialized. Cannot execute scrape_filtered_details.")
            return {"error": "Critical: Site interactor/detail_handler not available."}

        detail_h = self._site_interactor.detail_handler # Convenience alias

        action_summary = {
            "securities_from_filter_file": 0,
            "securities_processed_attempted": 0,
            "details_successfully_scraped_count": 0, 
            "documents_downloaded_total_count": 0,
            "errors_by_isin": {} # Store errors per ISIN
        }
        
        output_dir = Path(params.output_directory)
        filter_file_path = output_dir / "filters" / params.filter_input_filename
        current_data_base_path = output_dir / "current" # Where details will be saved

        if not filter_file_path.exists() or not filter_file_path.is_file():
            self._logger_instance.error(f"Filter file not found: {filter_file_path}")
            action_summary["error"] = f"Filter file {params.filter_input_filename} not found at {filter_file_path}."
            return action_summary

        try:
            with open(filter_file_path, 'r', encoding='utf-8') as f:
                securities_to_scrape_meta = json.load(f)
        except json.JSONDecodeError as e_json:
            self._logger_instance.error(f"Error decoding JSON from filter file {filter_file_path}: {e_json}")
            action_summary["error"] = f"Invalid JSON in filter file {filter_file_path}."
            return action_summary
        
        action_summary["securities_from_filter_file"] = len(securities_to_scrape_meta)
        
        securities_to_process_list = securities_to_scrape_meta
        if params.max_securities_to_process_from_filter is not None and params.max_securities_to_process_from_filter >= 0:
            securities_to_process_list = securities_to_scrape_meta[:params.max_securities_to_process_from_filter]
            self._logger_instance.info(f"Limiting processing to first {len(securities_to_process_list)} securities from filter due to 'max_securities_to_process_from_filter'.")

        self._logger_instance.info(f"Processing details for {len(securities_to_process_list)} securities from filter file {params.filter_input_filename}.")

        for sec_info in securities_to_process_list:
            action_summary["securities_processed_attempted"] += 1
            isin = sec_info.get("isin")
            detail_url_path_segment = sec_info.get("detail_url_path") # E.g., "/securities/detail?id=xxxxx"
            board_category = sec_info.get("board_category") # E.g., "primary_board_A"

            if not (isin and detail_url_path_segment and board_category):
                err_msg = f"Skipping security due to missing info in filter entry: ISIN='{isin}', URLPath='{detail_url_path_segment}', Board='{board_category}'. Entry: {sec_info}"
                self._logger_instance.warning(err_msg)
                action_summary.setdefault("skipped_securities_filter_format_issue", []).append(err_msg)
                if isin: action_summary["errors_by_isin"].setdefault(isin, []).append("Missing key data in filter file entry.")
                continue

            # Extract URL ID from the path segment for navigation if DetailPageHandler expects ID not full path
            url_id_match = re.search(r"id=([a-f0-9-]+)", detail_url_path_segment, re.IGNORECASE)
            if not url_id_match:
                err_msg = f"Could not extract URL ID from detail_url_path '{detail_url_path_segment}' for ISIN {isin}. Skipping."
                self._logger_instance.error(err_msg)
                action_summary["errors_by_isin"].setdefault(isin, []).append(err_msg)
                continue
            security_url_id = url_id_match.group(1)

            self._logger_instance.info(f"Scraping details for ISIN: {isin}, URL ID: {security_url_id} (from path: {detail_url_path_segment})")
            
            # Define output paths for this security's details
            # Path: <output_directory>/current/<board_category>/<ISIN>/
            security_output_path_isin_dir = current_data_base_path / board_category / isin
            security_output_path_isin_dir.mkdir(parents=True, exist_ok=True)
            
            documents_output_path_for_isin = security_output_path_isin_dir / "documents"
            # The DetailPageHandler's download method should create 'documents' if it doesn't exist.

            try:
                # 1. Navigate to the detail page
                await self._site_interactor.navigate_to_security_detail_page(security_url_id) # Uses the OtcmSiteInteractor navigation method
                
                # 2. Extract all renderable data (tables, descriptions)
                #    The DetailPageHandler should define what "all_tables_data" means.
                #    It should be a dictionary where keys are like "security_details", "file_list_metadata", etc.
                all_page_data_from_detail = await detail_h.extract_all_renderable_data() # Call method on DetailPageHandler
                
                for data_key_name, data_content in all_page_data_from_detail.items():
                    if data_content: # Only save if data was extracted for this key
                        # Use a consistent naming scheme for the JSON files, e.g., security_info.json, underwriter_info.json
                        # This assumes data_key_name from extract_all_renderable_data is suitable for filenames.
                        json_file_name = f"{data_key_name}.json" 
                        await self._save_json(data_content, security_output_path_isin_dir / json_file_name)
                        self._logger_instance.debug(f"Saved '{json_file_name}' for ISIN {isin}.")
                    else:
                        self._logger_instance.debug(f"No data extracted for key '{data_key_name}' for ISIN {isin}.")
                
                # 3. Download documents
                #    The `extract_all_renderable_data` should have populated a 'file_list_metadata' key
                #    which `download_documents_from_page` can use.
                file_list_for_download = all_page_data_from_detail.get("file_list_metadata", [])
                if isinstance(file_list_for_download, list) and file_list_for_download:
                    self._logger_instance.info(f"Found {len(file_list_for_download)} files to potentially download for ISIN {isin}.")
                    downloaded_file_infos = await detail_h.download_documents_from_page(
                        file_list_meta=file_list_for_download,
                        download_dir=documents_output_path_for_isin
                    )
                    action_summary["documents_downloaded_total_count"] += len(downloaded_file_infos)
                    if downloaded_file_infos:
                         self._logger_instance.info(f"Downloaded {len(downloaded_file_infos)} documents for ISIN {isin}.")
                else:
                    self._logger_instance.info(f"No files listed or file metadata missing for download for ISIN {isin}.")

                action_summary["details_successfully_scraped_count"] += 1
                self._logger_instance.info(f"Successfully processed details for ISIN {isin}.")

            except (PageNavigationError, ElementNotFoundError, DataExtractionError, PageStateError, OtcmInteractionError) as otc_e:
                err_msg = f"Interaction module error for ISIN {isin}: {str(otc_e)}"
                self._logger_instance.error(err_msg, exc_info=True) # Log with full traceback for these
                action_summary["errors_by_isin"].setdefault(isin, []).append(err_msg)
            except OTCMNScraperToolError as tool_e: # E.g. from _save_json
                err_msg = f"Tool error during detail scraping for ISIN {isin}: {str(tool_e)}"
                self._logger_instance.error(err_msg, exc_info=True)
                action_summary["errors_by_isin"].setdefault(isin, []).append(err_msg)
            except Exception as e: # Catch-all for unexpected
                err_msg = f"Unexpected error scraping details for ISIN {isin}: {e}"
                self._logger_instance.error(err_msg, exc_info=True)
                action_summary["errors_by_isin"].setdefault(isin, []).append(err_msg)
        
        self._logger_instance.info(f"--- Finished _handle_scrape_filtered_details_action ---")
        return action_summary


    # _run method for synchronous execution (Langchain standard, though we are async-first)
    def _run(self, *args: Any, **kwargs: Any) -> str:
        """Synchronous execution is not recommended for this async-native tool."""
        self._logger_instance.warning(f"{self.name} is async-native. Using `_run` is not recommended and may block.")
        # A simple way to run async code from sync, but can have issues in some event loops.
        # For proper sync execution, one might need to manage an event loop explicitly.
        # LangChain's default tool execution might handle this if it calls arun from a loop.
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If in an existing loop (e.g. Jupyter), create a task.
                # This is a simplified approach; consider nest_asyncio if issues arise.
                self._logger_instance.debug("Event loop is running, creating task for _arun.")
                future = asyncio.ensure_future(self.arun(*args, **kwargs)) # arun is the public Langchain method
                # This will not block here in a running loop, but the caller of _run might not get the result immediately.
                # This is generally problematic for LangChain's sync _run expectations.
                return "Async task created. Result will be available when the task completes. (Sync _run is not ideal)"
            else:
                self._logger_instance.debug("No event loop running, using asyncio.run for _arun.")
                return loop.run_until_complete(self.arun(*args, **kwargs))
        except RuntimeError as e:
            if "cannot be called from a running event loop" in str(e) or "no current event loop" in str(e):
                 self._logger_instance.error(f"Event loop issue in _run: {e}. Consider using `arun` directly or nest_asyncio if in Jupyter.")
                 raise NotImplementedError(f"{self.name}._run() encountered an event loop issue. Use `arun` or ensure proper async context.") from e
            raise
        
    async def close(self): # Optional, if any resources need explicit cleanup not handled by Playwright context manager
        self._logger_instance.debug(f"{self.name} close() called. Currently no specific resources to clean other than Playwright context.")

    # __del__ can be problematic with async resources if not careful.
    # Playwright browser/context closure is best handled by its async context manager in _arun.
    def __del__(self):
         if hasattr(self, '_logger_instance') and self._logger_instance: 
              self._logger_instance.debug(f"{self.name} (ID: {id(self)}) instance being deleted.")

# [Previous code from OTCMNScraperTool class and imports remains above]
# ...
# class OTCMNScraperTool(BaseTool, BaseModel):
# ... (all the code for the tool class itself) ...

async def main_test_otcmn_scraper_tool():
    # --- Logging Setup ---
    # Choose a default log level for testing. Can be overridden by environment variable.
    default_log_level_str = "INFO" # Use INFO for less verbose board scrape, DEBUG for more detail
    log_level_str = os.getenv("LOG_LEVEL_TOOL_TEST", default_log_level_str).upper()
    
    try:
        log_level = getattr(logging, log_level_str)
    except AttributeError:
        print(f"Warning: Invalid LOG_LEVEL_TOOL_TEST '{log_level_str}'. Defaulting to {default_log_level_str}.")
        log_level = getattr(logging, default_log_level_str)

    # Configure root logger - this will affect all loggers unless they have specific handlers/levels
    logging.basicConfig(
        level=log_level, 
        format='%(asctime)s - %(name)s - %(levelname)s - [%(process)d:%(threadName)s] - %(message)s',
        stream=sys.stdout, # Ensure logs go to stdout for visibility
        force=True # Override any existing basicConfig
    )
    
    # Specifically control Playwright's noisy logs unless we are deep debugging Playwright itself.
    # If main log level is DEBUG, Playwright logs at INFO. Otherwise, WARNING.
    playwright_log_level = logging.INFO if log_level == logging.DEBUG else logging.WARNING
    logging.getLogger("playwright").setLevel(playwright_log_level)

    # Logger for this test function
    test_logger = logging.getLogger(__name__ + ".OTCMNScraperTool_TestRunner")
    test_logger.setLevel(log_level) # Ensure this logger also respects the chosen level
    # test_logger.propagate = False # if you want to stop it from going to root, but usually not needed if root is configured.
    
    test_logger.info(f"--- Starting OTCMNScraperTool Standalone Test Runner ---")
    test_logger.info(f"Global Log Level set to: {logging.getLevelName(log_level)}")
    test_logger.info(f"Playwright Log Level set to: {logging.getLevelName(playwright_log_level)}")
    test_logger.info(f"Test Runner Log Level set to: {logging.getLevelName(test_logger.level)}")


    # --- Test Configuration ---
    # Using a relative path for test output. Ensure this script is run from a context where this path makes sense.
    test_base_output_dir = Path("./otcmn_tool_test_output") 
    
    # Optional: Clean up test directory for a completely fresh run (be careful with this)
    # import shutil
    # if test_base_output_dir.exists():
    #     test_logger.warning(f"Removing existing test output directory: {test_base_output_dir.resolve()}")
    #     shutil.rmtree(test_base_output_dir)
    
    try:
        test_base_output_dir.mkdir(parents=True, exist_ok=True)
        test_logger.info(f"Test output will be saved in: {test_base_output_dir.resolve()}")
    except Exception as e_dir:
        test_logger.error(f"Could not create test output directory {test_base_output_dir}: {e_dir}", exc_info=True)
        return # Cannot proceed without output directory

    # --- CDP Endpoint ---
    # Get CDP endpoint from command line argument if provided, otherwise it's None.
    cdp_arg = None
    if len(sys.argv) > 1 and sys.argv[1].startswith("ws://"):
        cdp_arg = sys.argv[1]
        test_logger.info(f"Using CDP endpoint from command line argument: {cdp_arg}")
    else:
        test_logger.info("No CDP endpoint provided via command line argument. Tool will launch its own browser for scraping actions.")

    tool_instance: Optional[OTCMNScraperTool] = None
    try:
        # Instantiate the tool
        tool_instance = OTCMNScraperTool()
        
        # Ensure the tool's internal logger also respects the desired log level for the test run
        if tool_instance._logger_instance:
            tool_instance._logger_instance.setLevel(log_level)
            # The OtcmSiteInteractor and its handlers get their logger from the tool's logger,
            # so their log levels will also be influenced by this.
            test_logger.debug(f"Tool's internal logger level set to: {logging.getLevelName(tool_instance._logger_instance.level)}")


        # --- [TEST CASE 1: scrape_boards] ---
        # test_logger.info("\n--- [Test Case: scrape_boards action] ---")
        # scrape_boards_params = ScrapeBoardsParams(
        #     output_directory=str(test_base_output_dir), # Pass the Path object as string
        #     max_listing_pages_per_board=2 # Limit pages for quicker testing; set to None or higher for full scrape
        # )
        # scrape_boards_action_input_dict = OTCMNScraperActionInput(
        #     action="scrape_boards",
        #     parameters=scrape_boards_params,
        #     cdp_endpoint_url=cdp_arg
        # ).model_dump(mode='json') # Use .model_dump() for Pydantic v2

        # test_logger.info(f"Tool input for 'scrape_boards': {json.dumps(scrape_boards_action_input_dict, indent=2)}")
        
        # # Call arun using dictionary unpacking for the arguments
        # result_boards_str = await tool_instance.arun(tool_input=scrape_boards_action_input_dict)
        
        # test_logger.info(f"'scrape_boards' action raw result string:\n{result_boards_str}")
        # try:
        #     result_boards_json = json.loads(result_boards_str)
        #     test_logger.info(f"'scrape_boards' action parsed JSON result:\n{json.dumps(result_boards_json, indent=2, ensure_ascii=False)}")
        #     if result_boards_json.get("status") != "success" or result_boards_json.get("errors"):
        #         test_logger.error(f"'scrape_boards' action reported status '{result_boards_json.get('status')}' or errors: {result_boards_json.get('errors')}")
        # except json.JSONDecodeError:
        #     test_logger.error(f"Failed to parse 'scrape_boards' result as JSON: {result_boards_str}")


        # --- [TEST CASE 2: filter_securities] ---
        # --- [TEST CASE 2B: filter_securities with multiple criteria] ---
        # test_logger.info("\n--- [Test Case 2B: filter_securities action - multiple criteria] ---")
        
        # from datetime import datetime, timedelta
        # # Calculate maturity cutoff date: 3 months from today
        # cutoff_date = datetime.now() + timedelta(days=6*30) # Approximate 6 months
        # maturity_cutoff_date_str = cutoff_date.strftime("%Y-%m-%d")
        # test_logger.info(f"Calculated maturity cutoff date for test: {maturity_cutoff_date_str}")

        # multi_filter_params = FilterSecuritiesParams(
        #     output_directory=str(test_base_output_dir),
        #     filter_output_filename="test_filter_mnt_specific_bonds.json",
        #     filter_currency="MNT",
        #     filter_interest_rate_min=12.0,
        #     filter_interest_rate_max=12.0,
        #     filter_maturity_cutoff_date=maturity_cutoff_date_str,
        #     filter_underwriter="Tavan Bogd Capital" # Case-insensitive substring match
        # )
        # multi_filter_action_input_dict = OTCMNScraperActionInput(
        #     action="filter_securities",
        #     parameters=multi_filter_params,
        #     cdp_endpoint_url=None 
        # ).model_dump(mode='json')

        # test_logger.info(f"Tool input for 'filter_securities' (multi-criteria): {json.dumps(multi_filter_action_input_dict, indent=2)}")
        # result_multi_filter_str = await tool_instance.arun(tool_input=multi_filter_action_input_dict)

        # test_logger.info(f"'filter_securities' (multi-criteria) action raw result string:\n{result_multi_filter_str}")
        # try:
        #     result_multi_filter_json = json.loads(result_multi_filter_str)
        #     test_logger.info(f"'filter_securities' (multi-criteria) action parsed JSON result:\n{json.dumps(result_multi_filter_json, indent=2, ensure_ascii=False)}")
        #     if result_multi_filter_json.get("status") != "success" or result_multi_filter_json.get("errors"):
        #          test_logger.error(f"'filter_securities' (multi-criteria) action reported status '{result_multi_filter_json.get('status')}' or errors: {result_multi_filter_json.get('errors')}")
        #     elif result_multi_filter_json.get("matching_securities_count", 0) == 0:
        #          test_logger.warning("Multi-criteria filter action succeeded but found 0 matching securities. This might be expected if no securities met all criteria.")

        # except json.JSONDecodeError:
        #     test_logger.error(f"Failed to parse 'filter_securities' (multi-criteria) result as JSON: {result_multi_filter_str}")
        
        # --- [TEST CASE 3: scrape_filtered_details] ---
        # # This test assumes a filter file (e.g., 'test_filter_mnt_specific_bonds.json') exists from the previous step.
        test_logger.info("\n--- [Test Case: scrape_filtered_details action] ---")
        
        # Check if the filter file from the previous step exists and has content
        filter_file_to_use = test_base_output_dir / "filters" / "test_filter_mnt_specific_bonds.json"
        if not filter_file_to_use.exists():
            test_logger.warning(f"Filter file '{filter_file_to_use}' not found. Skipping 'scrape_filtered_details' test.")
        else:
            try:
                with open(filter_file_to_use, 'r') as f_check:
                    if not json.load(f_check): # Check if file is empty list
                        test_logger.warning(f"Filter file '{filter_file_to_use}' is empty. Skipping 'scrape_filtered_details' as there's nothing to scrape.")
                        # Optionally delete the empty filter file if it causes issues for re-runs
                        # filter_file_to_use.unlink() 
                    else:
                        scrape_details_params = ScrapeFilteredDetailsParams(
                            output_directory=str(test_base_output_dir),
                            filter_input_filename="test_filter_mnt_specific_bonds.json", # Must match filename from filter_securities
                            max_securities_to_process_from_filter=2 # Limit for testing; None for all
                        )
                        scrape_details_action_input_dict = OTCMNScraperActionInput(
                            action="scrape_filtered_details",
                            parameters=scrape_details_params,
                            cdp_endpoint_url=cdp_arg # CDP might be needed again
                        ).model_dump(mode='json')

                        test_logger.info(f"Tool input for 'scrape_filtered_details': {json.dumps(scrape_details_action_input_dict, indent=2)}")
                        result_details_str = await tool_instance.arun(tool_input=scrape_details_action_input_dict)
                        
                        test_logger.info(f"'scrape_filtered_details' action raw result string:\n{result_details_str}")
                        try:
                            result_details_json = json.loads(result_details_str)
                            test_logger.info(f"'scrape_filtered_details' action parsed JSON result:\n{json.dumps(result_details_json, indent=2, ensure_ascii=False)}")
                            if result_details_json.get("status") != "success" or result_details_json.get("errors_by_isin"):
                                test_logger.error(f"'scrape_filtered_details' action reported status '{result_details_json.get('status')}' or errors: {result_details_json.get('errors_by_isin')}")
                        except json.JSONDecodeError:
                            test_logger.error(f"Failed to parse 'scrape_filtered_details' result as JSON: {result_details_str}")
            except json.JSONDecodeError:
                 test_logger.error(f"Filter file '{filter_file_to_use}' seems to be invalid JSON. Skipping 'scrape_filtered_details'.")
            except Exception as e_read_filter:
                 test_logger.error(f"Error reading filter file '{filter_file_to_use}': {e_read_filter}. Skipping 'scrape_filtered_details'.")

    except Exception as e:
        test_logger.error(f"An unexpected error occurred during the OTCMNScraperTool test run: {e}", exc_info=True)
    finally:
        if tool_instance:
            await tool_instance.close() # Though close currently does little, it's good practice for future resource cleanup
        test_logger.info(f"--- OTCMNScraperTool Standalone Test Runner Finished ---")


if __name__ == "__main__":
    # To run this test from the directory containing the tool file (e.g., your_project_root/src/tools/):
    # Example: python otcmn_scraper_tool_file.py 
    # Or with CDP: python otcmn_scraper_tool_file.py ws://127.0.0.1:9222/devtools/browser/your-cdp-id
    # Ensure your PYTHONPATH is set up correctly if src.otcmn_interaction is not found.
    # e.g., export PYTHONPATH="${PYTHONPATH}:/path/to/your_project_root" (if running from outside src)
    # or run as a module from project root: python -m src.tools.otcmn_scraper_tool_file <cdp_arg_if_any>
    
    # Python's asyncio.run is a good way to run the main async function
    asyncio.run(main_test_otcmn_scraper_tool())