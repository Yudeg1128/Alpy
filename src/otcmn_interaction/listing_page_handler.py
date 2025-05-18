# src/otcmn_interaction/listing_page_handler.py

import asyncio
import logging
import re
from typing import List, Dict, Any, Optional, Literal, Tuple
from pathlib import Path # For screenshot paths if used

from playwright.async_api import Page, Locator, TimeoutError as PlaywrightTimeoutError, expect

from .common import (
    BasePageHandler, OtcmInteractionError, ElementNotFoundError, 
    DataExtractionError, PageStateError, ISIN_COLUMN_HEADER_TEXT
)


class ListingPageHandler(BasePageHandler):
    # --- Selectors for Listing Page ---
    MAIN_TABS_CONTAINER_SELECTOR = "div.ant-tabs-nav > div.ant-tabs-nav-wrap > div.ant-tabs-nav-list"
    BOARD_FILTERS_CONTAINER_SELECTOR = "div.ant-flex.css-m1bb5c.flex-wrap:has(p:text-is('Board:'))"
    BOARD_FILTER_TAG_SELECTOR_TEMPLATE = "span.ant-tag-checkable:has(span:text-is('{}'))"
    ALL_BOARD_FILTER_TAGS_SELECTOR = BOARD_FILTERS_CONTAINER_SELECTOR + " > span.ant-tag-checkable:has(span)"
    ACTIVE_BOARD_FILTER_CLASS = "ant-tag-checkable-checked"

    SECURITIES_TABLE_WRAPPER_SELECTOR = "div.ant-table-wrapper.custom-table"
    # This selector should point directly to the <table> HTML element
    SECURITIES_TABLE_SELECTOR = (
        SECURITIES_TABLE_WRAPPER_SELECTOR +
        " > div.ant-spin-nested-loading > div.ant-spin-container > div.ant-table" +
        " > div.ant-table-container > div.ant-table-content > table"
    )
    # Selectors relative to a table_loc (which is the <table> element)
    TABLE_HEADER_ROW_SELECTOR = "thead.ant-table-thead > tr" 
    TABLE_HEADER_CELL_SELECTOR = "th.ant-table-cell" 
    TABLE_DATA_ROW_SELECTOR = "tbody.ant-table-tbody > tr.ant-table-row" # Ensure this matches rows with data
    TABLE_DATA_CELL_IN_ROW_SELECTOR = "td.ant-table-cell"
    DETAIL_LINK_IN_ROW_SELECTOR_TEMPLATE = "a[href*='/securities/detail?id=']"

    PAGINATION_UL_SELECTOR = "ul.ant-pagination"
    PAGINATION_NEXT_LI_SELECTOR = PAGINATION_UL_SELECTOR + " > li.ant-pagination-next"
    PAGINATION_PREV_LI_SELECTOR = PAGINATION_UL_SELECTOR + " > li.ant-pagination-prev"
    PAGINATION_ITEM_LI_SELECTOR_TEMPLATE = PAGINATION_UL_SELECTOR + " > li.ant-pagination-item-{}"
    PAGINATION_ACTIVE_ITEM_SELECTOR = PAGINATION_UL_SELECTOR + " > li.ant-pagination-item-active"
    PAGINATION_ITEM_LINK_CLASS = "ant-pagination-item-link"
    PAGINATION_ITEM_WITH_TEXT_TEMPLATE = PAGINATION_UL_SELECTOR + " > li.ant-pagination-item:has(a:text-is('{}'))" # More robust for page "1"

    LOADING_SPINNER_SELECTOR = "div.ant-spin-spinning" # General Ant Design spinner

    # --- Mappings ---
    BOARD_CHAR_TO_SITE_TEXT_MAP: Dict[str, str] = {"A": "A", "B": "B", "C": "C"}
    MAIN_TAB_KEY_TO_SITE_TEXT_MAP: Dict[str, str] = {"primary": "Primary", "secondary": "Secondary"}

    def __init__(self, page: Page, logger: logging.Logger, default_timeout: float):
        super().__init__(page, logger, default_timeout)
        self._current_isin_column_index: Optional[int] = None
        self._current_headers: Optional[List[str]] = None
        self._last_selected_main_tab_key: Optional[str] = None
        self._last_selected_board_char: Optional[str] = None

    async def _wait_for_table_update(self, context_msg: str = "table to update"):
        self._logger.debug(f"Starting _wait_for_table_update for '{context_msg}'.")
        
        spinner_loc = self._page.locator(self.LOADING_SPINNER_SELECTOR).first
        try:
            await spinner_loc.wait_for(state="visible", timeout=1500) # Increased slightly
            self._logger.debug(f"Spinner detected for '{context_msg}'. Waiting for it to hide.")
            await spinner_loc.wait_for(state="hidden", timeout=self._default_timeout)
            self._logger.debug(f"Spinner hidden for '{context_msg}'.")
        except PlaywrightTimeoutError:
            self._logger.debug(f"Spinner not prominently active or timed out for '{context_msg}'. This might be normal.")
        
        table_element_loc_check = self._page.locator(self.SECURITIES_TABLE_SELECTOR).first
        try:
            await self._wait_for_locator(
                table_element_loc_check,
                state="attached",
                description="main table element (<table>) to be attached",
                timeout_override=self._default_timeout / 2 
            )
            self._logger.debug("Main table element (<table>) is attached.")
        except ElementNotFoundError as e:
            self._logger.error(f"Main table element (<table>) not found/attached after action for '{context_msg}'. Error: {e}")
            # await self._page.screenshot(path=f"debug_table_not_attached_{context_msg.replace(' ', '_')}.png")
            raise DataExtractionError(f"Main table element not found after '{context_msg}'.") from e

        # JS Selectors for the function - ensure SECURITIES_TABLE_SELECTOR points to the <table>
        # Escape single quotes in the selector strings for JS compatibility
        table_sel_js = self.SECURITIES_TABLE_SELECTOR.replace("'", "\\'")
        header_cell_sel_js_relative = "thead > tr > th" # Relative to the table element
        data_row_sel_js_relative = "tbody > tr.ant-table-row" # Relative to the table element

        js_condition = f"""
        (() => {{
            const tableElement = document.querySelector('{table_sel_js}');
            if (!tableElement) {{ console.warn('JS Check: Table element not found with selector: {table_sel_js}'); return false; }}

            const headerCell = tableElement.querySelector('{header_cell_sel_js_relative}');
            const dataRow = tableElement.querySelector('{data_row_sel_js_relative}');
            
            const isRenderedAndVisible = el => {{
                if (!el) return false;
                const style = window.getComputedStyle(el);
                if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return false;
                if (el.offsetParent === null) return false; 
                const rect = el.getBoundingClientRect();
                return rect.width > 0 && rect.height > 0;
            }};
            
            const headerVisible = headerCell && isRenderedAndVisible(headerCell);
            const dataRowVisible = dataRow && isRenderedAndVisible(dataRow);
            // console.log('JS Check: headerVisible=' + headerVisible + ', dataRowVisible=' + dataRowVisible);
            return headerVisible || dataRowVisible;
        }})()
        """
        try:
            await self._page.wait_for_function(js_condition, timeout=self._default_timeout)
            self._logger.info(f"JS check: Confirmed either first header cell or first data row is rendered and visible for '{context_msg}'. Table assumed ready.")
        except PlaywrightTimeoutError as pte:
            self._logger.error(f"Timeout (via JS check) waiting for EITHER first header cell OR first data row to become visible for '{context_msg}'.")
            # Debugging if JS check fails:
            # html_dump = await self._page.content()
            # self._logger.debug(f"Page HTML at JS timeout ({context_msg}):\n{html_dump[:2000]}...") # Log beginning of HTML
            # await self._page.screenshot(path=f"debug_table_content_js_fail_{context_msg.replace(' ', '_')}.png")
            raise DataExtractionError(f"JS check: Neither header cell nor data row became visible after '{context_msg}'.") from pte
        
        self._logger.debug(f"_wait_for_table_update for '{context_msg}' completed successfully.")

    async def verify_initial_elements(self) -> None:
        self._logger.info("Verifying initial elements on securities listing page...")
        await self._wait_for_locator(self._page.locator(self.MAIN_TABS_CONTAINER_SELECTOR).first, description="Main tabs container")
        await self._wait_for_locator(self._page.locator(self.BOARD_FILTERS_CONTAINER_SELECTOR).first, description="Board filters container")
        await self._wait_for_locator(self._page.locator(self.SECURITIES_TABLE_WRAPPER_SELECTOR).first, description="Securities table wrapper")
        primary_tab_text = self.MAIN_TAB_KEY_TO_SITE_TEXT_MAP.get("primary", "Primary")
        await self._wait_for_locator(
            self._page.locator(self.MAIN_TABS_CONTAINER_SELECTOR).get_by_role('tab', name=re.compile(f"^{re.escape(primary_tab_text)}$", re.I)).first,
            description=f"'{primary_tab_text}' tab button"
        )
        board_a_text = self.BOARD_CHAR_TO_SITE_TEXT_MAP.get("A", "A")
        await self._wait_for_locator(
             self._page.locator(self.BOARD_FILTERS_CONTAINER_SELECTOR).locator(self.BOARD_FILTER_TAG_SELECTOR_TEMPLATE.format(board_a_text)).first,
             description=f"Board '{board_a_text}' filter tag"
        )
        self._logger.info("Initial listing page elements verified.")

    def reset_header_cache(self) -> None:
        self._logger.debug("Resetting table header cache.")
        self._current_headers = None
        self._current_isin_column_index = None

    async def select_main_tab(self, tab_key: Literal["primary", "secondary"]) -> None:
        site_text = self.MAIN_TAB_KEY_TO_SITE_TEXT_MAP.get(tab_key)
        if not site_text:
            raise ValueError(f"Invalid tab_key: {tab_key}. Must be 'primary' or 'secondary'.")

        self._logger.info(f"Attempting to select main tab: '{site_text}' (key: {tab_key}).")
        
        # If already on this tab, and the board context has also been set (meaning not the first action for this tab)
        # then no action is needed here. The _go_to_first_page_of_listing will be handled by the orchestrator if necessary.
        if tab_key == self._last_selected_main_tab_key:
            self._logger.info(f"Main tab '{tab_key}' is already considered the active selection context. No click performed by select_main_tab.")
            # However, if the board context IS None, it means this is the first board being processed for this already-active tab
            # In this case, we still need to ensure the table is updated and headers are fresh.
            if self._last_selected_board_char is None:
                self._logger.debug(f"Tab '{tab_key}' was active, but board context is fresh. Ensuring table update and resetting headers.")
                await self._wait_for_table_update(f"content for already active tab '{site_text}' (board context fresh)")
                self.reset_header_cache()
            return 

        tab_button_loc = self._page.locator(self.MAIN_TABS_CONTAINER_SELECTOR).get_by_role('tab', name=re.compile(f"^{re.escape(site_text)}$", re.I)).first
        
        try:
            await self._wait_for_locator(tab_button_loc, description=f"'{site_text}' tab button to be attached/visible")
        except ElementNotFoundError:
            self._logger.error(f"Tab button for '{site_text}' not found.")
            raise # Re-raise to stop the process

        # Check if already active, though the top check should mostly cover this.
        # This is more about avoiding an unnecessary click if possible.
        is_active_now = await tab_button_loc.get_attribute("aria-selected") == "true"
        
        if not is_active_now:
            await self._safe_click(tab_button_loc, description=f"'{site_text}' tab button")
            # Verify it became active using Playwright's expect for robustness
            try:
                await expect(tab_button_loc).to_have_attribute("aria-selected", "true", timeout=self._default_timeout / 2)
                self._logger.info(f"Clicked and confirmed tab '{site_text}' is active.")
            except PlaywrightTimeoutError as e_expect:
                self._logger.error(f"Failed to confirm tab '{site_text}' became active after click: {e_expect}")
                # await self._page.screenshot(path=f"debug_tab_not_active_{tab_key}.png")
                raise PageStateError(f"Tab '{site_text}' did not become active after click.") from e_expect
        else:
            self._logger.info(f"Main tab '{site_text}' was already active (re-confirmed).")

        # Whether clicked or already active, ensure table is updated for this tab state
        await self._wait_for_table_update(f"content after selecting/confirming tab '{site_text}'")
        self.reset_header_cache() # Always reset headers on any effective tab change or confirmation
        
        self._last_selected_main_tab_key = tab_key
        self._last_selected_board_char = None # Reset board context as we've switched/confirmed tab

    async def select_board_filter_exclusively(self, board_char_to_select: Literal["A", "B", "C"]) -> None:
        target_site_text = self.BOARD_CHAR_TO_SITE_TEXT_MAP.get(board_char_to_select)
        if not target_site_text:
            raise ValueError(f"Invalid board_char_to_select: {board_char_to_select}. Must be 'A', 'B', or 'C'.")

        self._logger.info(f"Attempting to exclusively select board filter: '{target_site_text}' for tab '{self._last_selected_main_tab_key}'.")

        # If this board is already selected for the current tab, no action needed here.
        if self._last_selected_board_char == board_char_to_select:
             self._logger.info(f"Board filter '{target_site_text}' is already selected for current tab. Skipping clicks.")
             # Even if already selected, ensure table is consistent if this is called fresh
             await self._wait_for_table_update(f"content for already selected board '{target_site_text}'")
             # Headers would have been set on initial selection, no reset needed if truly no change.
             return

        filters_container_loc = self._page.locator(self.BOARD_FILTERS_CONTAINER_SELECTOR)
        await self._wait_for_locator(filters_container_loc, description="Board filters container")

        board_filter_chars_to_manage = ["A", "B", "C"]
        action_taken_on_filters = False

        # Phase 1: Deselect other active A, B, C tags that are NOT the target
        for board_char_in_loop in board_filter_chars_to_manage:
            if board_char_in_loop == board_char_to_select:
                continue
            current_tag_site_text = self.BOARD_CHAR_TO_SITE_TEXT_MAP.get(board_char_in_loop)
            current_tag_loc = filters_container_loc.locator(self.BOARD_FILTER_TAG_SELECTOR_TEMPLATE.format(current_tag_site_text))
            try:
                # Check if the tag is attached before trying to get its class or click
                await current_tag_loc.wait_for(state="attached", timeout=1000) # Short timeout is okay
                tag_class_attribute = await current_tag_loc.get_attribute("class") or ""
                is_active = self.ACTIVE_BOARD_FILTER_CLASS in tag_class_attribute
                if is_active:
                    self._logger.debug(f"Board filter '{current_tag_site_text}' is active and not target. Clicking to deactivate.")
                    await self._safe_click(current_tag_loc, description=f"board filter '{current_tag_site_text}' (to deactivate)")
                    action_taken_on_filters = True
                    await self._page.wait_for_timeout(300) # Brief pause for UI, AntD sometimes needs it
            except PlaywrightTimeoutError: # Tag not found/attached quickly, assume it's not there or not relevant to deselect
                self._logger.debug(f"Board filter tag '{current_tag_site_text}' not found or not attached during deselect phase. Skipping.")
            except Exception as e_deselect: # Catch other errors during deselect
                 self._logger.error(f"Error de-selecting board '{current_tag_site_text}': {e_deselect}", exc_info=False)


        # Phase 2: Ensure the target tag is active
        target_tag_loc = filters_container_loc.locator(self.BOARD_FILTER_TAG_SELECTOR_TEMPLATE.format(target_site_text))
        try:
            await target_tag_loc.wait_for(state="attached", timeout=self._default_timeout / 4) # Wait a bit longer for target tag
        except PlaywrightTimeoutError as e:
            # await self._page.screenshot(path=f"debug_target_board_tag_not_found_{target_site_text}.png")
            raise ElementNotFoundError(f"Target board filter tag '{target_site_text}' not found/attached.") from e
        
        target_tag_class_attribute = await target_tag_loc.get_attribute("class") or ""
        is_target_active = self.ACTIVE_BOARD_FILTER_CLASS in target_tag_class_attribute
        
        if not is_target_active:
            self._logger.debug(f"Target board filter '{target_site_text}' not active, clicking to activate.")
            await self._safe_click(target_tag_loc, description=f"board filter '{target_site_text}' (to activate)")
            action_taken_on_filters = True
        else:
            self._logger.debug(f"Target board filter '{target_site_text}' was already active.")

        # If any filter was clicked OR if the board we intend to select is different from the last one
        if action_taken_on_filters or self._last_selected_board_char != board_char_to_select:
            self._logger.info(f"Board filter selection changed to '{target_site_text}' or context requires update. Waiting for table.")
            await self._wait_for_table_update(f"content after setting filter to '{target_site_text}'")
            self.reset_header_cache() 
            # _go_to_first_page_of_listing will be called by the orchestrator (_handle_scrape_boards_action)
        else:
            self._logger.info(f"No filter clicks performed for board '{target_site_text}', and board context is the same. Assumed correct state.")

        # Final Verification (important)
        final_target_class_check = await target_tag_loc.get_attribute("class") or ""
        if not (self.ACTIVE_BOARD_FILTER_CLASS in final_target_class_check):
            # await self._page.screenshot(path=f"debug_target_not_active_final_{target_site_text}.png")
            raise PageStateError(f"VERIFICATION FAILED: Target board filter '{target_site_text}' NOT active after process. Class: '{final_target_class_check}'")
        self._logger.debug(f"Final check: Target '{target_site_text}' is confirmed active.")

        for board_char_in_loop_verify in board_filter_chars_to_manage:
            if board_char_in_loop_verify == board_char_to_select: continue
            other_tag_site_text = self.BOARD_CHAR_TO_SITE_TEXT_MAP.get(board_char_in_loop_verify)
            other_tag_loc = filters_container_loc.locator(self.BOARD_FILTER_TAG_SELECTOR_TEMPLATE.format(other_tag_site_text))
            try:
                await other_tag_loc.wait_for(state="attached", timeout=1000)
                other_tag_class = await other_tag_loc.get_attribute("class") or ""
                if self.ACTIVE_BOARD_FILTER_CLASS in other_tag_class:
                    # await self._page.screenshot(path=f"debug_other_still_active_{other_tag_site_text}_when_target_is_{target_site_text}.png")
                    raise PageStateError(f"VERIFICATION FAILED: Filter '{other_tag_site_text}' IS STILL ACTIVE with target '{target_site_text}'. Class: '{other_tag_class}'")
            except PlaywrightTimeoutError: self._logger.debug(f"Verify: Filter '{other_tag_site_text}' not found during final check, assuming inactive.")
            
        self._logger.info(f"Successfully selected and verified board filter: '{target_site_text}'.")
        self._last_selected_board_char = board_char_to_select

    async def _extract_headers_with_playwright(self, table_loc: Locator) -> Optional[List[str]]:
        self._logger.debug("Attempting header extraction with Playwright locators...")
        try:
            header_row_loc = table_loc.locator(self.TABLE_HEADER_ROW_SELECTOR).first
            await header_row_loc.wait_for(state="attached", timeout=3000) # Quick check if <tr> is there
            
            header_cells_locators = await header_row_loc.locator(self.TABLE_HEADER_CELL_SELECTOR).all()
            if not header_cells_locators:
                self._logger.warning("Playwright: No header cells (th) found using locators.")
                return None

            # Check visibility of the first cell - if this fails, Playwright method fails.
            await self._wait_for_locator(header_cells_locators[0], 
                                         state="visible", 
                                         description="Playwright: First header cell visibility", 
                                         timeout_override=5000)

            headers_list: List[str] = []
            for cell_loc in header_cells_locators:
                text_parts = await cell_loc.all_inner_texts()
                full_text = " ".join(part.strip() for part in text_parts if part.strip()).strip()
                headers_list.append(full_text)
            
            if not headers_list or not any(h for h in headers_list):
                self._logger.warning("Playwright: Header list is empty or all headers are blank.")
                return None # Indicate failure to get meaningful headers
            
            self._logger.info(f"Playwright: Successfully extracted headers: {headers_list}")
            return headers_list
        except (PlaywrightTimeoutError, ElementNotFoundError) as pte: # Catch specific errors
            self._logger.warning(f"Playwright: Timeout/ElementNotFound during header extraction: {str(pte).splitlines()[0]}")
            return None
        except Exception as e: # Catch any other unexpected error
            self._logger.error(f"Playwright: Unexpected error during header extraction: {e}", exc_info=True)
            return None

    async def _extract_headers_with_javascript(self) -> Optional[List[str]]:
        self._logger.info("Attempting header extraction with JavaScript fallback.")
        
        # Ensure table selector used in JS is specific enough
        table_sel_js = self.SECURITIES_TABLE_SELECTOR.replace("'", "\\'")
        header_cell_sel_js_relative = "thead > tr > th.ant-table-cell" # Be specific about 'th'

        js_script = f"""
        (() => {{
            const table = document.querySelector('{table_sel_js}');
            if (!table) {{
                // console.warn('JS Fallback: Table not found with selector: {table_sel_js}');
                return null;
            }}
            const headerRow = table.querySelector('thead > tr');
            if (!headerRow) {{
                // console.warn('JS Fallback: Header row (thead > tr) not found in table.');
                return null;
            }}
            const headerCells = Array.from(headerRow.querySelectorAll('{header_cell_sel_js_relative}'));
            if (!headerCells || headerCells.length === 0) {{
                // console.warn('JS Fallback: No header cells (th) found in header row.');
                return []; // Return empty list if no cells
            }}
            // console.log('JS Fallback: Found ' + headerCells.length + ' header cells.');
            return headerCells.map(th => th.innerText.trim());
        }})()
        """
        try:
            headers = await self._page.evaluate(js_script)
            
            if headers is None:
                self._logger.error("JavaScript Fallback: querySelector for table or header row returned null (elements not found by JS).")
                return None
            if not isinstance(headers, list):
                self._logger.error(f"JavaScript Fallback: Header extraction returned non-list type: {type(headers)}")
                return None
            # If headers is an empty list, it means JS found table & header row but no 'th' cells.
            
            self._logger.info(f"JavaScript Fallback: Extracted headers: {headers}")
            return headers
        except Exception as e:
            self._logger.error(f"JavaScript Fallback: Error during JS header extraction: {e}", exc_info=True)
            return None

    async def _extract_headers(self, table_loc: Locator) -> None:
        self._logger.info(f"Initiating header extraction for table: {table_loc}")
        
        # Attempt 1: Playwright locators
        headers = await self._extract_headers_with_playwright(table_loc)

        # Attempt 2: JavaScript fallback if Playwright failed
        if headers is None or not any(h.strip() for h in headers): # Check if list is None or all headers are blank
            self._logger.warning("Playwright header extraction failed or yielded unusable headers. Trying JavaScript fallback.")
            headers = await self._extract_headers_with_javascript()

        if headers is None or not any(h.strip() for h in headers): # Both methods failed
            self._logger.error("CRITICAL: Failed to extract usable headers using both Playwright and JavaScript methods.")
            self._current_headers = [] 
            self._current_isin_column_index = None
            # Taking a screenshot here might be useful if this state is reached
            # await self._page.screenshot(path="debug_critical_header_extraction_failure.png")
            raise DataExtractionError("Failed to extract table headers after multiple attempts (Playwright & JS).")

        self._current_headers = [h.strip() for h in headers] # No need to filter empty strings here if JS/Playwright already did
        if not self._current_headers: # Should be caught by the check above, but defensive
             raise DataExtractionError("All extracted headers are blank or empty strings after processing.")
        self._logger.info(f"Final successfully extracted headers: {self._current_headers}")

        try:
            self._current_isin_column_index = self._current_headers.index(ISIN_COLUMN_HEADER_TEXT)
            self._logger.info(f"'{ISIN_COLUMN_HEADER_TEXT}' column found at index {self._current_isin_column_index}.")
        except ValueError:
            self._current_isin_column_index = None
            self._logger.warning(f"CRITICAL WARNING: '{ISIN_COLUMN_HEADER_TEXT}' column NOT found in extracted headers: {self._current_headers}.")

    async def extract_current_page_data(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        self._logger.info("Extracting data from current listing page...")
        
        # _wait_for_table_update (called by tab/board selection) should ensure table is semantically ready.
        # Now, get a fresh locator for the table.
        final_table_loc = self._page.locator(self.SECURITIES_TABLE_SELECTOR).first
        
        try:
            await self._wait_for_locator(final_table_loc, 
                                         state="visible", 
                                         description="Main table element (final_table_loc) to be visible", 
                                         timeout_override=10000) # 10s for table to be visible
            self._logger.debug("Main securities table element (final_table_loc) is confirmed visible for data extraction.")
        except ElementNotFoundError as e:
            self._logger.error(f"Main table element not visible when starting data extraction. Cannot proceed. Error: {e}")
            # await self._page.screenshot(path="debug_extract_current_page_data_table_not_visible.png")
            return [], [] 

        if self._current_headers is None: # Headers not cached or reset
            await self._extract_headers(final_table_loc) 
        
        # Check if _extract_headers failed critically (it raises DataExtractionError now)
        # or resulted in empty headers (which it also handles by setting self._current_headers to [])
        if not self._current_headers: # If still None or empty list after trying to extract
            self._logger.warning("No headers were extracted or available. Cannot process rows.")
            return [], [] # Return empty headers and empty row list
        
        data_rows_locs = await final_table_loc.locator(self.TABLE_DATA_ROW_SELECTOR).all()
        if not data_rows_locs:
            self._logger.info("No data rows found on the current page using selector.")
            return self._current_headers, []

        parsed_rows_data: List[Dict[str, Any]] = []
        self._logger.debug(f"Processing {len(data_rows_locs)} potential data rows...")
        for i, row_loc in enumerate(data_rows_locs):
            if "ant-table-placeholder" in (await row_loc.get_attribute("class") or ""):
                self._logger.debug(f"Row {i}: Skipping placeholder row.")
                continue
            
            # Ensure row is visible before attempting to extract data from it
            try:
                await self._wait_for_locator(row_loc, state="visible", description=f"data row {i} to be visible", timeout_override=2000) # Quick check
            except ElementNotFoundError:
                self._logger.warning(f"Row {i} not visible, skipping row data extraction.")
                continue

            row_data = await self._extract_row_data(row_loc, self._current_headers) 
            if row_data:
                parsed_rows_data.append(row_data)
        
        self._logger.info(f"Successfully extracted {len(parsed_rows_data)} data rows from the page.")
        return self._current_headers, parsed_rows_data

    async def _extract_row_data(self, row_loc: Locator, headers: List[str]) -> Optional[Dict[str, Any]]:
        # This method logic seems okay, ensure headers param is used.
        cell_locs = await row_loc.locator(self.TABLE_DATA_CELL_IN_ROW_SELECTOR).all()
        
        if len(cell_locs) != len(headers):
            self._logger.warning(f"Row data cell count ({len(cell_locs)}) mismatch with header count ({len(headers)}). Skipping row. HTML: {await row_loc.inner_html(timeout=1000)}")
            return None

        row_values: List[str] = []
        for i, cell_loc in enumerate(cell_locs):
            cell_texts_list = await cell_loc.all_inner_texts()
            cell_text = " ".join(t.strip() for t in cell_texts_list).strip()
            row_values.append(cell_text)

        row_dict = dict(zip(headers, row_values))

        first_cell_with_link = row_loc.locator(f"{self.TABLE_DATA_CELL_IN_ROW_SELECTOR}:has({self.DETAIL_LINK_IN_ROW_SELECTOR_TEMPLATE})").first
        detail_link_loc = first_cell_with_link.locator(self.DETAIL_LINK_IN_ROW_SELECTOR_TEMPLATE).first
        
        detail_url_path = await self._get_attribute(detail_link_loc, "href", "detail link in row", default=None)
        if detail_url_path:
            row_dict["_detail_url_path"] = detail_url_path
            match_id = re.search(r"id=([a-f0-9-]+)", detail_url_path, re.IGNORECASE)
            if match_id: row_dict["_url_id"] = match_id.group(1)

        if self._current_isin_column_index is not None and self._current_isin_column_index < len(headers):
            isin_header_on_site = headers[self._current_isin_column_index]
            isin_value_from_row = row_dict.get(isin_header_on_site, "").strip()
            row_dict[ISIN_COLUMN_HEADER_TEXT] = isin_value_from_row if isin_value_from_row else None
        elif ISIN_COLUMN_HEADER_TEXT in row_dict:
            row_dict[ISIN_COLUMN_HEADER_TEXT] = str(row_dict[ISIN_COLUMN_HEADER_TEXT]).strip() or None
        else:
            row_dict[ISIN_COLUMN_HEADER_TEXT] = None # Ensure key exists even if ISIN not found

        return row_dict

    async def go_to_next_page(self) -> bool:
        # Using the last provided version of go_to_next_page that includes JS check and JS click fallback
        self._logger.info("Attempting to go to the next page...")

        pagination_ul_sel_js = self.PAGINATION_UL_SELECTOR.replace("'", "\\'")
        next_button_li_sel_js_relative = "li.ant-pagination-next" 
        button_inside_li_sel_js_relative = "button.ant-pagination-item-link"

        js_next_button_ready_script = f"""
        (() => {{
            const results = {{ log: [], finalState: 'initial_fail' }};
            results.log.push('JS Check: Starting pagination check.');
            const paginationUl = document.querySelector('{pagination_ul_sel_js}');
            if (!paginationUl) {{ results.log.push('JS Check: Pagination UL not found'); results.finalState = 'ul_not_found'; return results; }}
            results.log.push('JS Check: Pagination UL found.');
            const nextButtonLi = paginationUl.querySelector('{next_button_li_sel_js_relative}');
            if (!nextButtonLi) {{ results.log.push('JS Check: Next Page LI not found'); results.finalState = 'li_not_found'; return results; }}
            results.log.push('JS Check: Next Page LI found. ClassList: ' + nextButtonLi.classList);
            results.log.push('JS Check: Next Page LI aria-disabled: ' + nextButtonLi.getAttribute('aria-disabled'));
            if (nextButtonLi.classList.contains('ant-pagination-disabled') || nextButtonLi.getAttribute('aria-disabled') === 'true') {{
                results.log.push('JS Check: Next Page LI is marked as disabled.');
                results.finalState = 'disabled'; return results;
            }}
            results.log.push('JS Check: Next Page LI is NOT marked as disabled.');
            const actualButton = nextButtonLi.querySelector('{button_inside_li_sel_js_relative}');
            if (!actualButton) {{ results.log.push('JS Check: Actual button not found in LI'); results.finalState = 'button_in_li_not_found'; return results; }}
            results.log.push('JS Check: Actual button found. Disabled: ' + actualButton.disabled);
            const isVisibleAndInteractable = el => {{
                if (!el) return false; const style = window.getComputedStyle(el); let elLog = [];
                if (style.display === 'none') elLog.push('display_none');
                if (style.visibility === 'hidden') elLog.push('visibility_hidden');
                if (style.opacity === '0') elLog.push('opacity_zero');
                if (el.offsetParent === null) elLog.push('offsetParent_null');
                const rect = el.getBoundingClientRect();
                if (!(rect.width > 0)) elLog.push('width_zero');
                if (!(rect.height > 0)) elLog.push('height_zero');
                if (el.disabled) elLog.push('el_disabled');
                if (elLog.length > 0) {{ results.log.push('JS Check: Button visibility/interactability issues: ' + elLog.join(', ')); return false; }}
                return true;
            }};
            if (!isVisibleAndInteractable(actualButton)) {{ results.log.push('JS Check: Actual button failed isVisibleAndInteractable.'); results.finalState = 'button_not_interactable'; return results; }}
            results.log.push('JS Check: All checks passed. Button ready.'); results.finalState = 'ready'; return results;
        }})()
        """
        js_result_obj = None
        try:
            js_result_obj = await self._page.wait_for_function(js_next_button_ready_script, timeout=self._default_timeout / 2)
            js_eval_result = await js_result_obj.json_value() if js_result_obj else {"finalState": "js_eval_failed", "log": ["JS eval object was null."]}
            if js_eval_result and js_eval_result.get("log"):
                for log_entry in js_eval_result["log"]: self._logger.info(log_entry)
            button_state_from_js = js_eval_result.get("finalState", "unknown_js_state")
            if button_state_from_js == 'ready': self._logger.info("JS evaluation: Next page button confirmed ready.")
            elif button_state_from_js == 'disabled': self._logger.info("JS evaluation: Next page button is disabled."); return False
            else: self._logger.warning(f"JS evaluation: Next page button not ready (state: {button_state_from_js})."); return False
        except PlaywrightTimeoutError:
            self._logger.error(f"Timeout executing/waiting for JS pagination check."); return False
        finally:
            if js_result_obj: await js_result_obj.dispose()

        pagination_ul_loc = self._page.locator(self.PAGINATION_UL_SELECTOR).first
        next_button_actual_button_loc = pagination_ul_loc.locator(f"li.ant-pagination-next > button.ant-pagination-item-link")
        try:
            await self._wait_for_locator(next_button_actual_button_loc, state="visible", description="Next page button (for Playwright click)", timeout_override=5000)
            await next_button_actual_button_loc.click(timeout=5000)
            self._logger.info("Playwright click successful for next page.")
        except Exception as e_pw_click:
            self._logger.warning(f"Playwright click on next page button failed: {str(e_pw_click).splitlines()[0]}. Attempting JS click fallback.")
            js_click_script = f"""
            (() => {{
                const paginationUl = document.querySelector('{pagination_ul_sel_js}');
                if (!paginationUl) return false;
                const nextButton = paginationUl.querySelector('{next_button_li_sel_js_relative} {button_inside_li_sel_js_relative}');
                if (nextButton) {{ nextButton.click(); return true; }}
                return false;
            }})()"""
            try:
                click_success = await self._page.evaluate(js_click_script)
                if not click_success: self._logger.error("JS click on next page button also failed."); return False
                self._logger.info("JavaScript click successful for next page.")
            except Exception as e_js_click: self._logger.error(f"Error during JS click on next page: {e_js_click}"); return False
        
        await self._wait_for_table_update("content after clicking next page")
        self._logger.info("Successfully processed next page action.")
        self.reset_header_cache() # Reset headers after successful pagination
        return True

    async def _go_to_first_page_of_listing(self, context_msg: str = "resetting to page 1") -> bool:
        self._logger.info(f"Attempting to ensure page 1 of listing for '{context_msg}' using JS-centric approach.")

        js_pagination_ul_selector = self.PAGINATION_UL_SELECTOR.replace("'", "\\'")
        js_page_1_li_text_check_selector = "li.ant-pagination-item"
        js_page_1_li_active_class = "ant-pagination-item-active" # Just the class name
        js_page_1_button_click_selector_relative_to_li = "a"
        js_prev_button_li_selector_relative = "li.ant-pagination-prev"
        js_prev_button_disabled_class = "ant-pagination-disabled"

        # JS to check state and click Page 1 if needed
        js_action_script = f"""
        async (args) => {{
            const {{ ulSelector, page1ItemSelector, page1ButtonSelector, activeClass, prevButtonSelector, prevDisabledClass, logPrefix }} = args;
            const results = {{ log: [], finalState: 'initial_fail', clicked: false }};
            const log = (msg) => results.log.push(`${{logPrefix}} ${{msg}}`);

            log('Starting JS Page 1 action_script.');
            const paginationUl = document.querySelector(ulSelector);
            if (!paginationUl) {{
                log('Pagination UL not found.');
                results.finalState = 'ul_not_found_assuming_single_page';
                return results;
            }}
            log('Pagination UL found.');

            let page1Li = null;
            const candidateLis = paginationUl.querySelectorAll(page1ItemSelector);
            for (let i = 0; i < candidateLis.length; i++) {{
                const li = candidateLis[i];
                const aTag = li.querySelector('a');
                if (aTag && aTag.innerText.trim() === '1') {{ page1Li = li; break; }}
            }}

            if (!page1Li) {{
                log('Page 1 LI item not found by iteration.');
                const prevButtonLi = paginationUl.querySelector(prevButtonSelector);
                if (prevButtonLi && prevButtonLi.classList.contains(prevDisabledClass)) {{
                    results.finalState = 'effectively_page_1_by_prev_disabled';
                }} else {{ results.finalState = 'page_1_li_not_found_and_prev_not_conclusive'; }}
                return results;
            }}
            log('Page 1 LI found.');

            if (page1Li.classList.contains(activeClass)) {{
                log('Page 1 LI is already active.');
                results.finalState = 'already_active'; return results;
            }}
            log('Page 1 LI is not active. Attempting to click.');
            const actualButton = page1Li.querySelector(page1ButtonSelector);
            if (!actualButton) {{ log('Button (<a>) in Page 1 LI not found.'); results.finalState = 'page_1_button_not_found'; return results; }}
            
            const style = window.getComputedStyle(actualButton); const rect = actualButton.getBoundingClientRect();
            if (style.display === 'none' || style.visibility === 'hidden' || rect.width === 0 || rect.height === 0) {{
                 log('Page 1 button found but not visible/interactable by basic JS check.');
                 results.finalState = 'page_1_button_not_interactable_by_js'; return results;
            }}
            actualButton.click(); results.clicked = true;
            log('Page 1 button clicked via JS.');
            results.finalState = 'clicked_pending_verification'; return results;
        }}
        """
        js_args = {
            "ulSelector": self.PAGINATION_UL_SELECTOR,
            "page1ItemSelector": "li.ant-pagination-item", "page1ButtonSelector": "a",
            "activeClass": "ant-pagination-item-active", "prevButtonSelector": "li.ant-pagination-prev",
            "prevDisabledClass": "ant-pagination-disabled", "logPrefix": f"JS-Action ({context_msg}):"
        }

        eval_result_json = None
        try:
            self._logger.debug(f"Evaluating JS for Page 1 state/action for '{context_msg}'.")
            eval_result_json = await self._page.evaluate(js_action_script, js_args)
            if eval_result_json and eval_result_json.get("log"):
                for log_entry in eval_result_json["log"]: self._logger.info(log_entry)
            js_final_state = eval_result_json.get("finalState", "js_eval_error")
            js_clicked_page_1 = eval_result_json.get("clicked", False)
        except Exception as e_eval:
            self._logger.error(f"Error evaluating JS for Page 1 action: {e_eval}", exc_info=True); return False

        if js_final_state == 'already_active' or js_final_state == 'effectively_page_1_by_prev_disabled' or js_final_state == 'ul_not_found_assuming_single_page':
            self._logger.info(f"Page 1 confirmed/assumed active for '{context_msg}' (JS state: {js_final_state}).")
            await self._wait_for_table_update(f"table state for presumed page 1 ({context_msg})"); return True
        
        if not js_clicked_page_1: # Covers 'page_1_button_not_found', 'page_1_button_not_interactable_by_js', etc.
            self._logger.warning(f"JS did not/could not click Page 1 for '{context_msg}' (JS state: {js_final_state}). Reset failed."); return False

        # If JS reported it clicked, wait for table update, then verify Page 1 is active using another JS predicate
        self._logger.info(f"JS reports Page 1 was clicked for '{context_msg}'. Waiting for table update and verifying state via JS.")
        await self._wait_for_table_update(f"content after JS click on Page 1 for '{context_msg}'")

        js_verification_script = f"""
        (() => {{
            const paginationUl = document.querySelector('{js_pagination_ul_selector}');
            if (!paginationUl) return false; // Should not happen if first script found it
            let page1Li = null;
            const candidateLis = paginationUl.querySelectorAll('{js_page_1_li_text_check_selector.replace("'", "\\'")}');
            for (let i = 0; i < candidateLis.length; i++) {{
                const li = candidateLis[i]; const aTag = li.querySelector('a');
                if (aTag && aTag.innerText.trim() === '1') {{ page1Li = li; break; }}
            }}
            if (!page1Li) return false; // Page 1 LI disappeared after click?
            return page1Li.classList.contains('{js_page_1_li_active_class.replace("'", "\\'")}');
        }})()
        """
        verification_timeout = 10000 # 10 seconds for verification
        try:
            await self._page.wait_for_function(js_verification_script, timeout=verification_timeout)
            self._logger.info(f"JS VERIFIED: Page 1 is active after click for '{context_msg}'.")
            return True
        except PlaywrightTimeoutError:
            self._logger.error(f"JS VERIFICATION FAILED: Page 1 did not become active after JS click for '{context_msg}' within {verification_timeout}ms.")
            # await self._page.screenshot(path=f"debug_page1_verify_fail_{context_msg.replace(' ','_')}.png")
            return False
        except Exception as e_verify_js:
            self._logger.error(f"Error during JS verification of Page 1 active state: {e_verify_js}", exc_info=True)
            return False