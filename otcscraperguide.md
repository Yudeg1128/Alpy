# Alpy OTCMN Scraper: Further Implementation Guide

This guide outlines the strategy for completing the `OTCMNScraperTool` and its `otcmn_interaction` module, focusing on robust interaction with the dynamic `otc.mn` website.

## 1. Understanding the Current `otcmn_interaction` Module & Critical Interaction Strategy

The `otcmn_interaction` module currently consists of:
*   `common.py`: Shared constants, custom exceptions, and a (currently minimal) `BasePageHandler`.
*   `listing_page_handler.py`: Handles interactions with the main securities listing page (`/securities`). This includes tab selection, board filter selection, header/data extraction, and pagination.
*   `detail_page_handler.py`: (Stub) Intended to handle interactions with individual security detail pages.
*   `interactor.py`: The `OtcmSiteInteractor` class, which acts as a primary interface, holding instances of page-specific handlers.

**CRITICAL INTERACTION STRATEGY: PRIORITIZE JAVASCRIPT EXECUTION**

Experience with `otc.mn` has definitively shown that **standard Playwright Python locators for visibility and actionability checks are unreliable due to the site's highly dynamic JavaScript-driven nature.** Elements may be present in the DOM but fail Playwright's visibility/attached checks, or locators may become stale very quickly due to DOM re-rendering, even if no visual change is apparent.

**FOR ALL FURTHER DEVELOPMENT ON THIS WEBSITE, ESPECIALLY FOR THE `DetailPageHandler` AND ANY NEW INTERACTIONS:**

1.  **Primary Wait/Validation Strategy: `page.wait_for_function()` with JavaScript Predicates.**
    *   Before attempting *any* interaction (click, data extraction) or making *any* assertion about an element's state (visibility, active class, content), use `await self._page.wait_for_function(your_javascript_predicate_string, timeout=...)`.
    *   The JavaScript predicate **MUST** directly query the live browser DOM (e.g., using `document.querySelector`, `element.classList.contains`, `window.getComputedStyle`, `element.getBoundingClientRect`) to confirm the target element is in the desired state (e.g., present, visible with non-zero dimensions, enabled, has specific text/attributes).
    *   **This JS predicate is the source of truth for element readiness.**

2.  **Primary Action Strategy: JavaScript Execution (`page.evaluate()`) for Clicks.**
    *   Once the JS predicate from `page.wait_for_function()` confirms an element (e.g., a button, a link) is ready to be clicked, **perform the click using `await self._page.evaluate("selector => document.querySelector(selector).click()", your_css_selector_string)`**.
    *   This bypasses Playwright's Python-side click actionability checks, which have proven problematic. The preceding JS predicate serves as the actionability check.

3.  **Primary Data Extraction Strategy: JavaScript Execution (`page.evaluate()`).**
    *   After JS predicates confirm the data elements are present and stable, extract data (text, attributes) using `await self._page.evaluate(...)`.
    *   The JavaScript function passed to `evaluate` should perform all necessary `querySelector`, iteration, and text/attribute extraction, returning a serializable object (e.g., string, number, boolean, array, or simple dictionary).
    *   **Example:** For extracting table data, the JS would find the table, iterate rows/cells, and build an array of objects.

4.  **Playwright Locators as a Secondary/Targeting Aid (within JS):**
    *   You can still use Playwright's Python locators to *construct CSS selector strings* that are then passed into your JavaScript functions for `document.querySelector`. This can be useful if Playwright's selector engine helps create complex selectors more easily. However, the *execution* of that selector for state checking or action should primarily happen within the JS context.
    *   Avoid `await python_locator.is_visible()`, `await python_locator.text_content()`, `await python_locator.click()` as primary methods if they cause timeouts or flakiness. Use them only if they prove stable for a specific, simple element.

**Why this JS-first approach is mandated for `otc.mn`:**
    *   **Synchronization:** It minimizes issues arising from desynchronization between Playwright's Python state and the browser's rapidly changing DOM. JS executes directly on the live DOM.
    *   **Visibility Nuances:** Bypasses potential discrepancies in how Playwright's visibility algorithm interprets complex CSS and dynamic rendering compared to direct browser APIs.
    *   **Proven Efficacy:** The success in fixing header extraction and pagination relied heavily on shifting validation and actions (like clicks as a fallback) to JavaScript execution. Continue this pattern.

## 2. `OtcmSiteInteractor` and Tool Connection

*   **`OtcmSiteInteractor` (`interactor.py`):**
    *   Initialized by `OTCMNScraperTool` with the Playwright `Page` object.
    *   Holds instances of `ListingPageHandler` and (to be implemented) `DetailPageHandler`.
    *   Provides high-level navigation methods (e.g., `navigate_to_securities_listing_page`, `navigate_to_security_detail_page`).
    *   The tool calls these navigation methods first.
    *   Then, the tool gets the relevant handler (e.g., `listing_handler = self._site_interactor.listing_handler`) and calls its methods to perform page-specific actions (tab/filter selection, data extraction).

*   **`OTCMNScraperTool` (Tool file):**
    *   Sets up Playwright (CDP or new browser).
    *   Instantiates `OtcmSiteInteractor`.
    *   **Orchestrates the workflow:**
        *   For `scrape_boards`: Calls interactor to navigate, gets listing handler, loops through tabs/boards, calls handler methods for selection and data extraction, handles pagination calls on the handler, and calls `_go_to_first_page_of_listing` on the handler at strategic reset points.
        *   For `filter_securities` (No Playwright): Directly processes JSON files.
        *   For `scrape_filtered_details` (To be implemented): Will read a filter file, then for each security:
            *   Call interactor to navigate to the detail page (e.g., `await self._site_interactor.navigate_to_security_detail_page(security_url_id)`).
            *   Get the `detail_handler` from `self._site_interactor`.
            *   Call methods on `detail_handler` to extract all required information (general info, underwriter, transaction history, file list) and download files, adhering to the JS-first interaction strategy.

## 3. Further Steps to Complete the Scraper Tool

**A. Implement `DetailPageHandler` (`src/otcmn_interaction/detail_page_handler.py`)**

   *   **Goal:** Extract all relevant data from a security detail page (e.g., `https://otc.mn/securities/detail?id=...`) and download associated documents.
   *   **Methods (following JS-first strategy):**
        1.  `async def verify_initial_elements(self) -> None`:
            *   Uses `page.wait_for_function` to confirm key sections/elements of the detail page are present and visible (e.g., main security info card title, document download section title if always present).
        2.  `async def extract_security_general_info(self) -> Dict[str, Any]`:
            *   JS predicate to ensure the general info section (likely an Ant Design `Descriptions` or `Card`) is rendered.
            *   `page.evaluate()` to scrape key-value pairs from this section.
        3.  `async def extract_underwriter_info(self) -> Dict[str, Any]`: Similar to above.
        4.  `async def extract_transaction_history(self) -> List[Dict[str, Any]]`:
            *   JS predicate for table readiness (e.g., header + at least one data row, or "no data" placeholder).
            *   `page.evaluate()` to scrape the transaction history table into a list of dictionaries.
        5.  `async def extract_file_list_metadata(self) -> List[Dict[str, str]]`:
            *   JS predicate for the file list section/table.
            *   `page.evaluate()` to get a list of files, including their display names and crucially, the **actual `href` or data attributes needed to trigger downloads.**
            *   Return: `[{"file_name": "...", "download_href": "...", "original_site_filename": "..."}, ...]`
        6.  `async def download_document(self, file_info: Dict[str, str], download_dir: Path) -> Path`:
            *   Takes a single file dictionary from `extract_file_list_metadata`.
            *   **Crucial:** The download might be triggered by a standard `<a>` link click or by JS.
            *   **If standard link:**
                *   JS predicate to ensure the specific download link/button for this file is visible and interactable.
                *   Use `page.expect_download()` context manager.
                *   `page.evaluate()` to click the link/button.
                *   `await download.save_as(...)`.
            *   **If JS-triggered download:** May need to analyze network requests or replicate the JS action that initiates the download. This is more complex. Assume standard links first.
        7.  `async def download_all_documents_on_page(self, download_dir: Path) -> List[Tuple[str, Path]]`:
            *   Calls `extract_file_list_metadata()`.
            *   Loops through the metadata, calling `download_document()` for each.
            *   Manages concurrent downloads carefully if attempting (Playwright handles one download event per click well; multiple simultaneous JS-triggered clicks might be complex). Sequential downloads are safer.
    *   **Selectors:** Define CSS selectors for all target sections, tables, and elements. These will be used *within* the JavaScript snippets.

**B. Implement `_handle_scrape_filtered_details_action` (in `OTCMNScraperTool`)**

   *   Reads the specified filter file (e.g., `my_filter.json`).
   *   Loops through the securities in the filter file. For each security:
        1.  Extract `isin`, `detail_url_path` (or `url_id` to construct it), `board_category`.
        2.  Create the output directory: `output_directory/current/<board_category>/<ISIN>/`.
        3.  Call `await self._site_interactor.navigate_to_security_detail_page(url_id)`.
        4.  Get `detail_h = self._site_interactor.detail_handler`.
        5.  Call `await detail_h.extract_security_general_info()` and save to `security_details.json` (or a more comprehensive name).
        6.  Call `await detail_h.extract_underwriter_info()` and save to `underwriter_info.json`.
        7.  Call `await detail_h.extract_transaction_history()` and save to `transaction_history.json`.
        8.  Call `file_metadata_list = await detail_h.extract_file_list_metadata()` and save to `file_list.json`.
        9.  Create `documents_output_path = output_directory/current/<board_category>/<ISIN>/documents/`.
        10. Call `downloaded_files = await detail_h.download_all_documents_on_page(documents_output_path)` (this method would use `file_metadata_list` if passed, or re-extract it).
        11. Update `summary_log` with counts of details scraped, documents downloaded, and any errors for this ISIN.
   *   Handle errors gracefully for each security (e.g., if a detail page fails, log it and move to the next security).

**C. Complete Test Script (`main_test_otcmn_scraper_tool`)**

   1.  Ensure the `tool_input` argument fix for `arun` calls is applied for `filter_securities` and `scrape_filtered_details`.
   2.  Add a comprehensive test case for `scrape_filtered_details`:
        *   Requires `scrape_boards` and `filter_securities` to run first to generate a filter file.
        *   Processes a small number of securities from the filter file.
        *   Checks for the creation of expected JSON detail files and downloaded documents (even if just checking for file existence and non-zero size for documents initially).
   3.  Test `scrape_boards` with `max_listing_pages_per_board` set to `None` or a higher number (e.g., 3-4) to thoroughly test multi-page pagination for all boards that have enough data.

**D. Refinement and Error Handling**

   *   Thoroughly review all timeout values. Default timeouts might be too long or too short for certain operations.
   *   Ensure consistent error handling and logging across all methods. Custom exceptions from `common.py` should be used.
   *   Consider adding retry mechanisms for transient network issues or very brief element unavailability, but only after confirming the core logic and selectors are as robust as possible. Use with caution to avoid masking fundamental problems.

By adhering strictly to the "JavaScript-first" interaction strategy for `otc.mn`, especially for element validation and actions, the remaining implementation should be significantly more stable.