import logging
import asyncio
import re
from urllib.parse import unquote, urlparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError, ElementHandle

from .common import (
    BasePageHandler,
    OtcmInteractionError,
    DataExtractionError,
    ElementNotFoundError,
    PageStateError
)

class DetailPageHandler(BasePageHandler):
    # Selectors for section titles (used to find the parent container of each section)
    # These should be general enough to find the title span within its card/section header.
    _SECURITY_DETAILS_TITLE_TEXT = "Security Details"
    _UNDERWRITER_DETAILS_TITLE_TEXT = "Underwriter Details"
    _DOCUMENTS_TITLE_TEXT = "Documents"
    _RFQ_TABLE_TITLE_TEXT = "Available RFQs"
    _DISCLOSURES_TABLE_TITLE_TEXT = "Security disclosures"
    _TRANSACTION_HISTORY_TABLE_TITLE_TEXT = "Transaction History"

    # Timeout for individual element interactions or short waits
    _ELEMENT_WAIT_TIMEOUT_MS = 10000 # 10 seconds

    def __init__(self, page: Page, logger: logging.Logger, default_timeout: float):
        super().__init__(page, logger, default_timeout)
        self._logger.info("DetailPageHandler initialized.")

    async def _find_section_container_by_title(self, title_text: str) -> Optional[ElementHandle]:
        """
        Finds the main container div of a section by looking for a specific title text.
        The assumption is that the title is a direct child or grandchild of the section's
        primary container we want to operate within.
        Adjust the JS path based on observed HTML structure.
        """
        self._logger.debug(f"Attempting to find section container for title: '{title_text}'")
        js_predicate = f"""
        () => {{
            const titles = Array.from(document.querySelectorAll('span.ant-typography.font-bold, div.text-lg.font-bold.text-white'));
            for (const titleEl of titles) {{
                if (titleEl.textContent && titleEl.textContent.trim() === "{title_text}") {{
                    // Heuristic: Try to find a common ancestor that's likely the section card/block
                    // This might need adjustment based on actual consistent DOM structure.
                    // Common pattern: title is in a header, which is a child of the section block.
                    let container = titleEl.closest('.ant-space-item > div'); // Common pattern for key-value & docs
                    if (!container) container = titleEl.closest('.ant-space-item'); // For tables where title is in a Row
                    if (!container) container = titleEl.closest('div.ant-card'); // If it's an Ant Card
                    if (!container && titleEl.parentElement && titleEl.parentElement.parentElement) {{
                         // For tables, the title might be in a row, and the table wrapper is a sibling or nearby.
                         // The logic here is to find the `ant-space-item` that *contains* the table related to this title.
                         // This specific JS would go into the table extraction method.
                         // For general key-value or document sections, simpler logic suffices.
                         if (["{self._RFQ_TABLE_TITLE_TEXT}", "{self._DISCLOSURES_TABLE_TITLE_TEXT}", "{self._TRANSACTION_HISTORY_TABLE_TITLE_TEXT}"].includes("{title_text}")) {{
                            let currentElement = titleEl;
                            for (let i=0; i<5; i++) {{ // Search up to 5 levels for ant-space-item containing title
                                if (currentElement.classList && currentElement.classList.contains('ant-space-item')) break;
                                currentElement = currentElement.parentElement;
                                if (!currentElement) return null;
                            }}
                            // Now find the next ant-space-item sibling that contains the table
                            let tableSpaceItem = currentElement.nextElementSibling;
                            while(tableSpaceItem && (!tableSpaceItem.querySelector('div.ant-table-wrapper'))) {{
                                tableSpaceItem = tableSpaceItem.nextElementSibling;
                            }}
                            return tableSpaceItem; // This is the ant-space-item containing the table
                         }} else {{
                            container = titleEl.parentElement.parentElement; // A common fallback
                         }}
                    }}
                    return container;
                }}
            }}
            return null;
        }}
        """
        try:
            await self._page.wait_for_function(js_predicate, timeout=self._ELEMENT_WAIT_TIMEOUT_MS)
            container_handle = await self._page.evaluate_handle(js_predicate)
            if container_handle.as_element():
                self._logger.debug(f"Found container for section '{title_text}'.")
                return container_handle.as_element()
            self._logger.warning(f"Container for section '{title_text}' not found or handle is not an element.")
            return None
        except PlaywrightTimeoutError:
            self._logger.warning(f"Timeout waiting for section container with title '{title_text}'.")
            return None
        except Exception as e:
            self._logger.error(f"Error finding section container for '{title_text}': {e}", exc_info=True)
            return None

    async def verify_initial_elements(self) -> None:
        """
        Verifies that key sections of the detail page are present using their titles.
        """
        self._logger.info("Verifying initial elements on detail page...")
        required_section_titles = [
            self._SECURITY_DETAILS_TITLE_TEXT,
            self._UNDERWRITER_DETAILS_TITLE_TEXT, # Optional, might not always be there
            self._DOCUMENTS_TITLE_TEXT,
            self._TRANSACTION_HISTORY_TABLE_TITLE_TEXT # This table itself might be empty but title should exist
        ]
        # found_all_required = True
        for title in required_section_titles:
            js_predicate = f"""
            () => {{
                const titles = Array.from(document.querySelectorAll('span.ant-typography.font-bold, div.text-lg.font-bold.text-white'));
                return titles.some(el => el.textContent && el.textContent.trim() === "{title}");
            }}
            """
            try:
                await self._page.wait_for_function(js_predicate, timeout=self._default_timeout)
                self._logger.debug(f"Section title '{title}' verified.")
            except PlaywrightTimeoutError:
                if title == self._UNDERWRITER_DETAILS_TITLE_TEXT:
                    self._logger.warning(f"Optional section title '{title}' (Underwriter Details) not found. This is acceptable.")
                elif title == self._TRANSACTION_HISTORY_TABLE_TITLE_TEXT:
                    self._logger.info(f"Section title '{title}' (Transaction History) not found. This is acceptable (e.g., primary market bonds or no history).")
                # If _DOCUMENTS_TITLE_TEXT could also be optional, add an elif here:
                # elif title == self._DOCUMENTS_TITLE_TEXT:
                #     self._logger.warning(f"Section title '{title}' (Documents) not found. This may be acceptable.")
                else: # For other titles like _SECURITY_DETAILS_TITLE_TEXT (and _DOCUMENTS_TITLE_TEXT by default if not made optional)
                    self._logger.error(f"CRITICAL: Required section title '{title}' not found or not visible within timeout.")
                    raise PageStateError(f"Required section '{title}' not found on detail page. Cannot reliably extract data.")
        
        # if not found_all_required: # Should be caught by individual raises now
        #     raise PageStateError("One or more required sections not found on the detail page.")
        self._logger.info("Initial element presence check completed.")

    async def _extract_main_header_info(self) -> Dict[str, Any]:
        """Extracts top-level security name, market type, and status."""
        self._logger.debug("Extracting main header info...")
        header_data = {}
        try:
            # Security Name/Symbol
            name_symbol_js = """
            () => {
                const titleEl = document.querySelector('.ant-page-header-heading-title .ant-space-item');
                return titleEl ? titleEl.textContent.trim() : null;
            }
            """
            header_data["name_and_symbol_display"] = await self._page.evaluate(name_symbol_js)

            # Market Type (e.g., SECONDARY)
            market_type_js = """
            () => {
                const marketTagEl = document.querySelector('.ant-page-header-heading-title .ant-tag');
                return marketTagEl ? marketTagEl.textContent.trim() : null;
            }
            """
            header_data["market_type"] = await self._page.evaluate(market_type_js)

            # Status (e.g., ACTIVE)
            status_js = """
            () => {
                const statusTagEl = document.querySelector('.ant-page-header-heading-extra span.ant-tag');
                return statusTagEl ? statusTagEl.textContent.trim() : null;
            }
            """
            header_data["status"] = await self._page.evaluate(status_js)
            self._logger.info(f"Extracted main header: {header_data}")
            return header_data
        except Exception as e:
            self._logger.error(f"Error extracting main header info: {e}", exc_info=True)
            raise DataExtractionError(f"Failed to extract main header info: {e}")


    async def _extract_key_value_section(self, section_title_text: str) -> Dict[str, str]:
        """
        Extracts key-value pairs from a section (e.g., Security Details, Underwriter Details).
        """
        self._logger.debug(f"Extracting key-value data for section: '{section_title_text}'")
        data: Dict[str, str] = {}
        
        section_container = await self._find_section_container_by_title(section_title_text)
        if not section_container:
            # If it's underwriter details, it might be optional, so don't raise error, return empty.
            if section_title_text == self._UNDERWRITER_DETAILS_TITLE_TEXT:
                self._logger.warning(f"Optional section '{section_title_text}' not found. Returning empty data.")
                return data
            raise ElementNotFoundError(f"Container for section '{section_title_text}' not found.")

        js_script = """
        (containerElement) => {
            const pairs = {};
            const rows = Array.from(containerElement.querySelectorAll('div.flex.items-center.justify-between.gap-2.border-b'));
            rows.forEach(row => {
                const labelEl = row.querySelector('span.ant-typography[style*="color: gray;"]');
                // Value can be in a span or an <a> tag (e.g., underwriter website)
                const valueEl = row.querySelector('span.ant-typography:not([style*="color: gray;"])') || row.querySelector('a');
                
                if (labelEl && valueEl) {
                    let label = labelEl.textContent.trim().replace(/:$/, ''); // Remove trailing colon
                    let value = valueEl.textContent.trim();
                    if (valueEl.tagName === 'A' && valueEl.href) {
                        // If it's a link, capture both text and href if desired, or prioritize one
                        // For simplicity, let's take text content. Caller can decide if href is needed.
                        // value = `${value} (${valueEl.href.trim()})`; // Example: "example.com (http://example.com)"
                    }
                    pairs[label] = value;
                }
            });
            return pairs;
        }
        """
        try:
            data = await section_container.evaluate(js_script)
            self._logger.info(f"Extracted {len(data)} key-value pairs from '{section_title_text}'.")
            return data
        except Exception as e:
            self._logger.error(f"Error evaluating JS for key-value extraction in '{section_title_text}': {e}", exc_info=True)
            raise DataExtractionError(f"Failed to extract key-value data from '{section_title_text}': {e}")


    async def _extract_document_list(self) -> List[Dict[str, str]]:
        """Extracts document names and their download links."""
        self._logger.debug("Extracting document list...")
        docs_meta: List[Dict[str, str]] = []
        
        section_container = await self._find_section_container_by_title(self._DOCUMENTS_TITLE_TEXT)
        if not section_container:
            self._logger.warning(f"Section '{self._DOCUMENTS_TITLE_TEXT}' not found. No documents to extract.")
            return docs_meta # No documents section, return empty list

        js_script = """
        (containerElement) => {
            const documents = [];
            const rows = Array.from(containerElement.querySelectorAll('div.flex.items-center.justify-between.gap-2.border-b'));
            rows.forEach(row => {
                const linkEl = row.querySelector('span.ant-typography > a.font-semibold');
                if (linkEl && linkEl.href && linkEl.textContent) {
                    documents.push({
                        file_name: linkEl.textContent.trim(),
                        download_href: linkEl.href.trim(),
                    });
                }
            });
            return documents;
        }
        """
        try:
            docs_meta = await section_container.evaluate(js_script)
            self._logger.info(f"Extracted metadata for {len(docs_meta)} documents.")
            return docs_meta
        except Exception as e:
            self._logger.error(f"Error evaluating JS for document list extraction: {e}", exc_info=True)
            raise DataExtractionError(f"Failed to extract document list: {e}")

    async def _extract_generic_table_data(self, section_title_text: str) -> List[Dict[str, Any]]:
        """
        Extracts data from a generic Ant Design table on the page (e.g., Transaction History).
        Currently extracts only the first page of data if paginated.
        """
        self._logger.debug(f"Extracting table data for section: '{section_title_text}'")
        
        # Find the ant-space-item that CONTAINS the table wrapper, using the title as a reference point.
        # This is more complex because the title and table are not direct parent/child.
        js_find_table_wrapper_container = f"""
        () => {{
            const titles = Array.from(document.querySelectorAll('div.text-lg.font-bold.text-white'));
            for (const titleEl of titles) {{
                if (titleEl.textContent && titleEl.textContent.trim() === "{section_title_text}") {{
                    // Navigate up to the common ancestor 'ant-space-item' for the title row
                    let titleSpaceItem = titleEl;
                    while(titleSpaceItem && (!titleSpaceItem.classList || !titleSpaceItem.classList.contains('ant-space-item'))) {{
                        titleSpaceItem = titleSpaceItem.parentElement;
                        if(!titleSpaceItem) return null; // Should not happen if title found
                    }}
                    
                    // The table is usually in the *next* 'ant-space-item' sibling
                    let tableSpaceItem = titleSpaceItem.nextElementSibling;
                    while(tableSpaceItem) {{
                        if (tableSpaceItem.classList && tableSpaceItem.classList.contains('ant-space-item') && tableSpaceItem.querySelector('div.ant-table-wrapper')) {{
                            return tableSpaceItem.querySelector('div.ant-table-wrapper');
                        }}
                        tableSpaceItem = tableSpaceItem.nextElementSibling;
                    }}
                    return null; // No subsequent sibling with a table found
                }}
            }}
            return null; // Title not found
        }}
        """
        try:
            await self._page.wait_for_function(js_find_table_wrapper_container, timeout=self._ELEMENT_WAIT_TIMEOUT_MS)
            table_wrapper_handle_obj = await self._page.evaluate_handle(js_find_table_wrapper_container)
            
            if not table_wrapper_handle_obj or not table_wrapper_handle_obj.as_element():
                self._logger.warning(f"Table wrapper for section '{section_title_text}' not found. Assuming no data or section absent.")
                return []
            table_wrapper_handle = table_wrapper_handle_obj.as_element()
        except PlaywrightTimeoutError:
            self._logger.warning(f"Timeout waiting for table wrapper for section '{section_title_text}'. Assuming no data.")
            return []
        except Exception as e_find:
            self._logger.error(f"Error finding table wrapper for '{section_title_text}': {e_find}", exc_info=True)
            raise ElementNotFoundError(f"Could not find table wrapper for '{section_title_text}'.")

        # --- BEGIN Inserted: Wait for table content (header, data, or "no data") to be ready ---
        self._logger.debug(f"Table wrapper for '{section_title_text}' found. Waiting for its content to be ready...")
        
        # JS to check for visible header, data row, or "No data" message within the table_wrapper_handle
        # The 'arg' in this JS function will be the table_wrapper_handle
        js_table_content_ready_condition = """
        (tableWrapperElement) => {
            if (!tableWrapperElement) { return false; }

            // Attempt to find the actual <table> element within the wrapper
            // Common AntD structure: wrapper > spin-container > table-div > container > content > table
            // Or simpler: wrapper > table directly if no spin/complex nesting
            const tableElement = tableWrapperElement.querySelector('div.ant-table table, table');
            if (!tableElement) {
                // console.warn('JS Detail Check: actual table element not found within wrapper.');
                return false;
            }

            const headerCell = tableElement.querySelector('thead > tr > th.ant-table-cell');
            const dataRow = tableElement.querySelector('tbody > tr.ant-table-row:not(.ant-table-measure-row):not(.ant-table-placeholder)');
            
            const isRenderedAndVisible = el => {
                if (!el) return false;
                const style = window.getComputedStyle(el);
                if (style.display === 'none' || style.visibility === 'hidden' || parseFloat(style.opacity) === 0) return false;
                // Check for offsetParent being null, but be careful if style.position is 'fixed'
                if (el.offsetParent === null && style.position !== 'fixed') return false; 
                const rect = el.getBoundingClientRect();
                return rect.width > 0 && rect.height > 0;
            };
            
            const headerVisible = headerCell && isRenderedAndVisible(headerCell);
            const dataRowVisible = dataRow && isRenderedAndVisible(dataRow);
            
            const noDataPlaceholder = tableElement.querySelector('tbody > tr.ant-table-placeholder .ant-empty-description');
            const noDataVisible = noDataPlaceholder && isRenderedAndVisible(noDataPlaceholder);

            // console.log(`JS Detail Content Check for '{section_title_text}': headerVisible=${headerVisible}, dataRowVisible=${dataRowVisible}, noDataVisible=${noDataVisible}`);
            return headerVisible || dataRowVisible || noDataVisible;
        }
        """
        try:
            await self._page.wait_for_function(
                js_table_content_ready_condition,
                arg=table_wrapper_handle, # Pass the handle as an argument to the JS function
                timeout=self._ELEMENT_WAIT_TIMEOUT_MS # Use a reasonable timeout for content to appear
            )
            self._logger.info(f"Content (header, data, or 'no data') for table '{section_title_text}' confirmed ready via JS check.")
        except PlaywrightTimeoutError:
            self._logger.warning(f"Timeout waiting for content of table '{section_title_text}' to become ready (header/data/no_data not visible). Proceeding with extraction, but it might yield 0 rows if content is truly absent or delayed.")
            # Depending on strictness, you could return [] here or raise DataExtractionError.
            # For now, let's allow the next step to try extraction. If that also yields nothing, it's consistent.
        except Exception as e_content_wait:
            self._logger.error(f"Error during JS wait for table content readiness in '{section_title_text}': {e_content_wait}", exc_info=True)
            # Decide if this is a critical error to stop or just a warning
            # For now, proceed to attempt extraction.
        # --- END Inserted ---

        js_extract_table_content = """
        (tableWrapperElement) => {
            // First, find the actual <table> element within the wrapper.
            const tableElement = tableWrapperElement.querySelector('div.ant-table table, table'); // Common AntD structure or direct table
            if (!tableElement) {
                // console.warn('JS Extract Content: Actual table element NOT FOUND within wrapper.');
                // Return an object indicating error, or just empty rows if Python side expects list.
                // For robustness, let's assume Python expects a list of rows.
                return []; 
            }

            // Check for "No data" placeholder relative to the tableElement's tbody or the wrapper as a fallback
            const noDataPlaceholderInTbody = tableElement.querySelector('tbody > tr.ant-table-placeholder .ant-empty-description');
            // Sometimes the "No Data" is directly in the wrapper if the entire table structure isn't rendered for empty state
            const noDataDirectInWrapper = tableWrapperElement.querySelector('.ant-empty-description'); 

            // Condition for "No Data":
            // 1. Placeholder found within tbody
            // 2. Placeholder found in wrapper AND there are no actual data rows in the tableElement
            let isNoDataState = false;
            if (noDataPlaceholderInTbody && noDataPlaceholderInTbody.textContent && noDataPlaceholderInTbody.textContent.trim().toLowerCase() === 'no data') {
                isNoDataState = true;
            } else if (noDataDirectInWrapper && noDataDirectInWrapper.textContent && noDataDirectInWrapper.textContent.trim().toLowerCase() === 'no data') {
                // If "no data" is in wrapper, double check tableElement has no data rows
                const actualDataRowsCheck = tableElement.querySelector('tbody tr.ant-table-row:not(.ant-table-measure-row):not(.ant-table-placeholder)');
                if (!actualDataRowsCheck) {
                    isNoDataState = true;
                }
            }

            if (isNoDataState) {
                // console.info('JS Extract Content: "No data" placeholder found and confirmed.');
                return []; // Return empty list of rows
            }

            const headers = Array.from(tableElement.querySelectorAll('thead th.ant-table-cell'))
                                .map(th => th.textContent.trim())
                                .filter(h => h); // Filter out empty headers
            
            const rows = [];
            const bodyRows = Array.from(tableElement.querySelectorAll('tbody.ant-table-tbody > tr[data-row-key]'));

            bodyRows.forEach(tr => {
                const rowData = {};
                const cells = Array.from(tr.querySelectorAll('td.ant-table-cell'));
                headers.forEach((header, index) => {
                    // If headers list was empty (e.g. no thead), use column index as key
                    const key = (headers.length > 0 && header) ? header : `column_${index}`;
                    if (cells[index]) {
                        rowData[key] = cells[index].textContent.trim();
                    } else {
                        rowData[key] = null; 
                    }
                });
                // Only add row if it has some data (not just keys with all null values, though current logic adds if keys exist)
                // A stricter check might be Object.values(rowData).some(val => val !== null && val !== '')
                if (Object.keys(rowData).length > 0) { 
                    rows.push(rowData);
                }
            });
            return rows; // This is what the Python side expects (List[Dict[str, Any]])
        }
        """
        # The try-except block for calling this JS and processing its result remains the same.
        # Ensure that if the JS itself returns an object with an error (if you adapt it to do so),
        # the Python side handles it, otherwise it expects a list of rows.
        # The version above is modified to return [] on error finding tableElement, or if "No data".
        try:
            table_data = await table_wrapper_handle.evaluate(js_extract_table_content)
            # --- BEGIN Debugging log for table_data content ---
            if isinstance(table_data, list):
                actual_len = len(table_data)
                self._logger.info(f"[DEBUG] For table '{section_title_text}': table_data is a list. Actual length: {actual_len}.")
                if actual_len > 0:
                    self._logger.info(f"[DEBUG] First row sample: {str(table_data[0])[:200]}...") # Log a sample
                else:
                    self._logger.info(f"[DEBUG] table_data is an empty list for '{section_title_text}'.")
            else:
                self._logger.warning(f"[DEBUG] For table '{section_title_text}': table_data IS NOT A LIST. Type: {type(table_data)}. Value: {str(table_data)[:200]}")
                actual_len = 0 # Or handle as error
            
            self._logger.info(f"Extracted {actual_len} rows from table '{section_title_text}'. (Logged length: {len(table_data) if isinstance(table_data, list) else 'N/A'})")
            # --- END Debugging log ---
            return table_data
        except Exception as e:
            self._logger.error(f"Error evaluating JS for table data extraction in '{section_title_text}': {e}", exc_info=True)
            raise DataExtractionError(f"Failed to extract table data from '{section_title_text}': {e}")

    async def extract_all_renderable_data(self) -> Dict[str, Any]:
        """
        Extracts all structured data from the security detail page.
        """
        self._logger.info("Starting extraction of all renderable data from detail page.")
        await self.verify_initial_elements() # Ensure basic page structure is loaded

        all_data: Dict[str, Any] = {}
        
        try:
            all_data["main_header_info"] = await self._extract_main_header_info()
        except DataExtractionError as e:
            self._logger.warning(f"Could not extract main header info: {e}")
            all_data["main_header_info"] = {"error": str(e)}

        try:
            all_data["security_details"] = await self._extract_key_value_section(self._SECURITY_DETAILS_TITLE_TEXT)
        except (ElementNotFoundError, DataExtractionError) as e:
            self._logger.error(f"Failed to extract Security Details: {e}")
            all_data["security_details"] = {"error": str(e)}
        
        try:
            # Underwriter details might be optional.
            all_data["underwriter_details"] = await self._extract_key_value_section(self._UNDERWRITER_DETAILS_TITLE_TEXT)
            if not all_data["underwriter_details"]:
                self._logger.info("No underwriter details found or section is empty.")
        except (ElementNotFoundError, DataExtractionError) as e: # Should not happen if ElementNotFoundError is handled in helper
            self._logger.warning(f"Issue extracting Underwriter Details (might be optional): {e}")
            all_data["underwriter_details"] = {"error": str(e)} # Or simply leave it out if truly optional

        try:
            all_data["file_list_metadata"] = await self._extract_document_list()
        except DataExtractionError as e:
            self._logger.error(f"Failed to extract document list: {e}")
            all_data["file_list_metadata"] = {"error": str(e)}
        
        # Tables
        # For now, these are extracted as-is. If a table is critical and missing, this might be an issue.
        try:
            all_data["rfq_table"] = await self._extract_generic_table_data(self._RFQ_TABLE_TITLE_TEXT)
        except (ElementNotFoundError, DataExtractionError) as e:
            self._logger.warning(f"Could not extract RFQ table (often empty): {e}")
            all_data["rfq_table"] = {"error": str(e)}

        try:
            all_data["disclosures_table"] = await self._extract_generic_table_data(self._DISCLOSURES_TABLE_TITLE_TEXT)
        except (ElementNotFoundError, DataExtractionError) as e:
            self._logger.warning(f"Could not extract Disclosures table (often empty): {e}")
            all_data["disclosures_table"] = {"error": str(e)}

        try:
            all_data["transaction_history_table"] = await self._extract_generic_table_data(self._TRANSACTION_HISTORY_TABLE_TITLE_TEXT)
        except (ElementNotFoundError, DataExtractionError) as e:
            self._logger.error(f"Failed to extract Transaction History table: {e}")
            all_data["transaction_history_table"] = {"error": str(e)}
            
        self._logger.info("Finished extracting all renderable data.")
        return all_data

    async def download_documents_from_page(
        self, 
        file_list_meta: List[Dict[str, Any]], 
        download_dir: Path
    ) -> Tuple[List[Tuple[str, Path]], List[Dict[str, str]]]: # Returns (successful_downloads, error_list)
        """
        Downloads documents listed in file_list_meta.
        file_list_meta items must contain 'file_name' and 'download_href'.
        """
        self._logger.info(f"Attempting to download {len(file_list_meta)} documents to '{download_dir}'.")
        
        successful_downloads: List[Tuple[str, Path]] = []
        download_errors: List[Dict[str, str]] = [] # Store dicts of {'file_name': name, 'error': reason}

        if not download_dir.exists():
            try:
                download_dir.mkdir(parents=True, exist_ok=True)
                self._logger.info(f"Created download directory: {download_dir}")
            except Exception as e_mkdir:
                self._logger.error(f"Could not create download directory {download_dir}: {e_mkdir}")
                # Add an error for each file that couldn't be downloaded due to this
                for file_info in file_list_meta:
                    file_name = file_info.get("file_name", "unknown_file")
                    download_errors.append({"file_name": file_name, "error": f"Download directory creation failed: {e_mkdir}"})
                return successful_downloads, download_errors

        for file_info in file_list_meta:
            original_file_name = file_info.get("file_name")
            download_href = file_info.get("download_href")

            if not original_file_name or not download_href:
                err_msg = f"Skipping download due to missing 'file_name' or 'download_href' in metadata: {file_info}"
                self._logger.warning(err_msg)
                download_errors.append({"file_name": original_file_name or "unknown", "error": "Missing metadata for download."})
                continue
            
            # Sanitize filename for local saving (simple version)
            # A more robust sanitizer might be needed depending on actual filenames.
            safe_local_filename = re.sub(r'[^\w\-. ()а-яА-ЯөӨүҮ]', '_', original_file_name) # Keep Mongolian chars
            # Ensure it's not excessively long
            safe_local_filename = (safe_local_filename[:200] + Path(safe_local_filename).suffix) if len(safe_local_filename) > 200 else safe_local_filename
            download_path = download_dir / safe_local_filename
            
            self._logger.debug(f"Attempting download: '{original_file_name}' from '{download_href}' to '{download_path}'.")

            # (inside the loop for file_info in file_list_meta)
            try:
                self._logger.debug(f"Attempting direct fetch for '{original_file_name}' from URL: {download_href}")
                # Use page.request to fetch the content directly
                # Set a timeout for the request itself, e.g., self._default_timeout
                response = await self._page.request.get(download_href, timeout=self._default_timeout)

                if response.ok: # status_code is 2xx
                    file_content = await response.body()
                    if not file_content:
                        err_msg = f"Fetched empty content for '{original_file_name}' from {download_href}."
                        self._logger.warning(err_msg)
                        download_errors.append({"file_name": original_file_name, "error": "Fetched empty content."})
                        continue # Skip to next file

                    # Write the binary content to file
                    with open(download_path, "wb") as f:
                        f.write(file_content)
                    
                    # Verify file exists and has size
                    if download_path.exists() and download_path.stat().st_size > 0:
                        self._logger.info(f"Successfully fetched and saved '{original_file_name}' to '{download_path}'. Size: {download_path.stat().st_size} bytes.")
                        successful_downloads.append((original_file_name, download_path))
                    else:
                        err_msg = f"File '{original_file_name}' saved to '{download_path}' but seems empty or missing post-save (after direct fetch)."
                        self._logger.error(err_msg)
                        download_errors.append({"file_name": original_file_name, "error": err_msg})
                        if download_path.exists(): download_path.unlink(missing_ok=True) # Clean up
                else:
                    # Handle non-OK responses (e.g., 403 for S3 AccessDenied/Expired)
                    status_code = response.status
                    try:
                        # Try to get some text from the error response body for logging
                        error_body_text = await response.text(timeout=2000) # Short timeout for error text
                        error_preview = error_body_text[:200].replace('\n', ' ') # Preview of error
                    except Exception:
                        error_preview = "Could not retrieve error body."

                    err_msg = f"HTTP Error {status_code} fetching '{original_file_name}' from {download_href}. Response: '{error_preview}'"
                    self._logger.warning(err_msg) # Log as warning as this is expected for expired links
                    download_errors.append({"file_name": original_file_name, "error": f"HTTP Error {status_code}", "details": error_preview})

            except PlaywrightTimeoutError as pte_request: # Timeout during page.request.get()
                err_msg = f"Timeout during HTTP GET request for '{original_file_name}': {str(pte_request).splitlines()[0]}"
                self._logger.error(err_msg)
                download_errors.append({"file_name": original_file_name, "error": "Request Timeout"})
            except Exception as e_request: # Other errors during request or file write
                err_msg = f"General error during direct fetch or save for '{original_file_name}': {e_request}"
                self._logger.error(err_msg, exc_info=True)
                download_errors.append({"file_name": original_file_name, "error": str(e_request)})
        
        self._logger.info(f"Download process finished. Success: {len(successful_downloads)}, Errors: {len(download_errors)}.")
        return successful_downloads, download_errors