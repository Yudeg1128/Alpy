# Project Plan: Alpy Financial Modeling Capability

**Version:** 1.0
**Date:** YYYY-MM-DD
**Author(s):** Alpy Development Team

## 1. Introduction & Goal

### 1.1. Purpose
To empower Alpy with the ability to automatically generate structured financial models for debt-issuing entities. This involves extracting financial data from unstructured documents, populating a predefined model schema, and enabling further analysis.

### 1.2. Initial Focus (Phase A Implementation)
The primary goal of the initial implementation phase is to reliably execute **Phase A: Historical Data Extraction**. This includes:
*   Parsing relevant financial documents (primarily PDFs like 10-Ks, prospectuses).
*   Identifying and extracting key historical financial data points.
*   Populating the "metadata" and "historical periods" sections of a standardized financial model JSON structure.
*   Implementing a quality check mechanism for the extracted historical data.

### 1.3. Core Inputs & Outputs
*   **Inputs:**
    *   `security_id`: Identifier for the target company.
    *   `documents_root_path`: Path to a directory containing financial documents for the `security_id`.
*   **Output (Phase A):**
    *   A JSON file representing the financial model, populated with `model_metadata` and all historical data sections from the `financial_statements_core` and relevant `supporting_schedules_debt_focus`.
    *   A Quality Report JSON detailing the success and any issues encountered during Phase A.

## 2. Core Architecture & Components

The financial modeling capability will be integrated using a modular approach, leveraging existing Alpy structures and introducing specialized components:

### 2.1. Main Alpy Agent (`src/agent.py`)
*   **Role:** The primary interface for the user. It will delegate the financial modeling task to a specialized high-level tool.
*   **Interaction:** Receives user requests to generate a financial model, invokes the `FinancialPhaseATool`, processes its output (Phase A JSON and Quality Report), and decides on next steps (e.g., proceed to projections, report errors, request user intervention).

### 2.2. Financial Modeling Module (`src/financial_modeling/`)
This new Python module will house the core logic for financial modeling tasks.

*   **`phase_a_orchestrator.py` -> `PhaseAOrchestrator` class:**
    *   **Role:** Not a traditional LLM-loop agent, but a Python class responsible for orchestrating the sequence of operations required to complete Phase A. It manages the sub-steps, calls necessary tools, and assembles the historical data portion of the financial model.
    *   **Why:** Encapsulates the complex multi-step logic of historical data extraction, making the process manageable and testable.
*   **`quality_checker.py` -> `QualityChecker` class:**
    *   **Role:** A Python class responsible for validating the output of the `PhaseAOrchestrator`. It performs schema checks, financial sanity checks, and can potentially use LLMs for more nuanced validation.
    *   **Why:** Ensures the reliability and accuracy of the extracted historical data before proceeding to projections or presenting to the user.
*   **`utils.py`:**
    *   **Role:** Contains helper functions for data normalization (e.g., converting "millions" to actual numbers, standardizing currencies), common financial calculations (used in sanity checks or later in projections), and other shared utilities for the financial modeling module.
    *   **Why:** Promotes code reuse and separation of concerns.
*   **`data_models.py` (Optional):**
    *   **Role:** May contain Pydantic models for intermediate data structures used during the financial modeling process, if the main schema is too unwieldy for certain steps.
    *   **Why:** Can improve type safety and clarity for internal data handling.

### 2.3. Specialized Tools (`src/tools/`)

*   **`phase_a_orchestrator.py` -> `PhaseAOrchestrator` class:**
    *   **Role:** A deterministic Python class that serves as the "project manager" for Phase A (Historical Data Extraction). It does **not** have its own LLM reasoning loop for orchestration. Instead, it executes a predefined sequence of steps to populate the historical sections of the financial model defined in `fund/financial_model_schema.json`. Its primary responsibilities include:
        1.  Managing the overall workflow for historical data extraction.
        2.  Invoking specialized tools (like `MCPDocumentParserTool` for document content and an LLM-based tool for interpreting that content) in a structured manner.
        3.  Preparing specific, targeted prompts (from `prompts/financial_modeling_prompts.yaml`) for each piece of information to be extracted by an LLM.
        4.  Systematically handling responses from tools, especially LLM tools. This includes:
            *   Validating the format and plausibility of LLM outputs.
            *   Implementing a tiered strategy for retries and re-prompting if an LLM fails to provide a satisfactory response for a specific data point.
            *   Logging all LLM interactions, successes, and failures meticulously.
        5.  Normalizing and structuring the extracted data.
        6.  Assembling the final Phase A JSON output, clearly marking any data points that could not be reliably extracted.
    *   **Why:** This approach ensures that the complex, multi-step process of historical data extraction is managed with procedural rigor. The Python class provides precise control over the sequence, data handling, and error management. While LLMs are used for their strength in interpreting unstructured text (especially from varied Mongolian PDFs), the `PhaseAOrchestrator` itself remains deterministic, making the process more predictable, testable, and its failures diagnosable. It ensures that even if individual LLM extractions fail, the overall process attempts to complete as much as possible and reports failures clearly.

*   **`quality_checker.py` -> `QualityChecker` class:**
    *   **Role:** A deterministic Python class acting as an "auditor" for the Phase A JSON output produced by the `PhaseAOrchestrator`. Its responsibilities include:
        1.  Performing strict schema validation against `fund/financial_model_schema.json`. This will identify missing required fields (which would be the result of extraction failures logged by the `PhaseAOrchestrator`).
        2.  Executing predefined, deterministic financial sanity checks (e.g., Assets = Liabilities + Equity for historical periods). These calculations are performed using Python, not an LLM.
        3.  (Optional advanced step) If sanity checks fail due to data that *was* extracted, it might formulate a targeted query for an LLM to re-examine the specific source excerpts related to the discrepant items, asking for clarification or re-validation of the LLM's prior extraction.
        4.  Generating a comprehensive Quality Report in a structured format (e.g., JSON). This report details all validation results, sanity check outcomes, lists any data points that failed extraction (including reasons logged by the orchestrator), and may offer suggestions for improving future extraction attempts (e.g., by highlighting consistently problematic prompts or document sections).
    *   **Why:** Provides an independent, rule-based verification of the `PhaseAOrchestrator`'s output. This separation ensures that the quality assessment is objective and based on defined financial and structural rules. The detailed report is crucial for understanding the reliability of the generated historical data and for guiding iterative improvements to the extraction process (e.g., prompt engineering, document pre-processing).

### 2.4. MCP Servers (`mcp_servers/`)

*   **`document_parser_server.py` -> `DocumentParserServer`:**
    *   **Role:** An MCP server dedicated to parsing various document formats (PDFs, DOCX, PPTX eventually).
    *   **Functionality:** Receives a document path, uses libraries like `PyMuPDF (fitz)` for PDFs (text and basic layout), `pdfplumber` for PDF table extraction, `python-docx` for DOCX files. Returns structured output (e.g., page-wise text, list of extracted tables with cell data).
    *   **Why:** Centralizes document parsing, which can be complex and require specific dependencies. Allows for independent development and optimization of parsing capabilities.

### 2.5. Configuration & Definitions

*   **`fund/financial_model_schema.json`:**
    *   **Role:** The definitive JSON Schema that describes the structure and data types of the target financial model. This includes metadata, assumptions, financial statements, supporting schedules, ratios, etc. The JSON example previously analyzed will be translated into this formal schema.
    *   **Why:** Provides a clear contract for data generation and enables programmatic validation of the output.
*   **`prompts/financial_modeling_prompts.yaml`:**
    *   **Role:** A YAML file containing specialized prompts tailored for LLM-based financial data extraction tasks (e.g., "Extract 'Total Revenue' for FY2023 from the provided text and tables," "Identify all outstanding debt tranches and their terms from this footnote.").
    *   **Why:** Separates financial-specific prompts from general agent prompts, allowing for easier management and refinement.
*   **`logs/financial_modeling.log`:**
    *   **Role:** A dedicated log file for all operations related to the financial modeling capability.
    *   **Why:** Facilitates debugging and monitoring of the financial modeling pipeline.

## 3. Workflow & Data Flow: Phase A - Historical Data Extraction

This section details the step-by-step process orchestrated by the `PhaseAOrchestrator`.

**Pre-requisite:** User provides `security_id` and `documents_root_path` to Alpy.

1.  **Initiation (Alpy Main Agent):**
    *   Alpy Agent receives the request.
    *   Alpy Agent calls `FinancialPhaseATool` with `security_id` and `documents_root_path`.

2.  **Orchestration (FinancialPhaseATool -> PhaseAOrchestrator):**
    *   The `FinancialPhaseATool._arun` method instantiates and starts the `PhaseAOrchestrator`.

3.  **Step 1: Document Discovery & Prioritization (PhaseAOrchestrator):**
    *   The orchestrator scans the `documents_root_path` for files.
    *   It prioritizes documents: e.g., latest 10-K/Annual Report first, then prospectuses, then 10-Qs.

4.  **Step 2: Document Parsing (PhaseAOrchestrator -> MCPDocumentParserTool -> DocumentParserServer):**
    *   For each priority document:
        *   Orchestrator calls `MCPDocumentParserTool` with the document path.
        *   `MCPDocumentParserTool` sends a request to `DocumentParserServer`.
        *   `DocumentParserServer` parses the document and returns extracted text (page-by-page) and structured tables (e.g., list of tables, where each table is a list of rows, and each row is a list of cells).
        *   Orchestrator stores this parsed content (e.g., in memory, associated with document name/page).

5.  **Step 3: Metadata Extraction (PhaseAOrchestrator -> LLM Tool):**
    *   Orchestrator prepares a prompt (from `financial_modeling_prompts.yaml`) for LLM.
    *   Context: Cover page text/initial sections from the primary parsed document (e.g., latest 10-K).
    *   Goal: Extract `target_company_name`, `ticker_symbol`, `currency`, `fiscal_year_end`.
    *   Orchestrator calls a generic LLM interaction tool/service (part of Alpy's core).
    *   Orchestrator parses the LLM's JSON response and populates the `model_metadata` section of its internal financial model structure.

6.  **Step 4: Historical Period Identification (PhaseAOrchestrator -> LLM Tool):**
    *   Prompt (from `financial_modeling_prompts.yaml`) for LLM.
    *   Context: Excerpts from financial statement sections of parsed key annual reports.
    *   Goal: Identify distinct, consistently reported historical financial periods and standardize them (e.g., "FY2023", "FY2022").
    *   Orchestrator populates `historical_period_labels` in its internal model.

7.  **Step 5: Line Item Extraction Loop (PhaseAOrchestrator -> LLM Tool & `financial_modeling/utils.py`)**

*   The `PhaseAOrchestrator` iterates through a predefined list of historical line items derived from `fin  ibutes of "Existing Debt Tranches").
*   For **each individual line item** to be extracted for **each historical period**:
    1.  **Internal State Check:** The orchestrator checks if this specific data point (e.g., "Revenue for FY2023") has already been attempted and marked as a persistent failure in a previous iteration (if such iterative refinement within a single Phase A run is implemented – initially, it might be one pass).
    2.  **Context Retrieval (RAG-like Preparation):**
        *   The orchestrator identifies the most relevant parsed document sections (text snippets, table data) from its working memory (content obtained in Step 2). This search is guided by the `line_item.name`, its `source_guidance_historical` from the schema, and the target `historical_period_label`.
        *   Example: For "Revenue FY2023," it looks for "Income Statement" tables or text discussing "Revenue" or "Борлуулалтын орлого" in documents pertaining to 2023.
    3.  **LLM Extraction Attempt & Detailed Failure Handling Strategy:**
        *   The orchestrator maintains a `max_attempts_per_item` counter (e.g., 3).
        *   **Attempt `n` (e.g., Attempt 1):**
            *   **Prompt Construction:** A specific prompt is assembled from `financial_modeling_prompts.yaml`.
                *   Example: "From the provided Mongolian financial document excerpts for period `[FY2023]`, extract the value for '`[line_item.name]`' (`[Mongolian term for line_item.name]`). Include the exact numerical value, reported currency (e.g., 'MNT', 'USD'), and unit (e.g., 'actuals', 'thousands', 'millions'). If the data is not found or not applicable, explicitly state 'NOT_FOUND'. Respond strictly in the following JSON format: `{\"value\": number_or_null, \"currency\": \"string_or_null\", \"unit\": \"string_or_null\", \"source_reference\": \"string_describing_source_page_table\", \"status\": \"EXTRACTED_SUCCESSFULLY_OR_CONFIRMED_NOT_FOUND\"}`. Context: `[Relevant text/table snippet]`."
            *   **LLM Call:** The orchestrator invokes the LLM tool with this prompt.
            *   **Response Logging:** The raw LLM response is logged immediately (e.g., to `financial_modeling.log`).
            *   **Output Validation & Processing:**
                *   **A. API/Network Error Check:**
                    *   If an API error occurs (e.g., timeout, 503 from LLM provider): Log error. If `current_attempt < api_retry_limit`, increment `current_attempt` for API error, wait with exponential backoff, and retry the *same* LLM call. If `api_retry_limit` reached, mark item as "EXTRACTION_FAILED_API_ERROR" and break from retry loop for this item.
                *   **B. JSON Format Validation:**
                    *   Attempt to parse LLM response as JSON.
                    *   If not valid JSON: Log error. Increment `current_attempt`. If `current_attempt < max_attempts_per_item`, go to next attempt (which will use a re-prompt). Else, mark item as "EXTRACTION_FAILED_INVALID_JSON_RESPONSE" and break.
                *   **C. Schema/Key Validation (for the expected JSON structure):**
                    *   Check if parsed JSON contains required keys (`value`, `currency`, `unit`, `status`).
                    *   If keys missing: Log error. Increment `current_attempt`. If `current_attempt < max_attempts_per_item`, go to next attempt (re-prompt). Else, mark item as "EXTRACTION_FAILED_MISSING_KEYS_IN_JSON" and break.
                *   **D. Status Check & Value Plausibility:**
                    *   If `status` is "CONFIRMED_NOT_FOUND": Mark item as "DATA_NOT_FOUND_AS_PER_LLM" and break (success in confirming absence).
                    *   If `status` is "EXTRACTED_SUCCESSFULLY":
                        *   Check `value` data type (is it `null`, a number, or convertible to a number?).
                        *   (Optional, very loose) Check if `value` is within an extremely broad plausible range (e.g., not negative for revenue).
                        *   If `value` is `null` but status is success: Treat as "DATA_NOT_FOUND_AS_PER_LLM".
                        *   If type/plausibility issue: Log error. Increment `current_attempt`. If `current_attempt < max_attempts_per_item`, go to next attempt (re-prompt). Else, mark item as "EXTRACTION_FAILED_IMPLAUSIBLE_VALUE" and break.
                    *   If `status` is something else or missing: Treat as an invalid response (go to C).
                *   **E. Re-Prompting Strategy for Next Attempt (if `current_attempt < max_attempts_per_item` and a recoverable error occurred in B, C, or D):**
                    *   The prompt for the next attempt will be modified.
                    *   If invalid JSON: "Your previous response was not valid JSON. Please ensure your output is strictly JSON. Extract `[line_item.name]`..."
                    *   If missing keys: "Your previous JSON response was missing required keys. Ensure `value`, `currency`, `unit`, `status` are present. Extract `[line_item.name]`..."
                    *   If implausible value: "The value extracted for `[line_item.name]` seemed implausible. Please re-verify from the context. Context: `[snippet]`. Extract `[line_item.name]`..."
                    *   If LLM returned text but not the requested JSON: "Please provide your answer strictly in the JSON format specified previously. Do not include conversational text outside the JSON. Extract `[line_item.name]`..."
                    *   Go back to "LLM Call" with the new prompt for this attempt.
            *   **If Extraction Successful (Valid JSON, plausible value, or CONFIRMED_NOT_FOUND status):**
                *   Proceed to Data Normalization (if value extracted).
                *   Store the result (value or "NOT_FOUND" status) and the `source_reference`.
                *   Break from the retry loop for this item.
    4.  **Data Normalization (if value extracted):**
        *   Use functions from `financial_modeling/utils.py` to convert units (e.g., "сая ₮" / "millions MNT" to `1000000 * value`), handle numerical formatting (remove commas, parse parentheses for negatives if culturally appropriate for Mongolian financials).
    5.  **Store Result in Internal Model:**
        *   The orchestrator updates its internal representation of the financial model for the specific `line_item` and `historical_period_label` with either the normalized value and its metadata (currency, original unit, source) or the "failure_status" (e.g., "EXTRACTION_FAILED_MAX_ATTEMPTS", "DATA_NOT_FOUND_AS_PER_LLM").

8.  **Step 6: Assemble Phase A JSON (PhaseAOrchestrator):**
    *   The orchestrator compiles all extracted, normalized, and stored historical data into a single JSON object that strictly adheres to the `fund/financial_model_schema.json`.

9.  **Step 7: Quality Check (FinancialPhaseATool or AlpyAgent -> QualityChecker)**

*   The Phase A JSON, containing both successfully extracted data and fields marked with extraction failure statuses (or `null`), is passed to the `QualityChecker` Python class.
*   **A. Schema Validation:**
    *   `QualityChecker` uses a JSON Schema validator (e.g., `jsonschema` library) against `fund/financial_model_schema.json`.
    *   **Outcome:** Reports any structural errors. If a *required* field in the schema is `null` or has an "EXTRACTION_FAILED_..." status, this constitutes a schema validation failure for completeness (even if the structure around it is fine).
*   **B. Deterministic Financial Sanity Checks (Python-based):**
    *   `QualityChecker` iterates through a predefined set of financial logic rules.
        *   Example Rule: "Historical Total Assets must equal Historical Total Liabilities + Historical Total Equity for each period."
        *   **Execution:**
            1.  Attempt to retrieve the necessary input values (e.g., "Total Assets FY2023," "Total Liabilities FY2023," "Total Equity FY2023") from the input Phase A JSON.
            2.  If any required input value for the check has an "EXTRACTION_FAILED_..." status or is `null`: The sanity check for that period is marked as "CANNOT_PERFORM_DUE_TO_MISSING_DATA: `[list of missing items]`."
            3.  If all inputs are available: Perform the calculation (e.g., `Assets - (Liabilities + Equity)`).
            4.  Compare result against expected outcome (e.g., difference should be close to zero, within a small tolerance for rounding).
            5.  Mark check as "PASSED" or "FAILED: `[details of discrepancy, e.g., Assets off by X]`."
*   **C. (Optional Advanced) LLM-Assisted Discrepancy Review:**
    *   If a sanity check *fails* due to conflicting *extracted* values (not missing data):
        *   `QualityChecker` might formulate a prompt for an LLM.
        *   Example: "The system extracted Total Assets = `A`, Total Liabilities = `L`, Total Equity = `E` for FY2023 from sources `S_A`, `S_L`, `S_E` respectively. However, `A != L + E`. Please review these source excerpts: `[Provide S_A, S_L, S_E]`. Is there a clear misinterpretation in one of the extractions, or a disclosed reconciling item?"
        *   This is a more nuanced validation step and adds complexity.
*   **D. Quality Report Generation:**
    *   `QualityChecker` assembles a structured JSON Quality Report, including:
        *   Overall summary (e.g., "Phase A completed with X% data point extraction success.").
        *   Schema validation results (pass/fail, list of errors).
        *   Detailed results for each financial sanity check (period, check name, status: PASSED, FAILED, CANNOT_PERFORM, details of failure/missing items).
        *   A consolidated list of all data points that failed extraction, along with their final "failure_status" logged by the `PhaseAOrchestrator` (e.g., `{"item": "Revenue_FY2022", "status": "EXTRACTION_FAILED_MAX_ATTEMPTS", "final_llm_error": "LLM_OUTPUT_INVALID_JSON_RESPONSE"}`).
        *   (If C implemented) Summaries of LLM reviews of discrepancies.
        *   Potential suggestions for improvement if patterns emerge (e.g., "LLM frequently failed to parse tables from Document X.pdf after page Y. Consider document pre-processing or prompt adjustment for table-heavy sections in such documents.").

10. **Completion (Alpy Main Agent):**
    *   `FinancialPhaseATool` returns the Phase A JSON and the Quality Report JSON to the main Alpy Agent.
    *   Alpy Agent examines the Quality Report and decides the next course of action.

## 4. File Structure Integration Summary

```
Alpy/
├── fund/
│   └── financial_model_schema.json         # Defines target model structure
├── logs/
│   └── financial_modeling.log              # Dedicated logs for this capability
├── mcp_servers/
│   └── document_parser_server.py         # MCP server for PDF/DOCX parsing
├── prompts/
│   └── financial_modeling_prompts.yaml   # LLM prompts for financial extraction
├── src/
│   ├── agent.py                            # Main Alpy Agent
│   ├── financial_modeling/                 # NEW MODULE
│   │   ├── __init__.py
│   │   ├── phase_a_orchestrator.py       # Logic for Phase A steps
│   │   ├── quality_checker.py            # Logic for validating Phase A output
│   │   ├── utils.py                      # Helper functions (normalization, calcs)
│   │   └── data_models.py                # (Optional) Pydantic models for internal use
│   └── tools/
│       ├── financial_phase_a_tool.py       # LangChain tool called by AlpyAgent
│       └── mcp_document_parser_tool.py   # LangChain client for DocumentParserServer
# ... other existing files and directories ...
```

## 5. Development Phases & Future Enhancements

### 5.1. Phase 1: Historical Data Extraction (Current Focus)
*   **Task 1:** Define and finalize `fund/financial_model_schema.json` (historical sections).
*   **Task 2:** Implement `DocumentParserServer` (`mcp_servers/document_parser_server.py`) for robust PDF text and table extraction.
*   **Task 3:** Implement `MCPDocumentParserTool` (`src/tools/mcp_document_parser_tool.py`).
*   **Task 4:** Create initial set of prompts in `prompts/financial_modeling_prompts.yaml` for metadata and key historical line items.
*   **Task 5:** Implement core logic of `PhaseAOrchestrator` (`src/financial_modeling/phase_a_orchestrator.py`), focusing on document iteration, context retrieval, and LLM calls for extraction.
*   **Task 6:** Implement `financial_modeling/utils.py` for data normalization.
*   **Task 7:** Implement `QualityChecker` (`src/financial_modeling/quality_checker.py`) with schema validation and basic financial sanity checks.
*   **Task 8:** Implement `FinancialPhaseATool` (`src/tools/financial_phase_a_tool.py`) to tie orchestrator and checker together.
*   **Task 9:** Integrate the `FinancialPhaseATool` into the main `AlpyAgent`.
*   **Task 10:** Rigorous testing with sample documents.

### 5.2. Phase 2: Assumptions & Projections
*   Extend `PhaseAOrchestrator` (or create `PhaseBOrchestrator`) to derive/input assumptions from historical data and document context (MD&A, guidance).
*   Implement a Python-based projection engine (likely in `financial_modeling/utils.py` or a new `projection_engine.py`) to generate projected financial statements based on assumptions.
*   Address circular dependencies in projections (e.g., Debt-Interest-Cash).

### 5.3. Phase 3: Ratios, Analysis & Advanced Features
*   Implement calculation of all credit metrics and ratios defined in the schema.
*   Develop logic for valuation context (DCF, Multiples).
*   Implement sensitivity and scenario analysis capabilities.
*   Enhance the `QualityCheckAgent` with more sophisticated financial logic and LLM-driven reviews.
*   Expand document type support in `DocumentParserServer` (DOCX, PPTX).
*   Improve RAG techniques for context retrieval.

## 6. Key Considerations
*   **LLM Choice:** Select LLMs proficient in financial context and structured data extraction.
*   **Prompt Engineering:** Iterative refinement of prompts in `financial_modeling_prompts.yaml` will be critical.
*   **Error Handling & Resilience:** Robust error handling is needed at each step (document parsing, LLM calls, data validation).
*   **Modularity:** Maintain clear separation of concerns between components.
*   **Testability:** Design components for unit and integration testing.
*   **Scalability:** While initial focus is single-run, consider how components might scale if Alpy needs to process many companies.