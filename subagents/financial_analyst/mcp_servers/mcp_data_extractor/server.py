"""
MCP Server for Structured Bond Data Extraction

This server extracts structured data from bond documents with a strong focus on unit normalization.
All financial values are converted to base units (not millions or billions) during extraction.
"""
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP
from mcp.types import Implementation
import logging
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../tools')))
from vector_ops_tool import VectorOpsTool

# --- Gemini LLM Setup ---
from langchain_google_genai import ChatGoogleGenerativeAI
sys.path.append(str(Path(__file__).parent.parent.parent / "financial_analyst"))
import config
from security_folder_utils import get_subfolder, require_security_folder
import tiktoken

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MCPDataExtractor")
logger.info("Starting MCP Data Extractor server")

# --- Gemini LLM Helper ---
def get_gemini_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-04-17",
        temperature=0.0,
        top_p=getattr(config, "LLM_TOP_P", 1.0),
        google_api_key=getattr(config, "GOOGLE_API_KEY", None),
    )

# --- MCP App Setup ---
mcp_app = FastMCP(
    name="MCPDataExtractorServer",
    version="0.1.0",
    description="MCP server for extracting structured bond data"
)

# --- Input Model ---
class ExtractionInput(BaseModel):
    security_id: str = Field(..., description="Security identifier")
    schema_path: Optional[str] = Field(None, description="Path to the schema JSON file")
    repair_prompt: Optional[str] = Field(None, description="OPTIONAL: If set, provides explicit repair instructions or feedback for the LLM to fix extraction errors. Used only for quality control or repair retries.")

class ExtractionOutput(BaseModel):
    status: str
    message: str
    output_path: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None

# --- Section Extraction Helpers ---

def build_section_prompt(section_name, section_desc, section_properties, docs_json, doc_schema, security_id=None, repair_prompt=None):
    """
    Build an LLM prompt for extracting a section. If repair_prompt is provided, append it as a user message for repair/quality control.
    For all financial sections, explicitly instruct unit normalization to ensure consistent base units.
    If security_id is provided and section is financial, try to use principal amount as source of truth for units.
    """
    unit_instructions = ""
    if section_name in ["bond_financials_historical", "bond_financials_projections", "bond_metadata"]:
        # Base unit normalization instructions
        unit_instructions = (
            "CRITICAL - UNIT NORMALIZATION INSTRUCTIONS:\n\n"
            "1. ALWAYS convert ALL financial values to their BASE UNITS (not millions or billions).\n"
        )
        
        # Add principal amount reference if available and not extracting bond_metadata itself
        if security_id and section_name != "bond_metadata":
            # Try to get principal amount from bond_metadata as source of truth
            principal_amount, currency = get_bond_principal_amount(security_id)
            
            if principal_amount is not None and currency is not None:
                # Calculate magnitude (millions or billions)
                magnitude = "billions" if principal_amount >= 1000000000 else "millions" if principal_amount >= 1000000 else "thousands"
                magnitude_value = principal_amount / 1000000000 if magnitude == "billions" else principal_amount / 1000000 if magnitude == "millions" else principal_amount / 1000
                
                unit_instructions += (
                    f"2. SOURCE OF TRUTH: The bond principal amount is {principal_amount} {currency} in BASE UNITS, which is {magnitude_value:.2f} {magnitude} {currency}.\n"
                    f"3. Use the bond principal amount as your REFERENCE POINT for unit consistency across all financial data.\n"
                    f"4. CRITICAL: ALL financial values MUST be normalized to the SAME BASE UNITS as the principal amount.\n"
                    f"5. CRITICAL: The base unit scale must be consistent throught the periods e.g. assets for 2022 and assets for 2023, 2024 etc must be on the same  scale and magnitude.\n"
                )
            else:
                pass
        
    financial_sanity_instructions = (
        "CRITICAL - FINANCIAL SANITY CHECKS:\n"
        "1. ALWAYS compare each financial value to the bond principal amount to ensure consistent magnitude\n"
        "2. If total assets appear too small compared to the bond principal amount, this indicates a unit error that MUST be fixed\n"
        "3. Ensure the company's financial scale makes logical sense relative to its bond issuance\n"
        "4. Apply your knowledge of financial statements and accounting principles to ensure data consistency\n"
        "5. Remember that financial statements are interconnected - balance sheets, income statements, and cash flow statements should tell a coherent story\n"
        "6. Look for and resolve any accounting inconsistencies that suggest unit conversion errors\n"
        "7. Examples of checks you might perform: verifying balance sheet equations, ensuring income statement components sum correctly, checking for reasonable financial ratios\n"
        "8. If you detect inconsistencies in magnitude across different sources, normalize ALL values to match the principal amount's units\n"
        "9. ALL financial values within the same statement MUST use the same magnitude as the principal amount\n"
        "10. If you find values that differ by factors of 1000 or 1000000, this indicates a unit conversion issue that MUST be resolved\n"
    )

    prompt = [
        {"role": "system", "content": (
            f"You are a financial data extraction assistant. You extract only the requested section from JSON documents.\n"
            f"Section: {section_name}\n"
            f"Description: {section_desc}\n"
            f"Extraction schema (JSONSchema): {json.dumps(section_properties, ensure_ascii=False)}\n"
            f"{unit_instructions}\n"
            f"{financial_sanity_instructions}\n"
            "ALWAYS output a JSON object matching the schema.\n"
            "ALWAYS include BOTH a 'comments' field and an 'input_quality' field in your output.\n"
            "- In 'comments', provide a concise, high-level summary of the extraction quality and data completeness. Focus on what data was successfully extracted, what was missing, and overall confidence in the extraction. DO NOT include unit conversion details in 'comments' - those belong in 'unit_conversion_notes'. Example good comment: 'Extracted comprehensive financial data for 2022-2024, including balance sheet and income statement metrics. Some cash flow metrics were missing for 2023. Overall high confidence in extracted values after unit normalization.'\n"
            "- In 'input_quality', explain which documents you actually used for extraction and why, and which documents were not useful and why. Be specific about the strengths and weaknesses of each input document for the extraction task."
        )},
        {"role": "user", "content": (
            "Here are the relevant JSON documents. Extract the section as a JSON object matching the extraction schema.\n"
            f"Relevant documents: {json.dumps(docs_json, ensure_ascii=False)}"
        )}
    ]
    if repair_prompt:
        prompt.append({"role": "user", "content": f"REPAIR INSTRUCTIONS: {repair_prompt}"})
    return prompt

def build_collateral_prompt(section_name, section_desc, section_properties, docs_json, doc_schema, repair_prompt=None):
    """
    Build a specific LLM prompt for extracting all possible securitization and collateral-related information.
    This prompt is tailored to maximize recall for any collateral, security, asset backing, pledge, or securitization-related details.
    If repair_prompt is provided, append it as a user message for repair/quality control.
    """
    # Unit normalization instructions for collateral values
    unit_instructions = (
        "CRITICAL - UNIT NORMALIZATION INSTRUCTIONS:\n"
        "1. For ANY financial values related to collateral (e.g., asset values, minimum collateral requirements, etc.):\n"
        "   - ALWAYS convert to BASE UNITS (not millions or billions)\n"
        "   - If values are in 'Million MNT' or 'сая төгрөг', multiply by 1,000,000\n"
        "   - If values are in 'Billion MNT' or 'тэрбум төгрөг', multiply by 1,000,000,000\n"
        "2. Document all unit conversions in the 'comments' field\n"
        "3. For collateral coverage ratios, ensure they are expressed as decimals (e.g., 1.2x = 120%)\n"
    )
    
    prompt = [
        {"role": "system", "content": (
            f"You are a financial data extraction assistant specializing in bond documents. Your task is to extract ALL information related to collateral, security, securitization, asset backing, pledged assets, guarantees, liens, or any form of credit enhancement.\n"
            f"Section: {section_name}\n"
            f"Description: {section_desc}\n"
            f"Extraction schema (JSONSchema): {json.dumps(section_properties, ensure_ascii=False)}\n"
            f"{unit_instructions}\n"
            f"Your output MUST be a single JSON object matching the Extraction schema provided. Do NOT include any other text or explanation in your response.\n"
            f"If a field is not found, set its value to null.\n"
            f"You MUST ALWAYS include a field called 'comments' in your output. In 'comments', provide a concise summary of the extraction quality, explicitly mention any missing, ambiguous, or low-confidence fields, and describe any issues or uncertainties encountered with the input data. If everything is perfect, state so explicitly.\n"
            f"When extracting this section, focus on identifying and synthesizing every detail about collateral, security interests, asset-backed features, pledged assets, guarantees, liens, negative pledge clauses, asset sale restrictions, and any other forms of credit enhancement or protection for bondholders.\n"
            f"If relevant information is implied or scattered across documents, synthesize it into a coherent response. If only partial or ambiguous information is present, explain your reasoning in the 'input_quality' field.\n"
            f"Additionally, ALWAYS include a field called 'input_quality' in your output. In 'input_quality', explain which documents you actually used for extraction and why, and which documents were not useful and why. Be specific about the strengths and weaknesses of each input document for the extraction task."
        )},
        {"role": "user", "content": (
            "Here are the relevant JSON documents. Extract the section as a JSON object matching the extraction schema.\n"
            f"Relevant documents: {json.dumps(docs_json, ensure_ascii=False)}"
        )}
    ]
    if repair_prompt:
        prompt.append({"role": "user", "content": f"REPAIR INSTRUCTIONS: {repair_prompt}"})
    return prompt

def build_issuer_profile_prompt(section_name, section_desc, section_properties, docs_json, doc_schema, repair_prompt=None):
    """
    Build a specific LLM prompt for extracting qualitative issuer business profile information.
    Requires comments and input_quality fields, focusing on synthesis and qualitative understanding.
    If repair_prompt is provided, append it as a user message for repair/quality control.
    """
    # Unit normalization instructions for issuer profile financial metrics
    unit_instructions = (
        "CRITICAL - UNIT NORMALIZATION INSTRUCTIONS:\n"
        "1. For ANY financial values mentioned in the issuer profile (e.g., revenue figures, market size, capital, etc.):\n"
        "   - ALWAYS convert to BASE UNITS (not millions or billions)\n"
        "   - If values are in 'Million MNT' or 'сая төгрөг', multiply by 1,000,000\n"
        "   - If values are in 'Billion MNT' or 'тэрбум төгрөг', multiply by 1,000,000,000\n"
        "2. Document all unit conversions in the 'comments' field\n"
        "3. When describing company size, market share, or other metrics, ensure all numerical values are in base units\n"
    )
    
    prompt = [
        {"role": "system", "content": (
            f"You are a financial data extraction assistant specializing in bond documents. Your task is to extract ALL relevant information about the issuer's business profile, operations, industry, strategy, strengths, risks, and management.\n"
            f"Section: {section_name}\n"
            f"Description: {section_desc}\n"
            f"Extraction schema (JSONSchema): {json.dumps(section_properties, ensure_ascii=False)}\n"
            f"{unit_instructions}\n"
            f"Your output MUST be a single JSON object matching the Extraction schema provided. Do NOT include any other text or explanation in your response.\n"
            f"If a field is not found, set its value to null.\n"
            f"You MUST ALWAYS include a field called 'comments' in your output. In 'comments', provide a concise summary of the extraction quality, explicitly mention any missing, ambiguous, or low-confidence fields, and describe any issues or uncertainties encountered with the input data. If everything is perfect, state so explicitly.\n"
            f"When extracting this section, focus on synthesizing qualitative, descriptive, and contextual information about the issuer, its business model, sector, strategy, competitive advantages, risks, and management, even if not presented in a structured way.\n"
            f"If relevant information is implied or scattered across documents, synthesize it into a coherent response. If only partial or ambiguous information is present, explain your reasoning in the 'input_quality' field.\n"
            f"Additionally, ALWAYS include a field called 'input_quality' in your output. In 'input_quality', explain which documents you actually used for extraction and why, and which documents were not useful and why. Be specific about the strengths and weaknesses of each input document for the extraction task."
        )},
        {"role": "user", "content": (
            "Here are the relevant JSON documents. Extract the section as a JSON object matching the extraction schema.\n"
            f"Relevant documents: {json.dumps(docs_json, ensure_ascii=False)}"
        )}
    ]
    if repair_prompt:
        prompt.append({"role": "user", "content": f"REPAIR INSTRUCTIONS: {repair_prompt}"})
    return prompt

def build_bond_metadata_prompt(section_name, section_desc, section_properties, docs_json, doc_schema, security_id=None, repair_prompt=None):
    """
    Build a specific LLM prompt for extracting bond metadata with special emphasis on principal amount extraction.
    Uses security_details.json's Total Amount Issued as the deterministic source of truth if available.
    The principal amount is the source of truth for unit normalization across all financial data.
    If repair_prompt is provided, append it as a user message for repair/quality control.
    """
    # Try to get the Total Amount Issued from security_details.json as the deterministic source of truth
    principal_instruction = ""
    if security_id:
        try:
            security_folder = get_security_folder(security_id)
            sec_details_path = security_folder / "security_details.json"
            if sec_details_path.exists():
                with open(sec_details_path, "r", encoding="utf-8") as f:
                    sec_details = json.load(f)
                
                total_amount = sec_details.get("Total Amount Issued")
                if total_amount:
                    # Extract numeric value and remove commas
                    amount_str = total_amount.replace(",", "").replace("₮", "").replace("MNT", "").strip()
                    try:
                        # Convert to base units if needed (e.g., if expressed in millions or billions)
                        principal_instruction = (
                            "DETERMINISTIC SOURCE OF TRUTH:\n"
                            f"The 'Total Amount Issued' from security_details.json is '{total_amount}'.\n"
                            f"You MUST use this value as the authoritative source for 'principal_amount_issued'.\n"
                            "Remember to convert to base units (remove commas, convert from millions/billions if needed).\n"
                        )
                        logger.info(f"Using deterministic principal amount from security_details.json: {total_amount}")
                    except ValueError:
                        logger.warning(f"Could not parse Total Amount Issued: {total_amount}")
        except Exception as e:
            logger.warning(f"Error reading security_details.json: {str(e)}")
    
    prompt = [
        {"role": "system", "content": (
            f"You are a bond data extraction expert. Extract the bond metadata from the provided documents.\n"
            f"Section: {section_name}\n"
            f"Description: {section_desc}\n"
            f"Extraction schema (JSONSchema): {json.dumps(section_properties, ensure_ascii=False)}\n\n"
            "CRITICAL - UNIT NORMALIZATION INSTRUCTIONS:\n\n"
            f"{principal_instruction}"
            "1. ALWAYS convert ALL financial values to their BASE UNITS (not millions or billions).\n"
            "2. The principal_amount_issued MUST be in BASE UNITS and is the SOURCE OF TRUTH for all other financial data.\n"
            "3. If you see 'Million MNT' or 'сая төгрөг', convert to base MNT by multiplying by 1,000,000.\n"
            "4. If you see 'Billion MNT' or 'тэрбум төгрөг', convert to base MNT by multiplying by 1,000,000,000.\n"
            "5. If you see 'Million USD', convert to base USD by multiplying by 1,000,000.\n"
            "6. If you see 'Billion USD', convert to base USD by multiplying by 1,000,000,000.\n"
            "7. For bond principal amount, ALWAYS convert to base units (not millions/billions).\n"
            "8. CRITICAL - APPLY LOGICAL CONSISTENCY CHECKS to the financial data:\n"
            "   - Ensure the company's financial scale makes logical sense relative to its bond issuance\n"
            "   - A company typically cannot issue bonds worth more than its total assets\n"
            "   - If total assets appear too small compared to the bond principal amount, this likely indicates a unit error\n"
            "   - If you detect inconsistencies in magnitude across different sources, use your judgment to determine the correct scale\n"
            "9. Set 'base_unit_of_measure' to the normalized unit (e.g., 'MNT' or 'USD').\n"
            "10. In 'unit_conversion_notes', document in detail:\n"
            "   - The original units found in the source documents\n"
            "   - Exact conversions performed for the principal amount\n"
            "   - Your reasoning for unit determination when not explicitly stated\n"
            "   - How you resolved any magnitude inconsistencies using logical checks\n"
            "11. NEVER leave values in millions or billions - ALL numerical values must be in BASE UNITS.\n\n"
            "EXAMPLES:\n"
            "- If principal is stated as '₮ 1,000,000,000', it must be extracted as 1000000000 with base_unit_of_measure='MNT'\n"
            "- If principal is stated as '5 million USD', it must be extracted as 5,000,000 with base_unit_of_measure='USD'\n"
            "ALWAYS output a JSON object matching the schema.\n"
            "ALWAYS include BOTH a 'comments' field and an 'input_quality' field in your output.\n"
            "- In 'comments', provide notes on extraction quality, missing data, or encountered problems.\n"
            "- In 'input_quality', explain which documents you actually used for extraction and why."
        )},
        {"role": "user", "content": (
            "Here are the relevant JSON documents. Extract the bond metadata as a JSON object matching the extraction schema.\n"
            f"Relevant documents: {json.dumps(docs_json, ensure_ascii=False)}"
        )}
    ]
    if repair_prompt:
        prompt.append({"role": "user", "content": f"REPAIR INSTRUCTIONS: {repair_prompt}"})
    return prompt


def build_industry_profile_prompt(section_name, section_desc, section_properties, docs_json, doc_schema, repair_prompt=None):
    """
    Build a specific LLM prompt for extracting qualitative industry profile information.
    Requires comments and input_quality fields, focusing on synthesis and qualitative understanding.
    If repair_prompt is provided, append it as a user message for repair/quality control.
    """
    # Define the path to industry_research.json
    INDUSTRY_RESEARCH_FILE = "/home/me/CascadeProjects/Alpy/subagents/financial_analyst/external_reports/industry_research.json"

    # Load industry research data if the file exists
    industry_research_data = {}
    if os.path.exists(INDUSTRY_RESEARCH_FILE):
        try:
            with open(INDUSTRY_RESEARCH_FILE, 'r') as f:
                industry_research_data = json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from {INDUSTRY_RESEARCH_FILE}: {e}")
        except Exception as e:
            logging.error(f"Error reading {INDUSTRY_RESEARCH_FILE}: {e}")

    # Add industry research data to docs_json if it's not empty
    if industry_research_data:
        docs_json.append({"type": "industry_research", "content": industry_research_data})

    prompt = [
        {"role": "system", "content": (
            f"You are a financial data extraction assistant specializing in bond documents. Your task is to extract ALL relevant information about the issuer's industry, sectoral context, market environment, trends, risks, and competitive landscape.\n"
            f"Section: {section_name}\n"
            f"Description: {section_desc}\n"
            f"Extraction schema (JSONSchema): {json.dumps(section_properties, ensure_ascii=False)}\n"
            f"Your output MUST be a single JSON object matching the Extraction schema provided. Do NOT include any other text or explanation in your response. All extracted values should be in English where applicable.\n"
            f"If a field is not found, set its value to null.\n"
            f"You MUST ALWAYS include a field called 'comments' in your output. In 'comments', provide a concise, high-level summary of the overall extraction quality and data completeness for this section. DO NOT include any details about unit conversions or normalization in 'comments'; those MUST be placed in 'unit_conversion_notes'. Instead, focus on: Was the extraction well done? Was there enough relevant data in the input documents? What key data points or sections were successfully extracted or were notably missing? Describe any significant challenges or uncertainties encountered with the input data or the extraction process itself. If everything was perfect and sufficient data was found, state so explicitly. For example, discuss data availability, consistency across documents (excluding unit issues), and the overall success of the extraction.\n"
            f"When extracting this section, you MUST infer the issuer's industry, sector, and market context from ALL available clues in the documents, even if not stated directly. If the precise industry is not mentioned, deduce the broad sector (e.g., financial, non-bank financial, fintech, manufacturing, etc.) based on the issuer's business activities, regulatory context, or other indirect evidence. Always provide your best classification of the issuer's industry or sector, and explain your reasoning in the 'comments' or 'input_quality' fields. Focus on synthesizing qualitative, descriptive, and contextual information about the issuer's industry, sector, market trends, opportunities, threats, and the competitive landscape, even if not presented in a structured way.\n"
            f"If relevant information is implied or scattered across documents, synthesize it into a coherent response. If only partial or ambiguous information is present, explain your reasoning in the 'input_quality' field.\n"
            f"Additionally, ALWAYS include a field called 'input_quality' in your output. In 'input_quality', explain which documents you actually used for extraction and why, and which documents were not useful and why. Be specific about the strengths and weaknesses of each input document for the extraction task."
        )},
        {"role": "user", "content": (
            "Here are the relevant JSON documents. Extract the section as a JSON object matching the extraction schema.\n"
            f"Relevant documents: {json.dumps(docs_json, ensure_ascii=False)}"
        )}
    ]
    if repair_prompt:
        prompt.append({"role": "user", "content": f"REPAIR INSTRUCTIONS: {repair_prompt}"})
    return prompt

def get_section_config(schema: dict, section: str):
    meta = schema["properties"].get(section, {})
    return {
        "faiss_query": meta.get("faiss_query") or meta.get("faiss_query_optimal"),
        "faiss_top_k": meta.get("faiss_top_k", 5),
        "description": meta.get("description", ""),
        "properties": meta.get("properties", {}),
    }

def get_doc_schema():
    schema_path = Path(__file__).parent / "data_schema.json"
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            full_schema = json.load(f)
            doc_schema = full_schema.get("properties", {})
            if not doc_schema:
                raise ValueError("No schema sections found")
            return doc_schema
    except Exception as e:
        logger.error(f"[Schema] Failed to load schema: {e}")
        raise

def log_section_debug(sec, indices, docs_json, idx_to_file, prompt, security_id):
    debug_folder = get_data_extraction_folder(security_id)
    debug_log_path = debug_folder / f"{sec}_extraction_debug.log"
    total_docs = len(docs_json)
    file_paths = [idx_to_file.get(idx) for idx in indices if idx_to_file.get(idx)]
    with open(debug_log_path, "w", encoding="utf-8") as dbg:
        dbg.write(f"\n=== Section: {sec} ===\n")
        dbg.write(f"docs_json_count: {total_docs}\n")
        dbg.write(f"files_sent ({len(file_paths)}):\n")
        for i, (fpath, doc) in enumerate(zip(file_paths, docs_json)):
            doc_str = json.dumps(doc, ensure_ascii=False)
            snippet = doc_str[:200].replace("\n", " ")
            dbg.write(f"  [{i+1}] {fpath}\n    length: {len(doc_str)} chars\n    snippet: {snippet}...\n")
        total_chars = sum(len(json.dumps(doc, ensure_ascii=False)) for doc in docs_json)
        dbg.write(f"total_docs_char_size: {total_chars}\n")
        prompt_str = json.dumps(prompt, ensure_ascii=False, indent=2)
        dbg.write(f"prompt_total_chars: {len(prompt_str)}\n")
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            prompt_tokens = sum(len(enc.encode(m['content'])) for m in prompt if 'content' in m)
            dbg.write(f"prompt_token_estimate: {prompt_tokens}\n")
        except Exception:
            pass

def section_vector_retrieve(vector_ops_tool, security_id, query, k):
    # sync helper for clarity
    import asyncio
    return asyncio.get_event_loop().run_until_complete(
        vector_ops_tool._arun(
            security_id=security_id,
            vector_store_type="text",
            action="search_similar",
            query=query,
            k=k
        )
    )

# --- Helper Functions ---

@mcp_app.tool(name="finalize_data_extraction")
async def finalize_data_extraction(input_data: ExtractionInput) -> ExtractionOutput:
    """
    Combine all section JSONs into a final data JSON if all required sections are present, as per data_schema.json.
    Save the result to the data_extraction folder. Return missing sections if incomplete.
    """
    security_id = input_data.security_id
    schema = get_schema(input_data.schema_path)
    required_sections = schema.get("required", [])
    # Prefer the board directory with the most required section files
    from financial_analyst import security_folder_utils
    board_dirs = security_folder_utils.list_board_dirs()
    best_candidate = None
    max_found = -1
    for board_dir in board_dirs:
        candidate = board_dir / security_id / "data_extraction"
        if candidate.exists() and candidate.is_dir():
            found = sum((candidate / f"{section}.json").exists() for section in required_sections)
            if found > max_found:
                max_found = found
                best_candidate = candidate
    if best_candidate:
        extraction_folder = best_candidate
        logger.info(f"[DEBUG] (finalize_data_extraction) Using best candidate extraction_folder: {extraction_folder}")
    else:
        extraction_folder = get_data_extraction_folder(security_id)
        logger.info(f"[DEBUG] (finalize_data_extraction) Fallback extraction_folder: {extraction_folder}")
    logger.info(f"[DEBUG] extraction_folder: {extraction_folder}")
    logger.info(f"[DEBUG] required_sections: {required_sections}")
    try:
        logger.info(f"[DEBUG] files in extraction_folder: {list(extraction_folder.iterdir())}")
    except Exception as e:
        logger.warning(f"[DEBUG] Could not list extraction_folder: {e}")
    combined = {}
    missing = []
    for section in required_sections:
        section_path = extraction_folder / f"{section}.json"
        logger.info(f"[DEBUG] Checking section_path: {section_path}")
        if not section_path.exists():
            logger.warning(f"[DEBUG] Section file not found: {section_path}")
            missing.append(section)
        else:
            try:
                with open(section_path, "r", encoding="utf-8") as f:
                    loaded_data = json.load(f)
                    # Now that all section files have a consistent structure with top-level keys,
                    # we can simply extract the section data from the loaded_data
                    if isinstance(loaded_data, dict) and section in loaded_data:
                        # Extract the section data from under the section key
                        combined[section] = loaded_data[section]
                        logger.info(f"[DEBUG] Extracted section data from {section} key")
                    else:
                        # This should not happen anymore since we've fixed all section extraction functions
                        # but keeping as a fallback
                        logger.warning(f"[DEBUG] Section {section} does not have expected top-level key structure")
                        combined[section] = loaded_data
                        logger.warning(f"[DEBUG] Using raw data for section {section}")
                        
                    logger.info(f"[DEBUG] Processed section {section}, keys: {list(combined[section].keys()) if isinstance(combined[section], dict) else 'not a dict'}")
                    
                    # Log the structure to help with debugging
                    if isinstance(combined[section], dict):
                        logger.info(f"[DEBUG] Section {section} has fields: {list(combined[section].keys())}")
                    else:
                        logger.info(f"[DEBUG] Section {section} is not a dictionary")
                        
                    # Check if we have comments or unit_conversion_notes at the right level
                    if isinstance(combined[section], dict):
                        if "comments" in combined[section]:
                            logger.info(f"[DEBUG] Section {section} has comments: {combined[section]['comments'][:100]}...")
                        if "unit_conversion_notes" in combined[section]:
                            logger.info(f"[DEBUG] Section {section} has unit_conversion_notes")
                    
            except Exception as e:
                missing.append(section)
    if missing:
        msg = f"Missing section data: {missing}"
        return ExtractionOutput(status="incomplete", message=msg, output_path=None, extracted_data=None)
    # IMPORTANT: We need to maintain the proper schema structure with top-level section keys
    # The schema expects data to be organized under their respective section keys
    # NOT flattened into a single dictionary
    structured_data = {}
    for section, section_data in combined.items():
        # Preserve the section key in the output
        structured_data[section] = section_data
        logger.info(f"[DEBUG] Added {section} data to structured output")
    
    # Add security_id to the top level for easier reference
    structured_data["security_id"] = security_id
    
    # Save the properly structured data
    final_path = extraction_folder / "complete_data.json"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"[DEBUG] Successfully created complete_data.json with {len(structured_data)} fields")
    logger.info(f"[DEBUG] Top-level keys: {list(structured_data.keys())[:10]}...")
    return ExtractionOutput(
        status="success", 
        message=f"Data extraction complete with {len(structured_data)} fields", 
        output_path=str(final_path), 
        extracted_data=structured_data
    )

def get_security_folder(security_id: str) -> Path:
    return require_security_folder(security_id)

def get_data_extraction_folder(security_id: str) -> Path:
    return get_subfolder(security_id, "data_extraction")

def get_vector_store_dir(security_id: str) -> Path:
    return get_subfolder(security_id, "vector_store_txt")

def get_vector_metadata_path(security_id: str) -> Path:
    return get_vector_store_dir(security_id) / f"{security_id}_vector_metadata.json"

def get_schema(schema_path: Optional[str]) -> Dict[str, Any]:
    if schema_path is None:
        schema_path = str(Path(__file__).parent / "data_schema.json")
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_vector_metadata(security_id: str):
    metadata_path = get_vector_metadata_path(security_id)
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return json.load(f)
    return None

def get_bond_principal_amount(security_id: str) -> tuple:
    """
    Extract the principal_amount_issued from bond_metadata.json as the source of truth for unit magnitude.
    Returns a tuple of (principal_amount, currency) or (None, None) if not found.
    """
    try:
        metadata_path = os.path.join(get_data_extraction_folder(security_id), "bond_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                principal = metadata.get("bond_metadata", {}).get("principal_amount_issued")
                currency = metadata.get("bond_metadata", {}).get("currency")
                if principal:
                    logger.info(f"Found bond principal amount: {principal} {currency}")
                    return principal, currency
    except Exception as e:
        logger.warning(f"Error reading bond principal amount: {str(e)}")
    
    return None, None

def load_txt_file(txt_path: str) -> dict | None:
    """
    Loads and returns the full JSON object from a file, or None on error.
    """
    if not txt_path or not os.path.exists(txt_path):
        return None
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[Extraction] Could not load JSON from {txt_path}: {e}")
        return None

def save_section_output_json(security_id: str, section: str, data: dict) -> str:
    folder = get_data_extraction_folder(security_id)
    out_path = folder / f"{section}.json"
    
    # Always ensure data has a top-level key matching the section name
    # This guarantees consistent structure for all section files
    if section not in data:
        data = {section: data}
        logger.info(f"[Extraction] Added top-level key '{section}' to section data before saving")
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return str(out_path)



@mcp_app.tool(name="bond_metadata")
async def bond_metadata(input_data: ExtractionInput) -> ExtractionOutput:
    logger.info(f"[bond_metadata] Extraction started for security_id={input_data.security_id}")
    logger.info("CRITICAL: Ensuring all financial values, especially bond principal, are extracted in BASE UNITS (not millions/billions)")
    security_id = input_data.security_id
    schema = get_schema(input_data.schema_path)
    llm = get_gemini_llm()
    vector_metadata = load_vector_metadata(security_id)
    # Build a map from index to file path
    idx_to_file = {item["index"]: item["file"] for item in vector_metadata}
    extracted = {}
    errors = []
    # --- Section-level Extraction Strategy ---
    section_name = "bond_metadata"
    section_meta = schema["properties"].get(section_name)
    if not section_meta:
        logger.warning(f"[Extraction] Section '{section_name}' not found in schema, skipping.")
        return ExtractionOutput(
            status="error",
            message=f"Section '{section_name}' not found in schema",
            output_path=None,
            extracted_data=None
        )
    query = section_meta.get("faiss_query_optimal", section_name)
    k = section_meta.get("faiss_top_k", 5)
    # Batch search
    tool = VectorOpsTool()
    try:
        result_json = await tool._arun(
            security_id=security_id,
            vector_store_type="text",
            action="search_similar",
            query=query,
            k=k
        )
        result = json.loads(result_json)
        indices = result.get("indices") or (result.get("result", {}).get("indices") if "result" in result else None)
        if not indices:
            raise RuntimeError(f"Batch search failed or mismatched. Got: {indices}")
    finally:
        await tool.close()
    # Prepend authoritative metadata files if present
    docs_json = []
    security_folder = get_security_folder(security_id)
    meta_files = [security_folder / "security_details.json", security_folder / "security_metadata.json"]
    seen = set()
    for f in meta_files:
        if f.exists():
            data = load_txt_file(str(f))
            if data is not None:
                docs_json.append(data)
                seen.add(f.name)
    # Add RAG results, skipping duplicates
    for idx in indices:
        fpath = idx_to_file.get(idx)
        fname = Path(fpath).name if fpath else None
        if fname and fname in seen:
            continue
        data = load_txt_file(fpath)
        if data is not None:
            docs_json.append(data)
            seen.add(fname)
    # Only pass the section's 'properties' dict to the prompt
    section_properties = section_meta.get("properties", {})
    if not section_properties:
        logger.warning(f"[Extraction] Section '{section_name}' has no 'properties' in schema, skipping.")
        return ExtractionOutput(
            status="error",
            message=f"Section '{section_name}' has no 'properties' in schema",
            output_path=None,
            extracted_data=None
        )
    # Load schema from data_schema.json in the same directory
    schema_path = Path(__file__).parent / "data_schema.json"
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            full_schema = json.load(f)
            doc_schema = full_schema.get("properties", {}).get("bond_metadata", {})
            if not doc_schema:
                raise ValueError("bond_metadata schema section not found")
    except Exception as e:
        logger.error(f"[Extraction] Failed to load bond_metadata schema: {e}")
        raise
    # --- Logging ---
    debug_folder = get_data_extraction_folder(security_id)
    debug_log_path = debug_folder / "bond_metadata_extraction_debug.log"
    total_docs = len(docs_json)
    file_paths = [idx_to_file.get(idx) for idx in indices if idx_to_file.get(idx)]
    
    # --- Unit Normalization Reminder ---
    logger.info("REMINDER: Bond principal amount MUST be in base units (not millions/billions)")
    logger.info("Example: If principal is stated as '1 billion MNT', it must be extracted as 1,000,000,000 MNT")
    with open(debug_log_path, "w", encoding="utf-8") as dbg:
        dbg.write(f"\n=== Section: {section_name} ===\n")
        dbg.write(f"docs_json_count: {total_docs}\n")
        dbg.write(f"files_sent ({len(file_paths)}):\n")
        for i, (fpath, doc) in enumerate(zip(file_paths, docs_json)):
            doc_str = json.dumps(doc, ensure_ascii=False)
            snippet = doc_str[:200].replace("\n", " ")
            dbg.write(f"  [{i+1}] {fpath}\n    length: {len(doc_str)} chars\n    snippet: {snippet}...\n")
        total_chars = sum(len(json.dumps(doc, ensure_ascii=False)) for doc in docs_json)
        dbg.write(f"total_docs_char_size: {total_chars}\n")
        
        # Use the specialized bond metadata prompt builder with emphasis on principal amount extraction
        # Pass security_id to use Total Amount Issued from security_details.json as deterministic source of truth
        prompt = build_bond_metadata_prompt(section_name, section_meta.get('description', ''), section_properties, docs_json, doc_schema, security_id=security_id, repair_prompt=input_data.repair_prompt)
        
        prompt_str = json.dumps(prompt, ensure_ascii=False, indent=2)
        dbg.write(f"prompt_total_chars: {len(prompt_str)}\n")
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            prompt_tokens = sum(len(enc.encode(m['content'])) for m in prompt if 'content' in m)
            dbg.write(f"prompt_token_estimate: {prompt_tokens}\n")
        except Exception:
            pass
    result = await extract_section(llm, section_name, section_meta.get("description", ""), section_properties, docs_json, doc_schema, prompt_messages=prompt)
    logger.info(f"[Extraction] LLM result for section '{section_name}': {str(result)[:200]}...")
    # Deterministic patch: fill issue_date from Closing Date in security_details.json if present
    try:
        security_folder = get_security_folder(security_id)
        sec_details_path = security_folder / "security_details.json"
        if sec_details_path.exists():
            with open(sec_details_path, "r", encoding="utf-8") as f:
                sec_details = json.load(f)
            closing_date = sec_details.get("Closing Date")
            if closing_date:
                if isinstance(result, dict) and 'issue_date' in result:
                    result['issue_date'] = closing_date
                elif isinstance(result, dict) and 'bond_metadata' in result and isinstance(result['bond_metadata'], dict):
                    result['bond_metadata']['issue_date'] = closing_date
    except Exception as e:
        logger.warning(f"[Deterministic issue_date patch] Failed: {e}")
    # Ensure 'comments' field is present
    if isinstance(result, dict) and 'comments' not in result:
        result['comments'] = 'No LLM comments provided.'
    output_path = save_section_output_json(security_id, section_name, result)
    return ExtractionOutput(
        status="success",
        message="Extraction complete",
        output_path=output_path,
        extracted_data={section_name: result}
    )

@mcp_app.tool(name="bond_financials_historical")
async def bond_financials_historical(input_data: ExtractionInput) -> ExtractionOutput:
    logger.info(f"[bond_financials_historical] Extraction started for security_id={input_data.security_id}")
    logger.info("CRITICAL: All financial statement values MUST be extracted in BASE UNITS (not millions/billions)")
    security_id = input_data.security_id
    schema = get_schema(input_data.schema_path)
    section_name = "bond_financials_historical"
    section_meta = schema["properties"].get(section_name)
    if not section_meta:
        logger.warning(f"[Extraction] Section '{section_name}' not found in schema, skipping.")
        return ExtractionOutput(status="error", message=f"Section '{section_name}' not found in schema", output_path=None, extracted_data=None)
    section_properties = section_meta.get("properties", {})
    if not section_properties:
        logger.warning(f"[Extraction] Section '{section_name}' has no 'properties' in schema, skipping.")
        return ExtractionOutput(status="error", message=f"Section '{section_name}' has no 'properties' in schema", output_path=None, extracted_data=None)
    # Deterministic search for financial tables
    security_folder = get_security_folder(security_id)
    parsed_images_dir = security_folder / "parsed_images"
    docs_json = []
    used_files = []
    # Refined deterministic selection: prioritize comprehensive, structured financial statements
    max_files = 15  # Fewer, more relevant docs
    table_keywords = [
        # English
        "Balance Sheet", "Income Statement", "Cash Flow", "Statement of Cash Flows", "Full Cash Flow Statement", "Cash Flow Details", "Depreciation", "Amortization", "Changes in Working Capital", "Receivables", "Payables", "Inventory", "Non-Cash Items", "Consolidated", "Summary", "Annual", "Semi-Annual",
        # Mongolian
        "Баланс", "Орлогын тайлан", "Мөнгөн урсгал", "Мөнгөн хөрөнгийн тайлан", "Бүрэн мөнгөн урсгалын тайлан", "Мөнгөн урсгалын дэлгэрэнгүй", "Элэгдэл", "Элэгдэл хорогдол", "Ашиглалтын өөрчлөлт", "Ажил гүйлгээний өөрчлөлт", "Авлага", "Өглөг", "Бараа материал", "Бэлэн бус зүйлс", "Нэгдсэн", "Хураангуй", "Жилийн", "Хагас жилийн"
    ]
    # Exclusion words logic removed for broader selection
    selected_count = 0
    if parsed_images_dir.exists():
        stop_selection = False
        for fname in sorted(os.listdir(parsed_images_dir)):
            if stop_selection:
                break
            if not fname.endswith(".json"):
                continue
            fpath = parsed_images_dir / fname
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                blocks = doc.get("content_blocks", [])
                for block in blocks:
                    if block.get("type") != "table":
                        continue
                    content_rows = block.get("content")
                    if not isinstance(content_rows, list) or not content_rows:
                        continue
                    first_row = content_rows[0]
                    if not isinstance(first_row, dict):
                        continue
                    keys = list(first_row.keys())
                    year_keys = [k for k in keys if isinstance(k, str) and k.strip().isdigit() and len(k.strip()) == 4]
                    metric_keywords = [
                        # English
                        "Assets", "Equity", "Net Profit", "Revenue", "Cash Flow", "Operating Activities", "Investing Activities", "Financing Activities", "Net Cash", "Cash from Operations", "Cash from Investing", "Cash from Financing", "Operating Cash Flow", "Free Cash Flow", "Full Cash Flow Statement", "Cash Flow Details", "Depreciation", "Amortization", "Changes in Working Capital", "Receivables", "Payables", "Inventory", "Non-Cash Items",
                        # Mongolian
                        "Өөрийн хөрөнгө", "Харьцаа үзүүлэлтүүд", "Мөнгөн урсгал", "Үйл ажиллагааны мөнгөн урсгал", "Хөрөнгө оруулалтын үйл ажиллагааны мөнгөн урсгал", "Санхүүгийн үйл ажиллагааны мөнгөн урсгал", "Цэвэр мөнгөн урсгал", "Үйл ажиллагааны мөнгөн урсгалын өөрчлөлт", "Элэгдэл", "Элэгдэл хорогдол", "Ашиглалтын өөрчлөлт", "Ажил гүйлгээний өөрчлөлт", "Авлага", "Өглөг", "Бараа материал", "Бэлэн бус зүйлс"
                    ]
                    metric_keys = [k for k in keys if any(m in k for m in metric_keywords)]
                    # Refined selection: prioritize blocks with table_keywords in title or keys, and multi-period
                    block_title = block.get("title", "")
                    block_title_lc = block_title.lower()
                    has_table_keyword = any(kw.lower() in block_title_lc for kw in table_keywords) or any(
                        kw.lower() in k.lower() for kw in table_keywords for k in keys)
                    # Loosened: select if (table keyword OR metric keyword OR >=2 year columns)
                    if has_table_keyword or metric_keys or len(year_keys) >= 2:
                        docs_json.append(doc)
                        used_files.append(str(fpath))
                        selected_count += 1
                        logger.info(f"[Extraction] Selected {fpath} for bond_financials_historical: year_keys={year_keys}, metric_keys={metric_keys}, table_keyword={has_table_keyword}")
                        if selected_count >= max_files:
                            logger.warning(f"[Extraction] Hit max_files={max_files} for bond_financials_historical. Stopping further selection.")
                            stop_selection = True
                            break
            except Exception as e:
                logger.warning(f"[Extraction] Could not process {fpath}: {e}")
    # Load schema from data_schema.json in the same directory
    schema_path = Path(__file__).parent / "data_schema.json"
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            full_schema = json.load(f)
            doc_schema = full_schema.get("properties", {}).get("bond_financials_historical", {})
            if not doc_schema:
                raise ValueError("bond_financials_historical schema section not found")
    except Exception as e:
        logger.error(f"[Extraction] Failed to load bond_financials_historical schema: {e}")
        raise

    llm = get_gemini_llm()
    # Logging
    debug_folder = get_data_extraction_folder(security_id)
    debug_log_path = debug_folder / "bond_financials_historical_extraction_debug.log"
    with open(debug_log_path, "w", encoding="utf-8") as dbg:
        dbg.write(f"\n=== Section: {section_name} ===\n")
        dbg.write(f"docs_json_count: {len(docs_json)}\n")
        dbg.write(f"schema_loaded: {bool(doc_schema)}\n")
        dbg.write(f"files_sent ({len(used_files)}):\n")
        for i, fpath in enumerate(used_files):
            dbg.write(f"  [{i+1}] {fpath}\n")
    # Get principal amount from bond_metadata as source of truth for unit normalization
    principal_amount, currency = get_bond_principal_amount(security_id)
    if principal_amount and currency:
        logger.info(f"[Extraction] Using bond principal amount as unit reference: {principal_amount} {currency}")
    else:
        logger.warning(f"[Extraction] No bond principal amount found for reference. Unit consistency may be affected.")
    
    # Pass security_id to build_section_prompt to use principal amount as source of truth
    result = await extract_section(llm, section_name, section_meta.get("description", ""), section_properties, docs_json, doc_schema, prompt_messages=build_section_prompt(section_name, section_meta.get("description", ""), section_properties, docs_json, doc_schema, security_id=security_id, repair_prompt=input_data.repair_prompt))
    logger.info(f"[Extraction] LLM result for section '{section_name}': {str(result)[:200]}...")

    # --- Deterministic unit normalization & validation ---
    if section_name in ["bond_financials_historical", "bond_financials_projections"]:
        # Check if base_unit_of_measure exists, if not add it
        if "base_unit_of_measure" not in result:
            result["base_unit_of_measure"] = "MNT"  # Default fallback
            logger.warning(f"[Extraction] No base_unit_of_measure detected in {section_name}, defaulted to 'MNT'")
        
        # Ensure unit_conversion_notes exists
        if "unit_conversion_notes" not in result:
            result["unit_conversion_notes"] = ""  # Initialize if missing
        
        # Check all financials dicts for numeric consistency
        hist_fin = result.get("historical_financial_statements") or result.get("projections")
        if isinstance(hist_fin, list):
            warnings = []
            for entry in hist_fin:
                fin = entry.get("financials", {})
                if not isinstance(fin, dict):
                    continue
                # Scan for values that might be in millions/billions instead of base units
                for k, v in fin.items():
                    if not isinstance(v, (int, float)):
                        warnings.append(f"WARNING: Non-numeric value for {k}: {v}")
            
            # Only add warnings if they don't already exist in the notes
            if warnings:
                existing_notes = result["unit_conversion_notes"]
                for warning in warnings:
                    if warning not in existing_notes:
                        if existing_notes and not existing_notes.endswith("\n"):
                            existing_notes += "\n"
                        existing_notes += warning
                result["unit_conversion_notes"] = existing_notes

    output_path = save_section_output_json(security_id, section_name, result)
    return ExtractionOutput(status="success", message="Extraction complete", output_path=output_path, extracted_data={section_name: result})

@mcp_app.tool(name="bond_financials_projections")
async def bond_financials_projections(input_data: ExtractionInput) -> ExtractionOutput:
    logger.info(f"[bond_financials_projections] Extraction started for security_id={input_data.security_id}")
    logger.info("CRITICAL: All projection values MUST be extracted in BASE UNITS (not millions/billions)")
    logger.info("Ensure consistency between historical financial units and projection units")
    security_id = input_data.security_id
    schema = get_schema(input_data.schema_path)
    section_name = "bond_financials_projections"
    section_meta = schema["properties"].get(section_name)
    if not section_meta:
        logger.warning(f"[Extraction] Section '{section_name}' not found in schema, skipping.")
        return ExtractionOutput(status="error", message=f"Section '{section_name}' not found in schema", output_path=None, extracted_data=None)
    section_properties = section_meta.get("properties", {})
    if not section_properties:
        logger.warning(f"[Extraction] Section '{section_name}' has no 'properties' in schema, skipping.")
        return ExtractionOutput(status="error", message=f"Section '{section_name}' has no 'properties' in schema", output_path=None, extracted_data=None)
    # Deterministic search for projection tables (analogous to historical)
    security_folder = get_security_folder(security_id)
    parsed_images_dir = security_folder / "parsed_images"
    docs_json = []
    used_files = []
    max_files = 12
    # Only strong forward-looking/projection keywords
    table_keywords = [
        # English
        "Projection", "Forecast", "Planned", "Budget",
        # Mongolian
        "Төлөвлөгөө", "Төсөөлөл", "Урьдчилсан", "Хэтийн төлөв", "Прогноз"
    ]
    selected_count = 0
    from datetime import datetime
    current_year = datetime.now().year
    if parsed_images_dir.exists():
        stop_selection = False
        for fname in sorted(os.listdir(parsed_images_dir)):
            if stop_selection:
                break
            if not fname.endswith(".json"):
                continue
            fpath = parsed_images_dir / fname
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                blocks = doc.get("content_blocks", [])
                for block in blocks:
                    if block.get("type") != "table":
                        continue
                    content_rows = block.get("content")
                    if not isinstance(content_rows, list) or not content_rows:
                        continue
                    first_row = content_rows[0]
                    if not isinstance(first_row, dict):
                        continue
                    keys = list(first_row.keys())
                    # Loosened: any column key with a 4+ digit number >= current year (any format)
                    future_col = False
                    for k in keys:
                        if not isinstance(k, str):
                            continue
                        for token in k.replace("/", " ").replace("-", " ").split():
                            if token.isdigit() and len(token) >= 4 and int(token) >= current_year:
                                future_col = True
                                break
                        if future_col:
                            break
                    block_title = block.get("title", "")
                    block_title_lc = block_title.lower()
                    has_table_keyword = any(kw.lower() in block_title_lc for kw in table_keywords) or any(
                        kw.lower() in k.lower() for kw in table_keywords for k in keys)
                    # Select if (projection keyword in title/keys OR any future-looking column)
                    if has_table_keyword or future_col:
                        docs_json.append(doc)
                        used_files.append(str(fpath))
                        selected_count += 1
                        logger.info(f"[Extraction] Selected {fpath} for bond_financials_projections: future_col={future_col}, table_keyword={has_table_keyword}")
                        if selected_count >= max_files:
                            logger.warning(f"[Extraction] Hit max_files={max_files} for bond_financials_projections. Stopping further selection.")
                            stop_selection = True
                            break
            except Exception as e:
                logger.warning(f"[Extraction] Could not process {fpath}: {e}")
    doc_schema = get_doc_schema()
    llm = get_gemini_llm()
    
    # Get principal amount from bond_metadata as source of truth for unit normalization
    principal_amount, currency = get_bond_principal_amount(security_id)
    if principal_amount and currency:
        logger.info(f"[Extraction] Using bond principal amount as unit reference: {principal_amount} {currency}")
    else:
        logger.warning(f"[Extraction] No bond principal amount found for reference. Unit consistency may be affected.")
    
    # Pass security_id to build_section_prompt to use principal amount as source of truth
    prompt = build_section_prompt(section_name, section_meta.get('description', ''), section_properties, docs_json, doc_schema, security_id=security_id, repair_prompt=input_data.repair_prompt)
    log_section_debug(section_name, list(range(len(docs_json))), docs_json, {i: f for i, f in enumerate(used_files)}, prompt, security_id)
    result = await extract_section(llm, section_name, section_meta.get("description", ""), section_properties, docs_json, doc_schema, prompt_messages=prompt)
    logger.info(f"[Extraction] LLM result for section '{section_name}': {str(result)[:200]}...")
    output_path = save_section_output_json(security_id, section_name, result)
    return ExtractionOutput(status="success", message="Extraction complete", output_path=output_path, extracted_data={section_name: result})

@mcp_app.tool(name="collateral_and_protective_clauses")
async def collateral_and_protective_clauses(input_data: ExtractionInput) -> ExtractionOutput:
    logger.info(f"[collateral_and_protective_clauses] Extraction started for security_id={input_data.security_id}")
    security_id = input_data.security_id
    schema = get_schema(input_data.schema_path)
    llm = get_gemini_llm()
    vector_metadata = load_vector_metadata(security_id)
    idx_to_file = {item["index"]: item["file"] for item in vector_metadata}
    section_name = "collateral_and_protective_clauses"
    section_meta = schema["properties"].get(section_name)
    if not section_meta:
        logger.warning(f"[Extraction] Section '{section_name}' not found in schema, skipping.")
        return ExtractionOutput(status="error", message=f"Section '{section_name}' not found in schema", output_path=None, extracted_data=None)
    query = section_meta.get("faiss_query_optimal", section_name)
    k = 10
    tool = VectorOpsTool()
    try:
        result_json = await tool._arun(security_id=security_id, vector_store_type="text", action="search_similar", query=query, k=k)
        result = json.loads(result_json)
        indices = result.get("indices") or (result.get("result", {}).get("indices") if "result" in result else None)
        if not indices:
            raise RuntimeError(f"Batch search failed or mismatched. Got: {indices}")
    finally:
        await tool.close()
    docs_json = [load_txt_file(idx_to_file.get(idx)) for idx in indices if idx_to_file.get(idx)]
    docs_json = [doc for doc in docs_json if doc is not None]
    # Always attach security_details.json if present (improves recall for collateral info)
    sec_details_path = get_security_folder(security_id) / 'security_details.json'
    if sec_details_path.exists():
        sec_details = load_txt_file(str(sec_details_path))
        if sec_details is not None:
            docs_json.append(sec_details)
    section_properties = section_meta.get("properties", {})
    if not section_properties:
        logger.warning(f"[Extraction] Section '{section_name}' has no 'properties' in schema, skipping.")
        return ExtractionOutput(status="error", message=f"Section '{section_name}' has no 'properties' in schema", output_path=None, extracted_data=None)
    doc_schema = get_doc_schema()
    collateral_prompt = build_collateral_prompt(section_name, section_meta.get('description', ''), section_properties, docs_json, doc_schema, repair_prompt=input_data.repair_prompt)
    log_section_debug(section_name, indices, docs_json, idx_to_file, collateral_prompt, security_id)
    result = await extract_section(llm, section_name, section_meta.get("description", ""), section_properties, docs_json, doc_schema, prompt_messages=collateral_prompt)
    logger.info(f"[Extraction] LLM result for section '{section_name}': {str(result)[:200]}...")
    output_path = save_section_output_json(security_id, section_name, result)
    return ExtractionOutput(status="success", message="Extraction complete", output_path=output_path, extracted_data={section_name: result})

@mcp_app.tool(name="issuer_business_profile")
async def issuer_business_profile(input_data: ExtractionInput) -> ExtractionOutput:
    logger.info(f"[issuer_business_profile] Extraction started for security_id={input_data.security_id}")
    security_id = input_data.security_id
    schema = get_schema(input_data.schema_path)
    llm = get_gemini_llm()
    vector_metadata = load_vector_metadata(security_id)
    idx_to_file = {item["index"]: item["file"] for item in vector_metadata}
    section_name = "issuer_business_profile"
    section_meta = schema["properties"].get(section_name)
    if not section_meta:
        logger.warning(f"[Extraction] Section '{section_name}' not found in schema, skipping.")
        return ExtractionOutput(status="error", message=f"Section '{section_name}' not found in schema", output_path=None, extracted_data=None)
    query = section_meta.get("faiss_query_optimal", section_name)
    k = 10
    tool = VectorOpsTool()
    try:
        result_json = await tool._arun(security_id=security_id, vector_store_type="text", action="search_similar", query=query, k=k)
        result = json.loads(result_json)
        indices = result.get("indices") or (result.get("result", {}).get("indices") if "result" in result else None)
        if not indices:
            raise RuntimeError(f"Batch search failed or mismatched. Got: {indices}")
    finally:
        await tool.close()
    docs_json = [load_txt_file(idx_to_file.get(idx)) for idx in indices if idx_to_file.get(idx)]
    docs_json = [doc for doc in docs_json if doc is not None]
    section_properties = section_meta.get("properties", {})
    if not section_properties:
        logger.warning(f"[Extraction] Section '{section_name}' has no 'properties' in schema, skipping.")
        return ExtractionOutput(status="error", message=f"Section '{section_name}' has no 'properties' in schema", output_path=None, extracted_data=None)
    doc_schema = get_doc_schema()
    issuer_prompt = build_issuer_profile_prompt(section_name, section_meta.get('description', ''), section_properties, docs_json, doc_schema, repair_prompt=input_data.repair_prompt)
    log_section_debug(section_name, indices, docs_json, idx_to_file, issuer_prompt, security_id)
    result = await extract_section(llm, section_name, section_meta.get("description", ""), section_properties, docs_json, doc_schema, prompt_messages=issuer_prompt)
    logger.info(f"[Extraction] LLM result for section '{section_name}': {str(result)[:200]}...")
    output_path = save_section_output_json(security_id, section_name, result)
    return ExtractionOutput(status="success", message="Extraction complete", output_path=output_path, extracted_data={section_name: result})

@mcp_app.tool(name="industry_profile")
async def industry_profile(input_data: ExtractionInput) -> ExtractionOutput:
    logger.info(f"[industry_profile] Extraction started for security_id={input_data.security_id}")
    security_id = input_data.security_id
    schema = get_schema(input_data.schema_path)
    llm = get_gemini_llm()
    vector_metadata = load_vector_metadata(security_id)
    idx_to_file = {item["index"]: item["file"] for item in vector_metadata}
    section_name = "industry_profile"
    section_meta = schema["properties"].get(section_name)
    if not section_meta:
        logger.warning(f"[Extraction] Section '{section_name}' not found in schema, skipping.")
        return ExtractionOutput(status="error", message=f"Section '{section_name}' not found in schema", output_path=None, extracted_data=None)
    query = section_meta.get("faiss_query_optimal", section_name)
    k = 10
    tool = VectorOpsTool()
    try:
        result_json = await tool._arun(security_id=security_id, vector_store_type="text", action="search_similar", query=query, k=k)
        result = json.loads(result_json)
        indices = result.get("indices") or (result.get("result", {}).get("indices") if "result" in result else None)
        if not indices:
            raise RuntimeError(f"Batch search failed or mismatched. Got: {indices}")
    finally:
        await tool.close()
    docs_json = [load_txt_file(idx_to_file.get(idx)) for idx in indices if idx_to_file.get(idx)]
    docs_json = [doc for doc in docs_json if doc is not None]
    section_properties = section_meta.get("properties", {})
    if not section_properties:
        logger.warning(f"[Extraction] Section '{section_name}' has no 'properties' in schema, skipping.")
        return ExtractionOutput(status="error", message=f"Section '{section_name}' has no 'properties' in schema", output_path=None, extracted_data=None)
    doc_schema = get_doc_schema()
    industry_prompt = build_industry_profile_prompt(section_name, section_meta.get('description', ''), section_properties, docs_json, doc_schema, repair_prompt=input_data.repair_prompt)
    log_section_debug(section_name, indices, docs_json, idx_to_file, industry_prompt, security_id)
    result = await extract_section(llm, section_name, section_meta.get("description", ""), section_properties, docs_json, doc_schema, prompt_messages=industry_prompt)
    logger.info(f"[Extraction] LLM result for section '{section_name}': {str(result)[:200]}...")
    output_path = save_section_output_json(security_id, section_name, result)
    return ExtractionOutput(status="success", message="Extraction complete", output_path=output_path, extracted_data={section_name: result})

import re

def extract_json_from_response(response_str: str):
    # Remove code block wrappers if present
    cleaned = response_str.strip()
    if cleaned.startswith('```json'):
        cleaned = cleaned[7:]
    if cleaned.startswith('```'):
        cleaned = cleaned[3:]
    if cleaned.endswith('```'):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    # Try to extract the first valid JSON object or array from the string
    json_pattern = re.compile(r'({[\s\S]*?}|\[[\s\S]*?\])', re.MULTILINE)
    matches = json_pattern.findall(cleaned)
    for match in matches:
        try:
            return json.loads(match)
        except Exception:
            continue
    # Try parsing the whole cleaned string as JSON
    try:
        return json.loads(cleaned)
    except Exception:
        # Return an empty dictionary with error information instead of the raw string
        logger.error(f"Failed to parse JSON response: {cleaned[:200]}...")
        return {
            "base_unit_of_measure": "MNT",
            "unit_conversion_notes": "Error parsing LLM response as JSON",
            "comments": f"ERROR: Failed to parse LLM response as JSON. Raw response: {cleaned[:200]}..."
        }

async def extract_section(llm, section_name: str, section_desc: str, section_properties: dict, docs_json: list, doc_schema: dict, prompt_messages=None) -> Any:
    import time
    prompt = prompt_messages if prompt_messages is not None else build_section_prompt(section_name, section_desc, section_properties, docs_json, doc_schema)

    # Log prompt size for debugging
    prompt_str = json.dumps(prompt, ensure_ascii=False, indent=2)
    logger.info(f"[Extraction] Prompt total characters: {len(prompt_str)}")
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        prompt_tokens = sum(len(enc.encode(m['content'])) for m in prompt if 'content' in m)
        logger.info(f"[Extraction] Prompt token estimate: {prompt_tokens}")
    except Exception:
        logger.warning("[Extraction] Could not estimate prompt tokens (tiktoken not available)")

    logger.info(f"[Extraction] [{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting LLM call for section '{section_name}'...")
    start_llm = time.time()
    response = await llm.ainvoke(prompt)
    end_llm = time.time()
    logger.info(f"[Extraction] [{time.strftime('%Y-%m-%d %H:%M:%S')}] LLM call complete. Elapsed: {end_llm-start_llm:.2f}s")

    logger.info(f"[Extraction] [{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting response parsing...")
    start_parse = time.time()
    extracted_json = extract_json_from_response(response.content)
    end_parse = time.time()
    logger.info(f"[Extraction] [{time.strftime('%Y-%m-%d %H:%M:%S')}] Response parsing complete. Elapsed: {end_parse-start_parse:.2f}s")
    
    # Validate unit normalization for financial sections
    if section_name in ["bond_metadata", "bond_financials_historical", "bond_financials_projections"] and isinstance(extracted_json, dict):
        # Log unit conversion information
        unit_notes = extracted_json.get("unit_conversion_notes")
        base_unit = extracted_json.get("base_unit_of_measure")
        logger.info(f"[Extraction] Unit normalization for {section_name}: Base unit = {base_unit}")
        if unit_notes:
            logger.info(f"[Extraction] Unit conversion notes: {unit_notes[:200]}...")
        
        # Check for bond principal in base units if this is bond_metadata
        if section_name == "bond_metadata":
            principal = extracted_json.get("principal_amount_issued")
            if principal and principal < 100000:
                logger.warning(f"[Extraction] POTENTIAL UNIT ERROR: Bond principal ({principal}) appears too small, may not be in base units")
            elif principal and principal > 100000000:
                logger.info(f"[Extraction] Bond principal ({principal}) appears to be in base units")
        
        # Check for comments about unit normalization
        comments = extracted_json.get("comments", "")
        if "unit" in comments.lower() or "million" in comments.lower() or "billion" in comments.lower():
            logger.info(f"[Extraction] Unit-related comments found: {comments[:200]}...")
    
    return extracted_json

async def run_stdio_server():
    await mcp_app.run_stdio_async()

def main():
    logger.info("Starting MCP Data Extractor server main loop")
    import asyncio
    asyncio.run(run_stdio_server())

if __name__ == "__main__":
    main()
