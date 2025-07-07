"""
MCP Server for Structured Bond Data Extraction

This server extracts structured data from bond documents with a strong focus on unit normalization.
All financial values are converted to base units (not millions or billions) during extraction.
"""
import asyncio
import json
import os
import re
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
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

# HistoricalDebtScheduleItem model has been moved to data_schema.json

class ExtractionOutput(BaseModel):
    status: str
    message: str
    output_path: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None

# --- Section Extraction Helpers ---

def build_section_prompt(section_name, section_desc, section_properties, docs_json, security_id=None, repair_prompt=None):
    """
    Build an LLM prompt for extracting a section. If repair_prompt is provided, append it as a user message for repair/quality control.
    For all financial sections, explicitly instruct unit normalization to ensure consistent base units.
    If security_id is provided and section is financial, try to use principal amount as source of truth for units.
    """
    principal_amount, currency = get_bond_principal_amount(security_id)

    # logger.info(f"section_name: {section_name}, section_desc: {section_desc}, section_properties: {section_properties}")
    
    if principal_amount is not None and currency is not None:
        # Calculate magnitude (millions or billions)
        magnitude = "billions" if principal_amount >= 1000000000 else "millions" if principal_amount >= 1000000 else "thousands"
        
        unit_instructions = (
            f"1. **ULTIMATE REFERENCE FOR SCALE & TARGET OUTPUT MAGNITUDE**: The bond principal amount is **{principal_amount} {currency}**. This is the absolute numerical value of the bond in **RAW {currency} BASE UNITS**. You MUST use this exact numerical value as the **definitive scale reference** for ALL other financial figures. Your output numbers for items like 'total_assets', 'total_revenue', etc., MUST numerically align with the magnitude of this principal amount (e.g., if principal is 1,000,000,000, expected assets are in the billions range, not millions, even if initially presented in millions in the source. You MUST convert them up to the billion scale). This is the single most important instruction for scaling.\n"
            f"2. **TARGET OUTPUT FORMAT**: Your numerical output for ALL financial values MUST be the actual, raw {currency} number (e.g., `1,234,567,890.0` for 1.234 billion {currency}, not `1.234`). Do NOT scale down or round for 'millions' or 'billions' in the final output number itself. The output numbers should be large, absolute {currency} values.\n"
            f"3. **COMPREHENSIVE CONVERSION STEPS (to achieve absolute RAW {currency} BASE UNITS)**:\n"
            f"   a. **Explicit MNT Units (e.g., 'сая төгрөгөөр'/'million MNT', 'тэрбум төгрөг'/'billion MNT', 'мянга төгрөг'/'thousands of MNT')**: If the source explicitly states MNT units, convert the *presented number* to raw MNT first (multiply by 1,000,000 for 'million', 1,000,000,000 for 'billion', or 1,000 for 'thousands' respectively). If TARGET OUTPUT UNIT ({currency}) is NOT MNT: Then convert this raw MNT value to raw {currency} using an appropriate exchange rate (e.g., `value_in_MNT / USD_MNT_Exchange_Rate`). *You must use the FX rate from the provided macroeconomic drivers if available, otherwise state in notes that MNT values were converted at an assumed rate or could not be converted if no rate is available.* \n"
            f"   b. **Explicit {currency} Units (e.g., 'million USD', 'billion USD', 'thousands of USD')**: If the source explicitly states units of '{currency}', convert the *presented number* to raw {currency} by multiplying by the corresponding factor (1,000,000 for million, 1,000,000,000 for billion, 1,000 for thousands).\n"
            f"   c. **CRITICAL: IMPLICIT/UNSTATED UNITS & MAGNITUDE CORRECTION**: This is the most important step for ensuring correct scaling. If a financial value is presented as a number *without an explicit unit*, OR if its explicitly stated unit leads to a magnitude that is *significantly misaligned* with the `principal_amount` (e.g., 1 billion bond principal, but 39 million total assets *after explicit conversion*), you **MUST infer an additional implied original scale** to bring it to the absolute RAW {currency} BASE UNITS, aligning with the `principal_amount`.\n"
            f"      - **Inference Priority (for numbers without explicit units, or numbers that are too small after explicit unit conversion):** Compare the numerical value you extracted (let's call it `extracted_value`) to the `principal_amount`.\n"
            f"          i. **If `extracted_value` is roughly 1/1,000,000th of `principal_amount` (or `principal_amount` / 1,000,000 times smaller):** It's highly likely the document implicitly presented this number in 'millions' but meant it to be interpreted as 'raw units' (i.e., '78' was meant to be '78 billion'). In this case, you **MUST multiply `extracted_value` by 1,000,000,000 (one billion)** to bring it to the raw {currency} base units. This is often the case for summary figures in millions which represent billions.\n"
            f"          ii. **If `extracted_value` is roughly 1/1,000th of `principal_amount` (or `principal_amount` / 1,000 times smaller):** It's highly likely the document implicitly presented this number in 'thousands'. You **MUST multiply `extracted_value` by 1,000,000 (one million)** to bring it to the raw {currency} base units.\n"
                    f"          iii. **If `extracted_value` is roughly 1/1,000,000th of `principal_amount` (or `principal_amount` / 1,000,000 times smaller) AND the context heavily suggests it's in millions:** This is tricky. If a number like '39' appears, and it is 1/25th of the principal (1 billion), it's highly likely the document meant '39 billion', but presented it as '39' in a context where 'million' is assumed for the whole document (leading to the original `39,000,000` issue). You **MUST then multiply `extracted_value` by the necessary factor (e.g., 1,000,000,000 if 39 is 39 billion) to match the principal's scale.** This means the document implicitly presented '39 billion' as '39 million' for some reason.\n"
            f"          iv. **If `extracted_value` already aligns closely with `principal_amount`'s magnitude:** Assume it's already in base units and no further multiplication (beyond currency conversion if needed) is required.\n"
            f"      - **Example (if {principal_amount} is 1,000,000,000 MNT):**\n"
            f"          - If you see a table labeled 'сая төгрөгөөр' (million MNT) and a number is '78,000'. The raw value is `78,000 * 1,000,000 = 78,000,000,000` MNT. This number now aligns with the billion-level principal.\n"
            f"          - If you see a paragraph stating 'Total Assets was 39 million MNT' and the bond is 1 billion MNT. This '39 million MNT' figure (39,000,000) is too small relative to a 1 billion bond. This indicates the *document itself* is implicitly scaling figures, and '39 million MNT' actually *represents* '39 billion MNT'. You **MUST convert this to 39,000,000,000 MNT** to align with the bond's scale. The instruction is to *force* the numbers to align.\n"
            f"          - If you see '226.27' in a summary table without explicit units, and the principal is 1,000,000,000, then `226.27` is far too small. It's likely `226.27` represents `226.27 billion`. You **MUST multiply `226.27` by 1,000,000,000** to get `226,270,000,000` MNT. This is the crucial inference.\n"
            f"4. **CONSISTENCY ACROSS PERIODS**: All financial values (e.g., total assets, revenue, expenses) for all periods must be in these **same absolute RAW {currency} BASE UNITS**.\n"
            f"5. **CONFLICT RESOLUTION AND PRIORITIZATION**: If you encounter conflicting scales or numbers for the same period (e.g., one document implies millions, another implies billions for the same item), you **MUST prioritize the value that can be most plausibly scaled to align with the magnitude of the {principal_amount}**. Your goal is to **always make the extracted figures align with the principal's magnitude**, even if it requires assuming an extra implicit scaling factor beyond what's explicitly stated.\n"
        )
    else:
        unit_instructions = ""
        
    financial_sanity_instructions = (
        "CRITICAL - FINANCIAL SANITY CHECKS:\n"
        "1. ALWAYS compare each financial value to the bond principal amount to ensure consistent magnitude\n"
        "2. ALL financial values within the same statement MUST use the same magnitude as the principal amount\n"
        "3. If you find values that differ by factors of 1000 or 1000000, this indicates a unit conversion issue that MUST be resolved\n"
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
    extraction_folder = get_data_extraction_folder(security_id)
    
    combined = {}
    missing = []
    
    for section in required_sections:
        section_path = extraction_folder / f"{section}.json"
        if not section_path.exists():
            missing.append(section)
            continue
            
        try:
            with open(section_path, "r", encoding="utf-8") as f:
                loaded_data = json.load(f)
                # Handle different possible JSON structures
                if isinstance(loaded_data, dict):
                    if section in loaded_data:
                        combined[section] = loaded_data[section]
                    else:
                        # If the section key doesn't exist, use the entire loaded data
                        combined[section] = loaded_data
                else:
                    # If it's not a dict, use as is
                    combined[section] = loaded_data
                    
                logger.info(f"Successfully loaded section: {section}")
        except Exception as e:
            logger.error(f"Error loading section {section}: {str(e)}")
            missing.append(section)
    
    if missing:
        return ExtractionOutput(
            status="incomplete", 
            message=f"Missing section data: {missing}", 
            output_path=None, 
            extracted_data=None
        )
    
    # Add security_id to the top level for reference
    combined["security_id"] = security_id
    
    # Save the structured data
    final_path = extraction_folder / "complete_data.json"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    
    return ExtractionOutput(
        status="success", 
        message=f"Data extraction complete with {len(combined)} fields", 
        output_path=str(final_path), 
        extracted_data=combined
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
        return ExtractionOutput(status="error", message=f"Section '{section_name}' not found in schema", output_path=None, extracted_data=None)
    section_desc = section_meta.get("description", "")
    section_properties = section_meta.get("properties", {})


    # Deterministic search for financial tables
    security_folder = get_security_folder(security_id)
    parsed_images_dir = security_folder / "parsed_images"
    docs_json = []
    used_files = []
    # Refined deterministic selection: prioritize comprehensive, structured financial statements
    max_files = 20  # Fewer, more relevant docs
    table_keywords = [
        # English - Core Financial Statements
        "Balance Sheet", "Statement of Financial Position", "Assets", "Liabilities", "Equity",
        "Income Statement", "Profit and Loss", "Statement of Operations", "Revenue", "Expenses",
        "Cash Flow", "Statement of Cash Flows", "Operating Activities", "Investing Activities", "Financing Activities",
        "Statement of Changes in Equity", "Statement of Shareholders' Equity",
        
        # Financial Statement Elements
        "Current Assets", "Non-Current Assets", "Property, Plant and Equipment", "Intangible Assets", "Goodwill",
        "Current Liabilities", "Non-Current Liabilities", "Long-term Debt", "Provisions",
        "Revenue", "Cost of Sales", "Gross Profit", "Operating Income", "EBITDA", "EBIT", "Net Income",
        "Depreciation", "Amortization", "Impairment", "Provisions",
        "Changes in Working Capital", "Receivables", "Payables", "Inventory", "Prepayments",
        "Non-Cash Items", "Stock-based Compensation", "Deferred Tax", "Foreign Exchange",
        
        # Financial Metrics and KPIs
        "Financial Highlights", "Key Performance Indicators", "Financial Ratios", "Margin Analysis",
        "Liquidity Ratios", "Solvency Ratios", "Profitability Ratios", "Efficiency Ratios",
        
        # Report Context
        "Consolidated", "Consolidation", "Segment Reporting", "Notes to Financial Statements",
        "Accounting Policies", "Critical Accounting Estimates", "Annual Report", "Interim Report",
        "Quarterly Report", "Semi-Annual Report", "Auditor's Report", "Management Discussion and Analysis",
        
        # Mongolian Translations - Core Financial Statements
        "Баланс", "Хөрөнгийн тайлан", "Хөрөнгө", "Өр төлбөр", "Өмчлөлийн эрх",
        "Орлогын тайлан", "Ашиг, алдагдлын тайлан", "Үйл ажиллагааны тайлан", "Орлого", "Зардал",
        "Мөнгөн урсгал", "Мөнгөн гүйлгээний тайлан", "Үйл ажиллагааны үйлдэл", "Хөрөнгө оруулалтын үйлдэл", "Санхүүжилтийн үйлдэл",
        "Өмчлөлийн өөрчлөлтийн тайлан", "Хувьцаа эзэмшигчдийн эрхийн тайлан",
        
        # Mongolian - Financial Elements
        "Эргэлтийн хөрөнгө", "Урт хугацааны хөрөнгө", "Үндсэн хөрөнгө", "Биет бус хөрөнгө", "Сайн дурын үнэлгээ",
        "Богино хугацаат өр төлбөр", "Урт хугацаат өр төлбөр", "Урт хугацааны зээл", "Нөөц",
        "Нийт орлого", "Борлуулалтын өртөг", "Нийт ашиг", "Үйл ажиллагааны ашиг", "EBITDA", "EBIT", "Цэвэр ашиг",
        "Элэгдэл", "Элэгдэл хорогдол", "Үнэ цэнийн бууралт", "Нөөц",
        "Ашиглалтын өөрчлөлт", "Авлага", "Өглөг", "Бараа материал", "Урьдчилсан төлбөр",
        "Бэлэн бус зүйлс", "Хувьцааны урамшуулал", "Хойшлогдсон татвар", "Валютын ханшны өөрчлөлт",
        
        # Mongolian - Report Context
        "Нэгдсэн", "Нэгтгэл", "Сегментийн тайлан", "Санхүүгийн тайлангийн тэмдэглэл",
        "Нягтлан бодох бүртгэлийн бодлого", "Шалгуур үзүүлэлтүүд", "Жилийн тайлан", "Улирлын тайлан",
        "Хагас жилийн тайлан", "Үнэлгээний тайлан", "Удирдлагын тайлан ба дүн шинжилгээ"
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

    llm = get_gemini_llm()
    # Logging
    debug_folder = get_data_extraction_folder(security_id)
    debug_log_path = debug_folder / "bond_financials_historical_extraction_debug.log"
    with open(debug_log_path, "w", encoding="utf-8") as dbg:
        dbg.write(f"\n=== Section: {section_name} ===\n")
        dbg.write(f"docs_json_count: {len(docs_json)}\n")
        dbg.write(f"files_sent ({len(used_files)}):\n")
        for i, fpath in enumerate(used_files):
            dbg.write(f"  [{i+1}] {fpath}\n")
    
    result = await extract_section(llm, section_name, section_desc, section_properties, docs_json, prompt_messages=build_section_prompt(section_name, section_desc, section_properties, docs_json, security_id=security_id, repair_prompt=input_data.repair_prompt))
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
    result = await extract_section(llm, section_name, section_meta.get("description", ""), section_properties, docs_json, prompt_messages=collateral_prompt)
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
    issuer_prompt = build_issuer_profile_prompt(section_name, section_meta.get('description', ''), section_properties, docs_json, repair_prompt=input_data.repair_prompt)
    log_section_debug(section_name, indices, docs_json, idx_to_file, issuer_prompt, security_id)
    result = await extract_section(llm, section_name, section_meta.get("description", ""), section_properties, docs_json, prompt_messages=issuer_prompt)
    logger.info(f"[Extraction] LLM result for section '{section_name}': {str(result)[:200]}...")
    output_path = save_section_output_json(security_id, section_name, result)
    return ExtractionOutput(status="success", message="Extraction complete", output_path=output_path, extracted_data={section_name: result})

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

async def extract_section(llm, section_name: str, section_desc: str, section_properties: dict, docs_json: list, prompt_messages=None) -> Any:
    import time
    prompt = prompt_messages if prompt_messages is not None else build_section_prompt(section_name, section_desc, section_properties, docs_json)

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


def build_historical_debt_schedule_prompt(docs_json: List[Dict[str, Any]], item_schema_properties: Dict[str, Any], security_id: Optional[str] = None, repair_prompt: Optional[str] = None):
    bond_financials_path = None
    issuer_profile_path = None

    if security_id:
        security_folder = get_security_folder(security_id) # Returns Path object or None
        if security_folder:
            bond_financials_path = str(security_folder / "data_extraction" / "bond_financials_historical.json")
            issuer_profile_path = str(security_folder / "data_extraction" / "issuer_business_profile.json")
        else:
            logger.warning(f"[build_historical_debt_schedule_prompt] Could not find security folder for {security_id}. Security-specific context files will be unavailable.")

    contextual_file_paths = {
        # "bond_financials_historical": bond_financials_path,
        "issuer_business_profile": issuer_profile_path,
        "country_research": "/home/me/CascadeProjects/Alpy/subagents/financial_analyst/external_reports/country_research.json",
        "industry_research": "/home/me/CascadeProjects/Alpy/subagents/financial_analyst/external_reports/industry_research.json"
    }

    contextual_data_str = "\n\n--- Additional Contextual Information ---"
    for name, path in contextual_file_paths.items():
        if path and os.path.exists(path):
            try:
                content = load_txt_file(path) # Assumes load_txt_file can handle JSON and returns dict/list
                if content:
                    # Truncate to avoid excessively long prompts
                    content_str = json.dumps(content, ensure_ascii=False, indent=None) # Compact JSON string
                    max_len = 1000 # Max characters per context file to keep prompt manageable
                    if len(content_str) > max_len:
                        content_str = content_str[:max_len] + "... (truncated)"
                    contextual_data_str += f"\n\n## Context from {name} ({os.path.basename(path)}):\n{content_str}"
                else:
                    contextual_data_str += f"\n\n## Context from {name} ({os.path.basename(path)}):\nFile empty or unreadable."
            except Exception as e:
                logger.warning(f"[build_historical_debt_schedule_prompt] Error loading contextual file {path}: {e}")
                contextual_data_str += f"\n\n## Context from {name} ({os.path.basename(path)}):\nError loading file."
        elif path:
            contextual_data_str += f"\n\n## Context from {name} ({os.path.basename(path)}):\nFile not found at {path}."
        else: # Path was None (e.g. security_id not provided for some files)
             contextual_data_str += f"\n\n## Context from {name}:\nPath not applicable for this request."
    contextual_data_str += "\n--- End of Additional Contextual Information ---"

    """
    Build an LLM prompt for extracting historical debt schedule information.
    Instructs the LLM to find all historical debt instruments and their details across multiple periods.
    """
    system_message = (
        "You are an expert financial analyst. Your task is to extract detailed historical debt schedule information. "
        "You will be given primary 'Document Excerpts' and 'Additional Contextual Information'. "
        "Prioritize information from 'Document Excerpts' but use 'Additional Contextual Information' to infer missing details and resolve ambiguities.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. SCHEMA COMPLIANCE: You MUST follow the schema exactly. Only include fields defined in the schema.\n"
        "2. REQUIRED FIELDS: The following fields are REQUIRED and must be present in every item:\n"
        "   - instrument_name: Name/type of the debt instrument\n"
        "   - principal_amount_issued: Original principal amount in base units (e.g., 1000000, not 1M)\n"
        "   - maturity_duration: Original duration from issuance to maturity (e.g., '5 years')\n"
        "   - interest_rate_pa: Annual interest rate (e.g., '7.5%' or 'LIBOR + 2.5%')\n"
        "3. DATE FORMAT: Use YYYY-MM-DD format for all dates.\n"
        "4. MONETARY VALUES: Always use base units (e.g., 1000000, not 1M).\n"
        "5. NO NULLS: Do not use null for required fields. Make reasonable estimates when needed.\n"
        "6. NOTES: Always include detailed notes explaining any assumptions or estimations.\n\n"
        "GUIDELINES:\n"
        "1. For each distinct debt instrument found in the 'Document Excerpts', extract its terms and outstanding amounts for all available historical periods. "
        "Structure your output as a JSON list, where each item in the list corresponds to a single debt instrument's status in a specific historical period, strictly following the provided schema.\n"
        "2. DATE INTERPRETATION: For ambiguous dates, assume they refer to the most recent plausible year. Cross-reference with the issuer's operational history when available.\n"
        "3. ESTIMATION: If a value is not explicitly stated, make a reasonable estimate based on the context and explain in the 'notes' field.\n"
        "4. MONETARY VALUES: Convert all values to base units (e.g., 1,234,567, not 1.23 million).\n"
        "5. SINGLE VALUES: Each field must contain only a single value. For multiple values, choose the most representative one and explain your choice in 'notes'.\n"
        "6. ACCURACY: Cross-reference information within the excerpts and with the contextual data.\n"
    )

    if security_id:
        principal_amount, currency = get_bond_principal_amount(security_id)
        if principal_amount and currency:
            unit_instructions = (
                f"IMPORTANT UNIT NORMALIZATION INSTRUCTIONS:\n"
                f"1. The primary bond for this issuer has a principal amount of {principal_amount} {currency}. Use this as a reference for the expected magnitude of other financial figures.\n"
                f"2. All extracted financial values (principal_amount_outstanding, interest_expense_for_period, etc.) MUST be in RAW BASE UNITS of the original currency of the debt instrument. For example, if a loan is 50 million USD, extract it as 50000000. Do not use abbreviations like 'MM' or 'K'."
                f"3. If a value is stated as 'USD 125.5 million', your output for that field should be 125500000."
                f"4. Pay close attention to the `currency` field for each debt item."
            )
            system_message += f"\n\n{unit_instructions}"

    user_message_content = (
        f"Please extract historical debt schedule information for security ID {security_id if security_id else 'N/A'}. "
        f"The required JSON schema for each item is: {json.dumps(item_schema_properties)}\n\n"
        f"Document Excerpts:\n{json.dumps(docs_json, indent=2, ensure_ascii=False)}\n"
        f"{contextual_data_str}"
    )
    # Let's rebuild user_message_content more cleanly.

    # Create a clear example of the expected output format
    example_output = [
        {
            "base_currency": "MNT",
            "instrument_name": "7.5% Senior Notes due 2028",
            "currency": "MNT",
            "principal_amount_issued": 1000000000.0,
            "issuance_date": "2023-01-15",
            "maturity_date": "2028-01-15",
            "maturity_duration": "5 years",
            "interest_rate_pa": "7.5%",
            "principal_repayments_made_during_period": 250000000.0,
            "source_document_reference": "2023 Annual Report, p. 55",
            "notes": "Interest rate is fixed. Principal is due at maturity."
        }
    ]

    # List of required fields that must be present in every item
    required_fields = [
        "instrument_name",
        "principal_amount_issued",
        "maturity_duration",
        "interest_rate_pa"
    ]

    user_message_parts = [
        "EXTRACT HISTORICAL DEBT SCHEDULE INFORMATION",
        "=========================================\n\n",
        "REQUIRED FIELDS (MUST BE PRESENT IN EVERY ITEM):",
        "- " + "\n- ".join(required_fields) + "\n\n",
        "SCHEMA PROPERTIES:",
        json.dumps(item_schema_properties, indent=2) + "\n\n",
        "EXAMPLE OUTPUT (single item shown, return a list):",
        json.dumps(example_output, indent=2) + "\n\n",
        "DOCUMENT EXCERPTS FOR ANALYSIS:",
        "================================"
    ]

    for i, doc_content in enumerate(docs_json, 1):
        user_message_parts.append(f"\n--- DOCUMENT EXCERPT {i} ---")
        user_message_parts.append(json.dumps(doc_content, ensure_ascii=False))
    
    # Add strict instructions
    user_message_parts.extend([
        "\nINSTRUCTIONS:",
        "1. Extract ALL historical debt instruments found in the documents.",
        "2. For each instrument, include ALL required fields listed above.",
        "3. DO NOT include any fields that are not in the schema.",
        "4. If a required field is not explicitly stated, make a reasonable estimate based on context.",
        "5. For dates, use YYYY-MM-DD format. For monetary values, use base units (e.g., 1000000 instead of 1M).",
        "6. If no data is found for an optional field, use null.",
        "7. Include a descriptive 'notes' field explaining any assumptions or estimations made.",
        "\nReturn a JSON list of debt schedule items following the schema exactly."
    ])
    
    user_message_content = "\n".join(user_message_parts)

    prompt_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message_content}
    ]

    if repair_prompt:
        prompt_messages.append({"role": "user", "content": f"Previous attempt had issues. Please consider this feedback: {repair_prompt}"})

    logger.info(f"[build_historical_debt_schedule_prompt] Generated prompt for security_id={security_id}")
    return prompt_messages

@mcp_app.tool(name="historical_debt_schedule")
async def historical_debt_schedule(input_data: ExtractionInput) -> ExtractionOutput:
    logger.info(f"[historical_debt_schedule] Extraction started for security_id={input_data.security_id}")
    security_id = input_data.security_id
    vector_ops_tool = VectorOpsTool()

    # Get the schema from data_schema.json
    schema = get_schema(input_data.schema_path)
    if not schema or "historical_debt_schedule" not in schema.get("properties", {}):
        logger.error(f"[historical_debt_schedule] Schema validation failed: 'historical_debt_schedule' not found in schema")
        return ExtractionOutput(status="error", message="Schema validation failed: 'historical_debt_schedule' not found in schema", output_path=None, extracted_data=None)
    
    # Get the item schema properties for the prompt
    item_schema_properties = schema["properties"]["historical_debt_schedule"]
    # item_schema_properties = item_schema.get("items", {})
    # logger.info(f"[historical_debt_schedule] Item schema properties: {item_schema_properties}")

    try:
        # Load vector metadata to map indices to file paths before running queries
        vector_metadata = load_vector_metadata(security_id)
        if not vector_metadata:
            logger.error(f"[historical_debt_schedule] Failed to load vector_metadata.json for security_id: {security_id}")
            return ExtractionOutput(status="error", message="Failed to load vector metadata.", output_path=None, extracted_data=None)
        
        idx_to_file = {item["index"]: item["file"] for item in vector_metadata if isinstance(item, dict) and "index" in item and "file" in item}
        if not idx_to_file:
            logger.error(f"[historical_debt_schedule] Vector metadata for {security_id} is empty or malformed (no index/file entries).")
            return ExtractionOutput(status="error", message="Vector metadata is empty or malformed.", output_path=None, extracted_data=None)

        # 1. Retrieve relevant document chunks using vector search
        # Using multiple specific queries to capture all debt instruments
        queries = [
            # Specific debt instruments
            "bond issuance schedule with maturity dates and interest rates",
            "loan agreements and credit facilities terms and conditions",
            "long-term debt schedule with principal and interest payments",
            "debt maturity profile and repayment schedule",
            "outstanding notes payable and bonds with terms",
            
            # Financial statement sections
            "notes to financial statements debt disclosures",
            "long-term liabilities section of balance sheet",
            "debt covenants and restrictions",
            
            # Non-current liabilities
            "non-current borrowings breakdown",
            "long-term loans and borrowings schedule",
            
            # Mongolian language queries - Enhanced
            "зээлийн гэрээ, бондын нөхцөл, хүүний төлбөрийн хуваарь, эргэн төлөлтийн төлөвлөгөө",  # Loan agreements, bond terms, interest payment schedule, repayment plan
            "урт болон богино хугацааны өрийн жагсаалт, эргэн төлөлтийн хуваарь",  # Long and short term debt list, repayment schedule
            "бондын эмиссийн мэдээлэл, хугацаа, хүү, нэрлэсэн үнэ",  # Bond issuance information, term, interest, face value
            "зээл, өрийн бичиг, бондын жагсаалт, эргэн төлөгдөх хугацаа",  # Loans, debt securities, bond list, maturity
            "санхүүгийн тайлан дахь өрийн мэдээлэл, тэтгэмж, нөхцөл",  # Debt information in financial statements, covenants, terms
            "зээлийн гэрээний нөхцөл, баталгаа, хариуцлага",  # Loan agreement terms, guarantees, liabilities
            "бондын эргэн төлөлт, хүү төлбөрийн хуваарь, валют"  # Bond repayment, interest payment schedule, currency
        ]
        
        # Combine all document chunks from multiple queries
        all_docs = []
        seen_indices = set()
        
        for query in queries:
            try:
                vector_search_result_str = await vector_ops_tool._arun(
                    security_id=security_id,
                    vector_store_type="text",
                    action="search_similar",
                    query=query,
                    k=12,  # Increased from 8 to 12 to capture more documents
                    min_similarity_score = 0.65,
                )
                
                if vector_search_result_str and isinstance(vector_search_result_str, str):
                    vector_search_result = json.loads(vector_search_result_str)
                    indices = vector_search_result.get("indices") or \
                             (vector_search_result.get("result", {}).get("indices") 
                              if isinstance(vector_search_result.get("result"), dict) else None)
                    
                    if indices and isinstance(indices, list):
                        for idx in indices:
                            if idx not in seen_indices:
                                seen_indices.add(idx)
                                file_path_str = idx_to_file.get(idx)
                                if file_path_str:
                                    doc_content = load_txt_file(file_path_str)
                                    if doc_content and isinstance(doc_content, dict):
                                        all_docs.append(doc_content)
                                    
            except Exception as e:
                logger.warning(f"[historical_debt_schedule] Error in vector search for query '{query}': {e}")
                continue
        
        docs_json = all_docs

        # Call VectorOpsTool and expect a JSON string as per collateral_and_protective_clauses
        vector_search_result_str = await vector_ops_tool._arun(
            security_id=security_id,
            vector_store_type="text",
            action="search_similar",
            query=query,
            k=8  # Retrieve top 8 relevant chunks
        )

        if not vector_search_result_str or not isinstance(vector_search_result_str, str):
            logger.error(f"[historical_debt_schedule] VectorOpsTool._arun did not return a JSON string. Got: {type(vector_search_result_str)}")
            return ExtractionOutput(status="error", message="Vector search returned unexpected data type.", output_path=None, extracted_data=None)

        try:
            vector_search_result = json.loads(vector_search_result_str)
        except json.JSONDecodeError as e:
            logger.error(f"[historical_debt_schedule] Failed to parse JSON from VectorOpsTool: {e}. String was: {vector_search_result_str[:500]}")
            return ExtractionOutput(status="error", message="Failed to parse vector search result.", output_path=None, extracted_data=None)
        
        # Extract indices based on the pattern in collateral_and_protective_clauses
        indices = vector_search_result.get("indices") or \
                  (vector_search_result.get("result", {}).get("indices") if isinstance(vector_search_result.get("result"), dict) else None)

        if not indices or not isinstance(indices, list):
            logger.error(f"[historical_debt_schedule] No 'indices' found in vector search result or not a list. Result: {json.dumps(vector_search_result, indent=2)}")
            return ExtractionOutput(status="error", message="No indices found in vector search result.", output_path=None, extracted_data=None)

        # Load document content using indices and the idx_to_file mapping
        docs_json = []
        for idx in indices:
            file_path_str = idx_to_file.get(idx)
            if file_path_str:
                # Ensure file_path_str is an absolute path or resolvable relative to a known base
                # Assuming file paths in vector_metadata.json are absolute or relative to security_folder
                # If relative, they might need to be joined with get_security_folder(security_id)
                # For now, assume load_txt_file can handle them or they are absolute.
                doc_content = load_txt_file(file_path_str) # load_txt_file expects str path
                if doc_content:
                    # Ensure doc_content is a dictionary as expected by build_historical_debt_schedule_prompt
                    if isinstance(doc_content, dict):
                        docs_json.append(doc_content)
                    else:
                        # If load_txt_file returns string content (e.g. for .txt files not .json chunks)
                        # we might need to wrap it or handle it differently. For now, log and skip.
                        logger.warning(f"[historical_debt_schedule] Loaded content for index {idx} from {file_path_str} is not a dict, skipping. Type: {type(doc_content)}")
                else:
                    logger.warning(f"[historical_debt_schedule] Failed to load content for index {idx} from path: {file_path_str}")
            else:
                logger.warning(f"[historical_debt_schedule] No file path found in idx_to_file for index: {idx}")
        
        if not docs_json:
            logger.warning(f"[historical_debt_schedule] No document content successfully loaded for the retrieved indices. Query: {query}")
            return ExtractionOutput(status="warning", message="No relevant document content could be loaded.", output_path=None, extracted_data=None)

        logger.info(f"[historical_debt_schedule] Retrieved and loaded content for {len(docs_json)} document chunks for {security_id}.")

    except Exception as e:
        logger.error(f"[historical_debt_schedule] Error during vector search for {security_id}: {e}")
        return ExtractionOutput(status="error", message=f"Vector search failed: {e}", output_path=None, extracted_data=None)

    # 2. Build the prompt with explicit instructions to find ALL debt instruments
    enhanced_repair_prompt = (input_data.repair_prompt or "") + "\n\nIMPORTANT: You MUST find and return ALL debt instruments mentioned in the documents, not just the most prominent one. Include every debt instrument you can identify, even if information is incomplete. Missing data points should be set to null. Pay special attention to different debt instruments with different names, interest rates, or maturity dates."
    
    prompt_messages = build_historical_debt_schedule_prompt(
        docs_json=docs_json, 
        item_schema_properties=item_schema_properties,
        security_id=security_id, 
        repair_prompt=enhanced_repair_prompt
    )

    # 3. Call LLM to extract data
    llm = get_gemini_llm()
    try:
        logger.info(f"[historical_debt_schedule] Sending request to LLM for {security_id}.")
        response = await llm.ainvoke(prompt_messages)
        extracted_data_raw = extract_json_from_response(response.content)
        logger.info(f"[historical_debt_schedule] Received LLM response for {security_id}.")

        # Handle both list and single dictionary responses
        if isinstance(extracted_data_raw, dict):
            # If it's a single debt item, wrap it in a list
            extracted_data_raw = [extracted_data_raw]
            logger.info("[historical_debt_schedule] Wrapped single debt item in a list")
        elif not isinstance(extracted_data_raw, list):
            error_msg = f"LLM did not return a valid response. Expected list or dict, got {type(extracted_data_raw)}"
            logger.warning(f"[historical_debt_schedule] {error_msg}")
            return ExtractionOutput(status="error", message=error_msg, output_path=None, extracted_data=extracted_data_raw)

        # Basic validation of each item in the list
        validated_debt_items = []
        for item_data in extracted_data_raw:
            try:
                # Basic type checking for required fields
                if not isinstance(item_data, dict):
                    logger.warning(f"[historical_debt_schedule] Item is not a dictionary: {item_data}")
                    continue
                    
                # Type conversion and validation for numeric fields
                numeric_fields = [
                    "principal_amount_outstanding",
                    "principal_repayments_made_during_period"
                ]
                
                valid_item = {}
                for field, value in item_data.items():
                    if field in numeric_fields and value is not None:
                        try:
                            valid_item[field] = float(value)
                        except (ValueError, TypeError):
                            logger.warning(f"[historical_debt_schedule] Invalid numeric value for {field}: {value}")
                            valid_item[field] = None
                    else:
                        valid_item[field] = value
                        
                validated_debt_items.append(valid_item)
                
            except Exception as e:
                logger.warning(f"[historical_debt_schedule] Error validating debt item: {e}. Item: {item_data}")
                # Skip invalid items

        if not validated_debt_items:
            logger.warning(f"[historical_debt_schedule] No valid debt items extracted after validation for {security_id}.")
            return ExtractionOutput(status="no_data", message="No valid historical debt items extracted.", output_path=None, extracted_data=None)

        extracted_data = {"historical_debt_schedule": validated_debt_items}

    except Exception as e:
        logger.error(f"[historical_debt_schedule] LLM call or data processing failed for {security_id}: {e}")
        return ExtractionOutput(status="error", message=f"LLM call or processing failed: {e}", output_path=None, extracted_data=None)

    # 4. Save the extracted data
    output_dir = get_data_extraction_folder(security_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "historical_debt_schedule.json"
    
    # Ensure extracted_data is properly structured
    if not isinstance(extracted_data, dict):
        extracted_data = {"historical_debt_schedule": []}
    elif not isinstance(extracted_data.get("historical_debt_schedule"), list):
        extracted_data["historical_debt_schedule"] = [extracted_data.get("historical_debt_schedule") or {}]
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, indent=2, ensure_ascii=False)
        logger.info(f"[historical_debt_schedule] Successfully saved {len(extracted_data.get('historical_debt_schedule', []))} items to {output_path}")
        return ExtractionOutput(
            status="success", 
            message=f"Successfully extracted {len(extracted_data.get('historical_debt_schedule', []))} historical debt items.",
            output_path=str(output_path), 
            extracted_data=extracted_data
        )
    except Exception as e:
        logger.error(f"[historical_debt_schedule] Failed to save data for {security_id} to {output_path}: {e}")
        return ExtractionOutput(
            status="error", 
            message=f"Failed to save data: {e}", 
            output_path=None, 
            extracted_data=extracted_data
        )

async def run_stdio_server():
    await mcp_app.run_stdio_async()

def main():
    logger.info("Starting MCP Data Extractor server main loop")
    import asyncio
    asyncio.run(run_stdio_server())

if __name__ == "__main__":
    main()
