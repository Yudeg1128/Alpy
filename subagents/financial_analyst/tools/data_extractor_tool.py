"""
LangChain Tool for Bond Data Extraction via MCP Server
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, model_validator, PrivateAttr
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import Implementation as MCPImplementation

logger = logging.getLogger(__name__)

class DataExtractorInput(BaseModel):
    security_id: str = Field(..., description="Security identifier")
    action: str = Field(..., description="REQUIRED: Name of the extraction action/section to run (e.g., 'bond_metadata', 'bond_financials_historical', etc. See below). Must match a valid MCP tool name.")
    repair_prompt: Optional[str] = Field(None, description="OPTIONAL: If set, provides explicit repair instructions or feedback for the LLM to fix extraction errors. Used only for quality control or repair retries.")


class DataExtractorTool(BaseTool, BaseModel):
    name: str = "data_extractor"
    description: str = (
        "Extract structured bond data for a security using the MCP data extractor server.\n"
        "\n"
        "INSTRUCTIONS: For ALL tool calls, you MUST supply both 'security_id' and 'action'.\n"
        "- 'action' is REQUIRED and must be set to the exact tool/section name you wish to extract (e.g., 'bond_metadata', 'bond_financials_historical', etc.)\n"
        "- 'repair_prompt' is OPTIONAL. If provided, it gives explicit feedback or instructions for the LLM to repair or improve extraction results for a section (e.g., after quality control).\n"
        "\n"
        "Available Actions (set section=None unless noted):\n"

        "1. bond_metadata: Extract bond_metadata section.\n"
        "2. bond_financials_historical: Extract historical financials.\n"
        "3. bond_financials_projections: Extract projections.\n"
        "4. collateral_and_protective_clauses: Extract collateral and protective clauses.\n"
        "5. issuer_business_profile: Extract issuer business profile.\n"
        "6. industry_profile: Extract industry profile.\n"
        "7. finalize_data_extraction: Assemble the final complete_data.json from all section JSONs.\n"
        "\n"
        "EXACT TOOL CALL FORMATS (JSON):\n"
        "- To extract only bond_metadata:\n"
        "  Action:\n"
        "  ```json\n"
        "  {\n    \"action\": \"data_extractor\",\n    \"action_input\": {\n      \"security_id\": \"SECURITY_ID\",\n      \"action\": \"bond_metadata\"\n    }\n  }\n"
        "  ```\n"

        "- To repair bond_metadata with a repair prompt:\n"
        "  Action:\n"
        "  ```json\n"
        "  {\n    \"action\": \"data_extractor\",\n    \"action_input\": {\n      \"security_id\": \"SECURITY_ID\",\n      \"action\": \"bond_metadata\",\n      \"repair_prompt\": \"IMPROVE trustee_agent data\"\n    }\n  }\n"
        "  ```\n"
        "- To finalize and combine all extracted section data:\n"
        "  Action:\n"
        "  ```json\n"
        "  {\n    \"action\": \"data_extractor\",\n    \"action_input\": {\n      \"security_id\": \"SECURITY_ID\",\n      \"action\": \"finalize_data_extraction\"\n    }\n  }\n"
        "  ```\n"
        "\n"
        "Never supply any schema_path argument. Always follow the above JSON format for tool calls."
    )
    args_schema: Type[BaseModel] = DataExtractorInput
    return_direct: bool = False
    _session: Optional[ClientSession] = PrivateAttr(default=None)
    _logger_instance: logging.Logger = PrivateAttr(default=None)
    _init_lock: Optional[asyncio.Lock] = PrivateAttr(default=None)
    _is_closed: bool = PrivateAttr(default=False)
    _client_info: MCPImplementation = PrivateAttr(default_factory=lambda: MCPImplementation(name="DataExtractorToolClient", version="0.1.0"))
    _server_script: Path = PrivateAttr(default=Path(__file__).parent.parent / "mcp_servers" / "mcp_data_extractor" / "server.py")
    _server_cwd_path: Path = PrivateAttr(default=Path(__file__).parent.parent / "mcp_servers" / "mcp_data_extractor")

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def _tool_post_init(self) -> 'DataExtractorTool':
        if self._logger_instance is None:
            self._logger_instance = logger.getChild(f"{self.name}.{id(self)}")
        if not self._init_lock:
            self._init_lock = asyncio.Lock()
        if not self._server_cwd_path:
            self._server_cwd_path = self._server_script.parent
        return self

    async def _initialize_async_primitives(self):
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()
        if self._session is not None:
            return

    def _get_server_params(self):
        return StdioServerParameters(
            command=sys.executable,
            args=[str(self._server_script)],
            cwd=str(self._server_cwd_path)
        )

    async def _arun(self, security_id: str, action: str, repair_prompt: Optional[str] = None) -> str:
        await self._initialize_async_primitives()
        params = self._get_server_params()
        async with stdio_client(params, errlog=sys.stderr) as (rs, ws):
            async with ClientSession(rs, ws, client_info=self._client_info) as session:
                await session.initialize()
                args = {
                    "input_data": {
                        "security_id": security_id,
                        "action": action
                    }
                }
                if repair_prompt is not None:
                    args["input_data"]["repair_prompt"] = repair_prompt
                # Debug log the tool call request
                if hasattr(self, '_logger_instance') and self._logger_instance:
                    self._logger_instance.info(f"Calling MCP tool '{action}' with arguments: {json.dumps(args)}")
                print(f"[DataExtractorTool] Calling MCP tool '{action}' with arguments: {json.dumps(args)}")
                tool_name = action
                tool_res = await session.call_tool(name=tool_name, arguments=args)
                if getattr(tool_res, "isError", False):
                    msg = getattr(tool_res, "message", None) or str(tool_res)
                    raise RuntimeError(f"Server error: {msg}")
                content = getattr(tool_res, "content", None)
                if not content:
                    raise RuntimeError("No content in server response")
                if hasattr(content, "dict"):
                    return json.dumps(content.dict())
                elif isinstance(content, dict):
                    return json.dumps(content)
                else:
                    return str(content)

    # Example usage for finalize_data_extraction:
    # await self._arun(security_id, section=None, action="finalize_data_extraction")

    async def close(self):
        if self._session:
            try:
                await self._session.close()
            except Exception as e:
                self._logger_instance.warning(f"Error closing MCP session: {e}")
            self._session = None
        self._is_closed = True

    def _run(self, *args, **kwargs):
        raise NotImplementedError("This tool only supports async execution.")

# --- Async Test Functions (matching txt_embedder_tool.py style) ---
import logging
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mcp_servers')))

async def test_bond_metadata(security_id=None):
    from mcp_data_extractor.server import ExtractionInput, bond_metadata
    logger = logging.getLogger("DataExtractorToolTest.bond_metadata")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info("Starting bond_metadata extraction test...")
    input_data = ExtractionInput(security_id=security_id)
    result = await bond_metadata(input_data)
    print("\n=== bond_metadata Extraction Result ===")
    print(result)
    logger.info("Test finished.")

async def test_bond_financials_historical(security_id=None, repair_prompt=None):
    from mcp_data_extractor.server import ExtractionInput, bond_financials_historical
    logger = logging.getLogger("DataExtractorToolTest.bond_financials_historical")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info("Starting bond_financials_historical extraction test...")
    input_data = ExtractionInput(security_id=security_id, repair_prompt=repair_prompt)
    if repair_prompt:
        logger.info(f"Using repair prompt: {repair_prompt}")
    result = await bond_financials_historical(input_data)
    print("\n=== bond_financials_historical Extraction Result ===")
    print(result)
    logger.info("Test finished.")

async def test_collateral_and_protective_clauses(security_id=None):
    print("Starting collateral_and_protective_clauses extraction test...")
    from mcp_servers.mcp_data_extractor.server import collateral_and_protective_clauses, ExtractionInput
    input_data = ExtractionInput(security_id=security_id)
    result = await collateral_and_protective_clauses(input_data)
    print(f"\n=== collateral_and_protective_clauses Extraction Result ===\n{result}")
    print("Test finished.")

async def test_issuer_business_profile(security_id=None):
    print("Starting issuer_business_profile extraction test...")
    from mcp_servers.mcp_data_extractor.server import issuer_business_profile, ExtractionInput
    input_data = ExtractionInput(security_id=security_id)
    result = await issuer_business_profile(input_data)
    print(f"\n=== issuer_business_profile Extraction Result ===\n{result}")
    print("Test finished.")

async def test_industry_profile(security_id=None):
    print("Starting industry_profile extraction test...")
    from mcp_servers.mcp_data_extractor.server import industry_profile, ExtractionInput
    input_data = ExtractionInput(security_id=security_id)
    result = await industry_profile(input_data)
    print(f"\n=== industry_profile Extraction Result ===\n{result}")
    print("Test finished.")

async def test_bond_financials_projections(security_id=None):
    print("Starting bond_financials_projections extraction test...")
    from mcp_servers.mcp_data_extractor.server import bond_financials_projections, ExtractionInput
    input_data = ExtractionInput(security_id=security_id)
    result = await bond_financials_projections(input_data)
    print(f"\n=== bond_financials_projections Extraction Result ===\n{result}")
    print("Test finished.")

async def test_finalize_data_extraction(security_id=None):
    print("Starting finalize_data_extraction test...")
    from mcp_servers.mcp_data_extractor.server import ExtractionInput, finalize_data_extraction
    input_data = ExtractionInput(security_id=security_id)
    result = await finalize_data_extraction(input_data)
    print(f"\n=== finalize_data_extraction Result ===\n{result}")
    print("Test finished.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run DataExtractorTool tests")
    parser.add_argument("--security_id", type=str, help="Security ID for testing")
    parser.add_argument("--test", type=str, choices=["bond_metadata", "bond_financials_historical", "bond_financials_projections", "collateral_and_protective_clauses", "issuer_business_profile", "industry_profile", "finalize_data_extraction"], help="Test to run")
    parser.add_argument("--repair_prompt", type=str, help="Repair prompt for testing")
    args = parser.parse_args()
    if args.test == "bond_metadata":
        asyncio.run(test_bond_metadata(security_id=args.security_id))
    elif args.test == "bond_financials_historical":
        asyncio.run(test_bond_financials_historical(security_id=args.security_id, repair_prompt=args.repair_prompt))
    elif args.test == "bond_financials_projections":
        asyncio.run(test_bond_financials_projections(security_id=args.security_id))
    elif args.test == "collateral_and_protective_clauses":
        asyncio.run(test_collateral_and_protective_clauses(security_id=args.security_id))
    elif args.test == "issuer_business_profile":
        asyncio.run(test_issuer_business_profile(security_id=args.security_id))
    elif args.test == "industry_profile":
        asyncio.run(test_industry_profile(security_id=args.security_id))
    elif args.test == "finalize_data_extraction":
        asyncio.run(test_finalize_data_extraction(security_id=args.security_id))
    else:
        parser.print_help()
