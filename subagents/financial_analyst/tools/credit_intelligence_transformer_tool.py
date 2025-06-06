"""
LangChain Tool for Credit Intelligence Transformation via MCP Server
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

class CreditIntelligenceInput(BaseModel):
    action: str = Field(..., description="Which phase action to run: 'generate_assumptions', 'run_financial_model', or 'summarize_credit_risk'.")
    security_id: str = Field(..., description="Security ID to analyze. The tool will automatically load the full extracted data for this security.")

class CreditIntelligenceTransformerTool(BaseTool, BaseModel):
    _session: ClientSession = PrivateAttr(default=None)
    _is_closed: bool = PrivateAttr(default=False)
    name: str = "credit_intelligence_transformer"
    description: str = (
        "CREDIT INTELLIGENCE TRANSFORMER TOOL\n"
        "Purpose: Run multi-phase credit analysis for a security using extracted data. "
        "Supports three actions: 'generate_assumptions', 'run_financial_model', 'summarize_credit_risk'. "
        "Automatically chains prior phases and loads all required extracted data for the given security_id.\n\n"
        "TOOL CALL FORMAT (MANDATORY):\n"
        "You MUST call this tool with the following nested JSON structure, per the ReAct prompt format:\n"
        "Action:\n"
        "```json\n"
        "{\n"
        "  \"action\": \"credit_intelligence_transformer\",\n"
        "  \"action_input\": {\n"
        "    \"security_id\": \"SECURITY_ID\",\n"
        "    \"action\": \"PHASE_ACTION\"\n"
        "  }\n"
        "}\n"
        "```\n"
        "Where PHASE_ACTION is one of: 'generate_assumptions', 'run_financial_model', or 'summarize_credit_risk'.\n\n"
        "EXACT CALL EXAMPLES:\n"
        "- To generate assumptions:\n"
        "  Action:\n"
        "  ```json\n"
        "  {\n"
        "    \"action\": \"credit_intelligence_transformer\",\n"
        "    \"action_input\": {\n"
        "      \"security_id\": \"MN0LNDB68390\",\n"
        "      \"action\": \"generate_assumptions\"\n"
        "    }\n"
        "  }\n"
        "  ```\n"
        "- To run the financial model (assumptions will be generated automatically):\n"
        "  Action:\n"
        "  ```json\n"
        "  {\n"
        "    \"action\": \"credit_intelligence_transformer\",\n"
        "    \"action_input\": {\n"
        "      \"security_id\": \"MN0LNDB68390\",\n"
        "      \"action\": \"run_financial_model\"\n"
        "    }\n"
        "  }\n"
        "  ```\n"
        "- To summarize credit risk (all prior phases are chained automatically):\n"
        "  Action:\n"
        "  ```json\n"
        "  {\n"
        "    \"action\": \"credit_intelligence_transformer\",\n"
        "    \"action_input\": {\n"
        "      \"security_id\": \"MN0LNDB68390\",\n"
        "      \"action\": \"summarize_credit_risk\"\n"
        "    }\n"
        "  }\n"
        "  ```\n\n"
        "OUTPUT FORMAT:\n"
        "Returns a dict containing the result of the requested phase.\n"
        "- For 'generate_assumptions': { ...assumptions... }\n"
        "- For 'run_financial_model': { ...financial_model_output... }\n"
        "- For 'summarize_credit_risk': { ...credit_summary... }\n\n"
        "NOTES FOR AGENT:\n"
        "- Always use the nested JSON tool call format as shown above.\n"
        "- Do NOT call phase actions directly; always use 'credit_intelligence_transformer' as the action, "
        "with 'action_input' containing both 'security_id' and 'action'.\n"
        "- For multi-phase workflows, simply call the highest phase needed; all dependencies will be handled automatically.\n"
        "- If you need intermediate output (e.g., assumptions), call that phase explicitly.\n"
        "- If you encounter errors, review the security_id and ensure extracted data exists.\n"
        "- See prompts.yaml for further agent formatting instructions.\n"
    )

    args_schema: Type[BaseModel] = CreditIntelligenceInput
    return_direct: bool = False
    _logger_instance: logging.Logger = PrivateAttr(default=None)
    _client_info: MCPImplementation = PrivateAttr(default_factory=lambda: MCPImplementation(name="CreditIntelligenceTransformerToolClient", version="0.1.0"))
    server_executable: str = sys.executable
    server_script: Path = Path(__file__).parent.parent / "mcp_servers" / "mcp_credit_intelligence_transformer" / "server.py"
    server_cwd_path: Path = server_script.parent

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def _tool_post_init(self):
        if self._logger_instance is None:
            self._logger_instance = logger.getChild(f"{self.name}.{id(self)}")
            if not self._logger_instance.hasHandlers():
                self._logger_instance.addHandler(logging.StreamHandler())
        self.server_script = self.server_script.resolve()
        self.server_cwd_path = self.server_cwd_path.resolve()
        self._logger_instance.info(f"Using server_cwd_path: {self.server_cwd_path}")
        return self

    def _get_server_params(self):
        return StdioServerParameters(
            command=self.server_executable,
            args=[str(self.server_script)],
            cwd=str(self.server_cwd_path)
        )


    async def close(self):
        if self._session:
            try:
                await self._session.close()
            except Exception as e:
                logger.warning(f"Error closing MCP session: {e}")
            self._session = None
        self._is_closed = True

    async def _arun(self, action: str, security_id: str):
        params = self._get_server_params()
        if not security_id:
            raise ValueError("security_id argument must be provided to tool client.")
        sec_dir = Path("/home/me/CascadeProjects/Alpy/otcmn_tool_test_output/current/secondary_board_B") / security_id / "data_extraction"
        data_path = sec_dir / "complete_data.json"
        if not data_path.exists():
            raise FileNotFoundError(f"complete_data.json not found for security_id {security_id}")
        with open(data_path, "r", encoding="utf-8") as f:
            full_data = json.load(f)
        async with stdio_client(params, errlog=sys.stderr) as (rs, ws):
            async with ClientSession(rs, ws, client_info=self._client_info) as session:
                await session.initialize()
                if action == "generate_assumptions":
                    # Call generate_assumptions directly with security_id
                    # The server will handle loading the data from the security folder
                    logger.info(f"Calling generate_assumptions for security_id {security_id}")
                    result = await session.call_tool(
                        name="generate_assumptions",
                        arguments={"security_id": security_id}
                    )
                    if getattr(result, "isError", False):
                        raise RuntimeError(f"MCP server error: {result.content}")
                    return result.content
                elif action == "run_financial_model":
                    # Call run_financial_model directly and deterministically
                    fm_result = await session.call_tool(
                        name="run_financial_model",
                        arguments={"security_id": security_id}
                    )
                    if getattr(fm_result, "isError", False):
                        raise RuntimeError(f"MCP server error (financial_model): {fm_result.content}")
                    return fm_result.content
                elif action == "summarize_credit_risk":
                    # generate_assumptions → run_financial_model → summarize_credit_risk
                    assumptions_result = await session.call_tool(
                        name="generate_assumptions",
                        arguments={"security_id": security_id}
                    )
                    if getattr(assumptions_result, "isError", False):
                        raise RuntimeError(f"MCP server error (assumptions): {assumptions_result.content}")
                    assumptions = assumptions_result.content
                    fm_input = {
                        "bond_financials_historical": full_data.get("bond_financials_historical", {}),
                        "assumptions": assumptions,
                        "bond_metadata": full_data.get("bond_metadata", {}),
                        "bond_financials_projections": full_data.get("bond_financials_projections")
                    }
                    fm_result = await session.call_tool(
                        name="run_financial_model",
                        arguments={"input_data": fm_input}
                    )
                    if getattr(fm_result, "isError", False):
                        raise RuntimeError(f"MCP server error (financial_model): {fm_result.content}")
                    fm_output = fm_result.content
                    summary_input = {
                        "financial_model_output": fm_output,
                        "qualitative_inputs": {
                            "issuer_business_profile": full_data.get("issuer_business_profile", {}),
                            "industry_profile": full_data.get("industry_profile", {}),
                            "collateral_and_protective_clauses": full_data.get("collateral_and_protective_clauses", {})
                        }
                    }
                    summary_result = await session.call_tool(
                        name="summarize_credit_risk",
                        arguments={"input_data": summary_input}
                    )
                    if getattr(summary_result, "isError", False):
                        raise RuntimeError(f"MCP server error (credit_risk): {summary_result.content}")
                    return summary_result.content
                else:
                    raise ValueError(f"Unknown action: {action}")



    def _run(self, *args, **kwargs):
        raise NotImplementedError("This tool only supports async execution.")

# --- Standalone Test Function ---
async def test_credit_intelligence_assumptions_tool(security_id="MN0LNDB68390"):
    print(f"Testing CreditIntelligenceTransformerTool (generate_assumptions) for security {security_id} with full extracted data...")
    tool = CreditIntelligenceTransformerTool()
    result = await tool._arun(action="generate_assumptions", security_id=security_id)
    print("\n=== Assumptions Tool Result ===")
    try:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except TypeError:
        print(str(result))
    await tool.close()

async def test_credit_intelligence_financial_model_tool(security_id="MN0LNDB68390"):
    print(f"Testing CreditIntelligenceTransformerTool (run_financial_model) for security {security_id} with full extracted data and assumptions...")
    tool = CreditIntelligenceTransformerTool()
    result = await tool._arun(action="run_financial_model", security_id=security_id)
    print("\n=== Financial Model Tool Result ===")
    try:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except TypeError:
        print(str(result))
    await tool.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_credit_intelligence_assumptions_tool())
    asyncio.run(test_credit_intelligence_financial_model_tool())
