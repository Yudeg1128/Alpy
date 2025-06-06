"""
LangChain Tool for Quality Checking Finalized Bond Data via MCP Server
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

class QualityCheckerInput(BaseModel):
    security_id: str = Field(..., description="Security identifier for the bond whose extracted data should be quality checked.")

class QualityCheckerTool(BaseTool, BaseModel):
    name: str = "quality_checker"
    description: str = (
        "Quality control for finalized bond extraction data.\n"
        "Call with: { 'security_id': <str> }\n"
        "Returns: QC report with fields: qc_status ('Pass' or 'Fail'), deterministic_qc_issues (list of detected critical problems), warnings (list of non-critical issues), value_corrections (object mapping paths to corrected values), and repair_instructions (for sections needing re-extraction).\n"
        "If qc_status is 'Fail', the quality checker will AUTOMATICALLY apply any possible direct corrections to the data file.\n"
        "The value_corrections object contains the exact paths and corrected values that were applied.\n"
        "For simple issues (like unit normalization), no further action is needed as corrections are applied directly.\n"
        "For more complex issues, the repair_instructions array will contain section-based instructions for complete re-extraction.\n"
        "Non-critical issues like documentation warnings, date format issues, and chronological inconsistencies are reported as warnings but do not cause QC failure.\n"
        "If repair_instructions are present, you MUST use them to call the data_extractor tool as follows:\n"
        "  Action: data_extractor\n"
        "  Action Input: {\n"
        "    'security_id': <same security_id>,\n"
        "    'action': <section_to_re_extract>,\n"
        "    'repair_prompt': <llm_repair_prompt_directive>\n"
        "  }\n"
        "IMPORTANT: The repair_prompt_directive will guide a COMPLETE re-extraction of the section with special focus on the problematic areas.\n"
        "Do not expect the repair_instructions to target specific fields - they will instruct a full section re-extraction with guidance on what aspects need special attention.\n"
        "Example Input: { 'security_id': 'MN0LNDB68390' }\n"
        "Example Output: { 'qc_status': 'Pass', 'corrections_applied': {}, 'fix_successful': false, 'warnings': [{ 'problem_type': 'MissingUnitNormalizationDocumentation', 'description': 'Missing unit conversion notes in bond_metadata.' }] }"
    )
    args_schema: Type[BaseModel] = QualityCheckerInput
    return_direct: bool = False
    _logger_instance: logging.Logger = PrivateAttr(default=None)
    _init_lock: Optional[asyncio.Lock] = PrivateAttr(default=None)
    _is_closed: bool = PrivateAttr(default=False)
    _client_info: MCPImplementation = PrivateAttr(default_factory=lambda: MCPImplementation(name="QualityCheckerToolClient", version="0.1.0"))
    server_executable: str = sys.executable
    server_script: Path = Path(__file__).parent.parent / "mcp_servers" / "mcp_quality_checker" / "server.py"
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

    async def _initialize_async_primitives(self):
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()

    def _get_server_params(self):
        return StdioServerParameters(
            command=self.server_executable,
            args=[str(self.server_script)],
            cwd=str(self.server_cwd_path)
        )

    async def _ensure_session_ready(self):
        await self._initialize_async_primitives()
        async with self._init_lock:
            if self._session is not None:
                return
            params = self._get_server_params()
            self._logger_instance.info(f"Starting MCP server: {params}")
            self._session = None
            try:
                async with stdio_client(params, errlog=sys.stderr) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream, client_info=self._client_info) as session:
                        await session.initialize()
                        self._session = session
            except Exception as e:
                self._logger_instance.error(f"Failed to start MCP session: {e}")
                raise

    async def _arun(self, security_id: str):
        # Locate finalized data and schema paths
        base_dir = Path("/home/me/CascadeProjects/Alpy/otcmn_tool_test_output/current")
        finalized_data_path = None
        for board_dir in base_dir.iterdir():
            if not board_dir.is_dir():
                continue
            sec_dir = board_dir / security_id / "data_extraction" / "complete_data.json"
            print(f"Checking for finalized data at: {sec_dir}")
            if sec_dir.exists():
                print(f"Found finalized data at: {sec_dir}")
                finalized_data_path = sec_dir
                break
        if not finalized_data_path or not finalized_data_path.exists():
            return {
                "qc_status": "error",
                "message": f"Finalized data file complete_data.json was not found for security_id {security_id}. Please ensure data extraction is completed before running quality checks.",
                "deterministic_qc_issues": [],
                "value_corrections": {}
            }
        schema_path = str(Path(__file__).parent.parent / "mcp_servers" / "mcp_quality_checker" / "qc_schema_with_corrections.json")
        arguments = {
            "input_data": {
                "finalized_data_path": str(finalized_data_path),
                "schema_path": schema_path
            }
        }
        from mcp.client.stdio import StdioServerParameters, stdio_client
        from mcp.client.session import ClientSession
        params = StdioServerParameters(
            command=self.server_executable,
            args=[str(self.server_script)],
            cwd=str(self.server_cwd_path)
        )
        async with stdio_client(params, errlog=sys.stderr) as (rs, ws):
            async with ClientSession(rs, ws, client_info=self._client_info) as session:
                await session.initialize()
                result = await session.call_tool(name="quality_check_final_data", arguments=arguments)
                if getattr(result, "isError", False):
                    raise RuntimeError(f"MCP server error: {result.content}")
                # Extract llm_assessment from TextContent response
                content = result.content
                # If response is a list of TextContent or strings
                if isinstance(content, list):
                    if not content:
                        return {"qc_status": "error", "message": "Empty response from server"}
                        
                    first_item = content[0]
                    if isinstance(first_item, str):
                        try:
                            qc_json = json.loads(first_item)
                            return qc_json.get("llm_assessment", qc_json)
                        except json.JSONDecodeError as e:
                            return {"qc_status": "error", "message": f"Server returned non-JSON string: {first_item}"}
                    elif hasattr(first_item, 'text'):
                        try:
                            qc_json = json.loads(first_item.text)
                            return qc_json.get("llm_assessment", qc_json)
                        except json.JSONDecodeError as e:
                            return {"qc_status": "error", "message": f"Failed to parse LLM QC JSON: {e}"}
                    else:
                        return {"qc_status": "error", "message": f"Unexpected response type: {type(first_item)}"}
                
                # If already dict
                if isinstance(content, dict):
                    if "llm_assessment" in content:
                        return content["llm_assessment"]
                    return content
                    
                # If string
                if isinstance(content, str):
                    try:
                        qc_json = json.loads(content)
                        return qc_json.get("llm_assessment", qc_json)
                    except json.JSONDecodeError:
                        return {"qc_status": "error", "message": f"Server returned non-JSON string: {content}"}
                        
                return {"qc_status": "error", "message": f"Unexpected response type: {type(content)}"}

    async def close(self):
        # No persistent session to close
        self._is_closed = True

    def _run(self, *args, **kwargs):
        raise NotImplementedError("This tool only supports async execution.")

# --- Standalone Test Function ---
async def test_quality_checker_tool():
    print("Testing QualityCheckerTool with security_id='MN0LNDB68390'...")
    tool = QualityCheckerTool()
    result = await tool._arun(security_id="MN0LNDB68390")
    print("\n=== QC Tool Result ===")
    try:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except TypeError:
        print(str(result))
    await tool.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_quality_checker_tool())
