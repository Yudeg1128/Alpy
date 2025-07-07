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
    action: str = Field(..., description="Which phase action to run: 'generate_assumptions', 'run_financial_model', 'generate_llm_financial_model', 'stress_test_financial_model', 'run_deterministic_scenarios', 'summarize_credit_risk', 'map_historical_data_to_schema', 'validate_financial_statements', 'validate_inter_period_consistency', or 'reconcile_debt_schedule'.")
    security_id: str = Field(..., description="Security ID to analyze. The tool will automatically load the full extracted data for this security.")

class CreditIntelligenceTransformerTool(BaseTool, BaseModel):
    _session: ClientSession = PrivateAttr(default=None)
    _is_closed: bool = PrivateAttr(default=False)
    name: str = "credit_intelligence_transformer"
    description: str = (
        "CREDIT INTELLIGENCE TRANSFORMER TOOL\n"
        "Purpose: Run multi-phase credit analysis for a security using extracted data. "
        "Supports thirteen actions: 'generate_assumptions', 'run_financial_model', 'generate_llm_financial_model', 'stress_test_financial_model', 'run_deterministic_scenarios', 'summarize_credit_risk', 'map_historical_data_to_schema', 'validate_financial_statements', 'validate_inter_period_consistency', 'generate_baseline_projection_drivers', 'run_deterministic_projections', 'reclassify_financial_plugs', 'reconcile_debt_schedule'. "
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
        "Where PHASE_ACTION is one of: 'generate_assumptions', 'run_financial_model', 'generate_llm_financial_model', 'stress_test_financial_model', 'run_deterministic_scenarios', 'summarize_credit_risk', or 'map_historical_data_to_schema'.\n\n"
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
        "  ```\n"
        "- To run stress tests on the financial model:\n"
        "  Action:\n"
        "  ```json\n"
        "  {\n"
        "    \"action\": \"credit_intelligence_transformer\",\n"
        "    \"action_input\": {\n"
        "      \"security_id\": \"MN0LNDB68390\",\n"
        "      \"action\": \"stress_test_financial_model\"\n"
        "    }\n"
        "  }\n"
        "  ```\n"
        "- To run deterministic scenarios for correlation analysis:\n"
        "  Action:\n"
        "  ```json\n"
        "  {\n"
        "    \"action\": \"credit_intelligence_transformer\",\n"
        "    \"action_input\": {\n"
        "      \"security_id\": \"MN0LNDB68390\",\n"
        "      \"action\": \"run_deterministic_scenarios\"\n"
        "    }\n"
        "  }\n"
        "  ```\n\n"
        "OUTPUT FORMAT:\n"
        "Returns a dict containing the result of the requested phase.\n"
        "- For 'generate_assumptions': { ...assumptions... }\n"
        "- For 'run_financial_model': { ...financial_model_output... }\n"
        "- For 'generate_llm_financial_model': { ...llm_financial_model_output_with_explanations... }\n"
        "- For 'stress_test_financial_model': { ...stress_test_scenarios_and_results... }\n"
        "- For 'run_deterministic_scenarios': [ {scenario_name, financial_metrics}, ... ]\n"
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

    async def _arun(self, action: str, security_id: str) -> Dict[str, Any]:
        logger.warning(f"Executing action: {action} for security_id: {security_id}")
        """Execute the credit intelligence transformer tool action."""
        # Validate action
        VALID_ACTIONS = ["generate_assumptions", "run_financial_model", "generate_llm_financial_model", 
                         "stress_test_financial_model", "run_deterministic_scenarios", "summarize_credit_risk",
                         "map_historical_data_to_schema", "validate_financial_statements", 
                         "validate_inter_period_consistency", "generate_baseline_projection_drivers",
                         "run_deterministic_projections", "reclassify_financial_plugs",
                         "reconcile_debt_schedule", "generate_jit_validation_schema",
                         "generate_jit_cfs_derivation_schema", "generate_jit_projections_schema",
                         "generate_projection_drivers"]
        if action not in VALID_ACTIONS:
            raise ValueError(f"Invalid action: {action}. Must be one of {VALID_ACTIONS}")
        
        # Validate security_id
        if not security_id:
            raise ValueError("security_id must be provided")
        
        params = self._get_server_params()
        if not security_id:
            raise ValueError("security_id argument must be provided to tool client.")
        
        async with stdio_client(params, errlog=sys.stderr) as (rs, ws):
            logger.warning("stdio_client connection established")
            async with ClientSession(rs, ws, client_info=self._client_info) as session:
                logger.warning("ClientSession created, initializing...")
                await session.initialize()
                logger.warning("ClientSession initialized successfully")
                if action == "generate_assumptions":
                    logger.warning(f"Calling generate_assumptions for security_id {security_id}")
                    result = await session.call_tool(
                        name="generate_assumptions",
                        arguments={"security_id": security_id}
                    )
                    if getattr(result, "isError", False):
                        raise RuntimeError(f"MCP server error: {result.content}")
                    return result.content
                elif action == "run_financial_model":
                    fm_result = await session.call_tool(
                        name="run_financial_model",
                        arguments={"security_id": security_id}
                    )
                    if getattr(fm_result, "isError", False):
                        raise RuntimeError(f"MCP server error (financial_model): {fm_result.content}")
                    return fm_result.content
                elif action == "generate_llm_financial_model":
                    llm_fm_result = await session.call_tool(
                        name="generate_llm_financial_model",
                        arguments={"security_id": security_id}
                    )
                    if getattr(llm_fm_result, "isError", False):
                        raise RuntimeError(f"MCP server error (llm_financial_model): {llm_fm_result.content}")
                    return llm_fm_result.content
                elif action == "stress_test_financial_model":
                    stress_result = await session.call_tool(
                        name="stress_test_financial_model",
                        arguments={"security_id": security_id}
                    )
                    if getattr(stress_result, "isError", False):
                        raise RuntimeError(f"MCP server error (stress_test): {stress_result.content}")
                    return stress_result.content
                elif action == "run_deterministic_scenarios":
                    det_scenarios_result = await session.call_tool(
                        name="run_deterministic_scenarios",
                        arguments={"security_id": security_id}
                    )
                    if getattr(det_scenarios_result, "isError", False):
                        raise RuntimeError(f"MCP server error (deterministic_scenarios): {det_scenarios_result.content}")
                    
                    # Handle TextContent objects in the result
                    content = det_scenarios_result.content
                    if isinstance(content, list):
                        # Convert TextContent objects to dictionaries
                        processed_content = []
                        for item in content:
                            if hasattr(item, 'content'):
                                # TextContent object - extract the content
                                try:
                                    # Try to parse as JSON if it's a string representation of JSON
                                    import json
                                    # Check if content has type and text attributes (TextContent structure)
                                    if hasattr(item.content, 'type') and hasattr(item.content, 'text') and item.content.type == 'text':
                                        processed_item = json.loads(item.content.text)
                                    else:
                                        # Regular string content
                                        processed_item = json.loads(str(item.content))
                                    processed_content.append(processed_item)
                                except (json.JSONDecodeError, AttributeError) as e:
                                    # If not JSON, just use the string content
                                    content_str = str(item.content)
                                    if hasattr(item.content, 'text'):
                                        content_str = item.content.text
                                    processed_content.append({"scenario_name": f"scenario_{len(processed_content)+1}", "scenario_data": content_str})
                            else:
                                # Already a dict or other serializable object
                                processed_content.append(item)
                        return processed_content
                    return det_scenarios_result.content
                elif action == "summarize_credit_risk":
                    logger.info(f"Calling summarize_credit_risk for security_id {security_id}")
                    summary_result = await session.call_tool(
                        name="summarize_credit_risk",
                        arguments={"security_id": security_id}
                    )
                    if getattr(summary_result, "isError", False):
                        raise RuntimeError(f"MCP server error (credit_risk): {summary_result.content}")
                    return summary_result.content
                elif action == "map_historical_data_to_schema":
                    logger.info(f"Calling map_historical_data_to_schema for security_id {security_id}")
                    mapping_result = await session.call_tool(
                        name="map_historical_data_to_schema",
                        arguments={"security_id": security_id}
                    )
                    if getattr(mapping_result, "isError", False):
                        raise RuntimeError(f"MCP server error (map_historical_data_to_schema): {mapping_result.content}")
                    return mapping_result.content
                elif action == "validate_financial_statements":
                    logger.info(f"Calling validate_financial_statements for security_id {security_id}")
                    validation_result = await session.call_tool(
                        name="validate_financial_statements",
                        arguments={"security_id": security_id}
                    )
                    if getattr(validation_result, "isError", False):
                        raise RuntimeError(f"MCP server error (validate_financial_statements): {validation_result.content}")
                    return validation_result.content
                elif action == "generate_baseline_projection_drivers":
                    logger.info(f"Calling generate_baseline_projection_drivers for security_id {security_id}")
                    baseline_drivers_result = await session.call_tool(
                        name="generate_baseline_projection_drivers",
                        arguments={"security_id": security_id}
                    )
                    if getattr(baseline_drivers_result, "isError", False):
                        raise RuntimeError(f"MCP server error (generate_baseline_projection_drivers): {baseline_drivers_result.content}")
                    return baseline_drivers_result.content
                elif action == "run_deterministic_projections":
                    logger.info(f"Calling run_deterministic_projections for security_id {security_id}")
                    projections_result = await session.call_tool(
                        name="run_deterministic_projections",
                        arguments={"security_id": security_id}
                    )
                    if getattr(projections_result, "isError", False):
                        raise RuntimeError(f"MCP server error (run_deterministic_projections_tool): {projections_result.content}")
                    return projections_result.content
                elif action == "reclassify_financial_plugs":
                    logger.info(f"Calling reclassify_financial_plugs for security_id {security_id}")
                    reclassify_result = await session.call_tool(
                        name="reclassify_financial_plugs",
                        arguments={"security_id": security_id}
                    )
                    if getattr(reclassify_result, "isError", False):
                        raise RuntimeError(f"MCP server error (reclassify_financial_plugs): {reclassify_result.content}")
                    return reclassify_result.content
                elif action == "reconcile_debt_schedule":
                    logger.info(f"Calling reconcile_debt_schedule for security_id {security_id}")
                    reconcile_result = await session.call_tool(
                        name="reconcile_debt_schedule",
                        arguments={"security_id": security_id}
                    )
                    if getattr(reconcile_result, "isError", False):
                        raise RuntimeError(f"MCP server error (reconcile_debt_schedule): {reconcile_result.content}")
                    return reconcile_result.content
                elif action == "generate_jit_validation_schema":
                    logger.info(f"Calling generate_jit_validation_schema for security_id {security_id}")
                    jit_schema_result = await session.call_tool(
                        name="generate_jit_validation_schema",
                        arguments={"security_id": security_id}
                    )
                    if getattr(jit_schema_result, "isError", False):
                        raise RuntimeError(f"MCP server error (generate_jit_validation_schema): {jit_schema_result.content}")
                    return jit_schema_result.content
                elif action == "generate_jit_cfs_derivation_schema":
                    logger.info(f"Calling generate_jit_cfs_derivation_schema for security_id {security_id}")
                    jit_schema_result = await session.call_tool(
                        name="generate_jit_cfs_derivation_schema",
                        arguments={"security_id": security_id}
                    )
                    if getattr(jit_schema_result, "isError", False):
                        raise RuntimeError(f"MCP server error (generate_jit_cfs_derivation_schema): {jit_schema_result.content}")
                    return jit_schema_result.content
                elif action == "generate_jit_projections_schema":
                    logger.info(f"Calling generate_jit_projections_schema for security_id {security_id}")
                    jit_schema_result = await session.call_tool(
                        name="generate_jit_projections_schema",
                        arguments={"security_id": security_id}
                    )
                    if getattr(jit_schema_result, "isError", False):
                        raise RuntimeError(f"MCP server error (generate_jit_projections_schema): {jit_schema_result.content}")
                    return jit_schema_result.content
                elif action == "generate_projection_drivers":
                    logger.info(f"Calling generate_projection_drivers for security_id {security_id}")
                    projection_drivers_result = await session.call_tool(
                        name="generate_projection_drivers",
                        arguments={"security_id": security_id}
                    )
                    if getattr(projection_drivers_result, "isError", False):
                        raise RuntimeError(f"MCP server error (generate_projection_drivers): {projection_drivers_result.content}")
                    return projection_drivers_result.content
                else:
                    raise ValueError(f"Unknown action: {action}")



    def _run(self, *args, **kwargs):
        raise NotImplementedError("This tool only supports async execution.")

# --- Standalone Test Function ---

async def test_credit_intelligence_stress_test_tool(security_id="MN0LNDB68390"):
    print(f"Testing CreditIntelligenceTransformerTool (stress_test_financial_model) for security {security_id}...")
    tool = CreditIntelligenceTransformerTool()
    
    # Run stress test without timeout
    print("Running stress test (this may take some time)...")
    try:
        result = await tool._arun(action="stress_test_financial_model", security_id=security_id)
    except Exception as e:
        print(f"Error running stress test: {e}")
        await tool.close()
        return
    print("\n=== Stress Test Tool Result ===")
    try:
        # Handle different result formats
        if isinstance(result, dict):
            # The result might have a nested 'scenarios' key
            if "scenarios" in result and isinstance(result["scenarios"], list):
                scenarios = result["scenarios"]
            else:
                # Or it might be a dict with scenarios as direct keys
                scenarios = result.get("scenarios", [])
        elif isinstance(result, list):
            scenarios = result
        else:
            scenarios = []
            print(f"Unexpected result type: {type(result)}")
        
        print(f"Raw result structure: {type(result)}")
        if isinstance(result, dict):
            print(f"Result keys: {list(result.keys())}")
        print(f"Scenarios type: {type(scenarios)}")
            
        print(f"Generated {len(scenarios)} stress test scenarios:")
        
        for i, scenario in enumerate(scenarios):
            if isinstance(scenario, dict):
                name = scenario.get('name', f'Scenario {i+1}')
                desc = scenario.get('description', 'No description')
                print(f"\nScenario {i+1}: {name}")
                print(f"Description: {desc[:150]}..." if len(desc) > 150 else f"Description: {desc}")
            else:
                print(f"\nScenario {i+1}: {str(scenario)[:50]}...")
        
        # Print location of results
        security_folder = Path(f"/home/me/CascadeProjects/Alpy/otcmn_tool_test_output/current/secondary_board_B/{security_id}/credit_analysis/stress_test")
        results_path = security_folder / "stress_test_results.json"
        print(f"\nFull results saved to {results_path}")
        
        # Print scenarios folder structure
        print("\nScenario folders:")
        for i, scenario in enumerate(scenarios):
            if isinstance(scenario, dict):
                scenario_name = scenario.get('name', f'scenario_{i+1}').replace(" ", "_").lower()
            else:
                scenario_name = f"scenario_{i+1}"
            print(f"  - {security_folder / scenario_name}")

    except Exception as e:
        print(f"Error processing result: {e}")
        print(f"Raw result: {str(result)[:500]}..." if len(str(result)) > 500 else str(result))
    await tool.close()


async def run_full_credit_workflow(security_id="MN0LNDB68390", use_llm_model=False):
    """Run the full credit intelligence workflow: financial model → stress test → deterministic scenarios → credit risk summary
    
    Args:
        security_id: The security ID to analyze
        use_llm_model: If True, use the LLM financial model instead of the deterministic one
    """
    print(f"\n=== Running full credit intelligence workflow for {security_id} ===\n")
    tool = CreditIntelligenceTransformerTool()
    
    # Step 1: Run financial model (either deterministic or LLM-based)
    if use_llm_model:
        print("Step 1: Running LLM-based financial model...")
        try:
            fm_result = await tool._arun(action="generate_llm_financial_model", security_id=security_id)
            print("✓ LLM financial model completed successfully")
        except Exception as e:
            print(f"Error running LLM financial model: {e}")
            print("Continuing anyway, in case files already exist...")
    else:
        print("Step 1: Running deterministic financial model...")
        try:
            fm_result = await tool._arun(action="run_financial_model", security_id=security_id)
            print("✓ Financial model completed successfully")
        except Exception as e:
            print(f"Error running financial model: {e}")
            print("Continuing anyway, in case files already exist...")
    
    # Step 2: Run stress test
    print("\nStep 2: Running stress test...")
    try:
        stress_result = await tool._arun(action="stress_test_financial_model", security_id=security_id)
        print("✓ Stress test completed successfully")
    except Exception as e:
        print(f"Error running stress test: {e}")
        print("Continuing with next steps anyway...")
    
    # Step 3: Run deterministic scenarios
    print("\nStep 3: Running deterministic scenarios...")
    try:
        det_scenarios_result = await tool._arun(action="run_deterministic_scenarios", security_id=security_id)
        print("✓ Deterministic scenarios completed successfully")
    except Exception as e:
        print(f"Error running deterministic scenarios: {e}")
        print("Continuing with credit risk summary anyway...")
    
    # Step 4: Run credit risk summary
    print("\nStep 4: Running credit risk summary...")
    result = await tool._arun(action="summarize_credit_risk", security_id=security_id)
    
    print("\n=== Credit Risk Summary Result ===")
    print(f"Result type: {type(result)}")
    
    # Pretty print the result if it's a dict
    if isinstance(result, dict):
        import json
        print(json.dumps(result, indent=2)[:1000])  # Show first 1000 chars
        print("...\n(output truncated)")
    else:
        print(str(result)[:1000])  # Show first 1000 chars
        print("...\n(output truncated)")
    
    print("\n✓ Full credit intelligence workflow completed successfully")
    await tool.close()


async def test_map_historical_data_to_schema_tool(security_id="MN0LNDB68390"):
    """Test the map_historical_data_to_schema tool action"""
    print(f"\n=== Testing map_historical_data_to_schema for {security_id} ===\n")
    tool = CreditIntelligenceTransformerTool()
    
    try:
        print("Calling map_historical_data_to_schema...")
        result = await tool._arun(action="map_historical_data_to_schema", security_id=security_id)
        
        print("\n=== Map Historical Data Result ===\n")
        print(f"Result type: {type(result)}")
        
        # Pretty print the result if it's a dict
        if isinstance(result, dict):
            print(json.dumps(result, indent=2)[:1000])  # Show first 1000 chars
            print("...\n(output truncated)")
            
            # Check if mapped_historical_data exists and has entries
            if "mapped_historical_data" in result and result["mapped_historical_data"]:
                print(f"\nSuccessfully mapped {len(result['mapped_historical_data'])} historical periods")
                
                # Check the first period's mapping decisions
                first_period = result["mapped_historical_data"][0]
                if "mapping_decisions" in first_period:
                    print(f"First period has {len(first_period['mapping_decisions'])} mapping decisions")
        
        print("\n✓ Map historical data test completed successfully")
    except asyncio.TimeoutError:
        print("\n❌ Test timed out - this may be normal if the LLM call takes a long time")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
    finally:
        await tool.close()


async def test_validate_financial_statements_tool(security_id="MN0LNDB68390"):
    """Test the validate_financial_statements action."""
    print(f"\n=== Testing validate_financial_statements for {security_id} ===\n")
    
    print("Calling validate_financial_statements...")
    tool = CreditIntelligenceTransformerTool()
    
    try:
        result = await tool._arun("validate_financial_statements", security_id)
        
        print("\n=== Financial Statement Validation Result ===\n")
        
        # Check for error response
        if isinstance(result, dict) and "error" in result:
            print(f"❌ Error: {result['error']}")
            if "validation_metadata" in result:
                print(f"Validation metadata: {result['validation_metadata']}")
            return
        
        # Print basic result information
        print(f"Result type: {type(result)}")
        
        # If we have a dictionary with validation_metadata
        if isinstance(result, dict) and "validation_metadata" in result:
            metadata = result["validation_metadata"]
            print(f"\nValidation summary: {metadata.get('validated_periods', 0)} periods validated, "
                  f"{metadata.get('skipped_periods', 0)} periods skipped")
            
            # Show summary of plug items if available
            if "validation_summary" in metadata:
                plug_count = sum(len(period.get("plug_items", [])) 
                               for period in metadata["validation_summary"] 
                               if "plug_items" in period)
                print(f"Total plug items added: {plug_count}")
        
        # If we have a list of validated periods
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            try:
                validated_periods = sum(1 for period in result if not period.get("skipped", False))
                skipped_periods = sum(1 for period in result if period.get("skipped", True))
                plug_items = sum(len(period.get("plug_items", [])) 
                               for period in result 
                               if not period.get("skipped", False) and "plug_items" in period)
                
                print(f"\nValidated periods: {validated_periods}")
                print(f"Skipped periods: {skipped_periods}")
                print(f"Plug items added: {plug_items}")
            except (AttributeError, TypeError) as e:
                print(f"Could not extract validation metrics: {e}")
        
        print("\n✓ Financial statement validation test completed successfully")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
    finally:
        await tool.close()


# Test the generate_baseline_projection_drivers action
async def test_generate_baseline_projection_drivers_tool(security_id="MN0LNDB68390"):
    """Run the generate_baseline_projection_drivers tool and return the result."""
    tool = None
    try:
        tool = CreditIntelligenceTransformerTool()
        result = await tool._arun(
            action="generate_baseline_projection_drivers", 
            security_id=security_id
        )
        return json.loads(result) if isinstance(result, str) else result
    finally:
        if tool and hasattr(tool, 'close') and callable(tool.close):
            await tool.close()


# Test the generate_jit_validation_schema action
async def test_generate_jit_validation_schema_tool(security_id: str = "MN0LNDB68390"):
    """Run the generate_jit_validation_schema tool and return the result."""
    tool = None
    try:
        tool = CreditIntelligenceTransformerTool()
        result = await tool._arun(
            action="generate_jit_validation_schema", 
            security_id=security_id
        )
        return json.loads(result) if isinstance(result, str) else result
    finally:
        if tool and hasattr(tool, 'close') and callable(tool.close):
            await tool.close()


async def test_generate_jit_projections_schema_tool(security_id: str = "MN0LNDB68390"):
    """Run the generate_jit_projections_schema tool and return the result."""
    tool = None
    try:
        tool = CreditIntelligenceTransformerTool()
        result = await tool._arun(
            action="generate_jit_projections_schema", 
            security_id=security_id
        )
        return json.loads(result) if isinstance(result, str) else result
    finally:
        if tool and hasattr(tool, 'close') and callable(tool.close):
            await tool.close()


# Test the generate_jit_cfs_derivation_schema action
async def test_generate_jit_cfs_derivation_schema_tool(security_id: str = "MN0LNDB68390"):
    """Run the generate_jit_cfs_derivation_schema tool and return the result."""
    tool = None
    try:
        tool = CreditIntelligenceTransformerTool()
        result = await tool._arun(
            action="generate_jit_cfs_derivation_schema", 
            security_id=security_id
        )
        return json.loads(result) if isinstance(result, str) else result
    finally:
        if tool and hasattr(tool, 'close') and callable(tool.close):
            await tool.close()


async def test_generate_projection_drivers_tool(security_id: str = "MN0LNDB68390"):
    """Run the generate_projection_drivers tool and return the result."""
    tool = None
    try:
        tool = CreditIntelligenceTransformerTool()
        result = await tool._arun(
            action="generate_projection_drivers", 
            security_id=security_id
        )
        return json.loads(result) if isinstance(result, str) else result
    finally:
        if tool and hasattr(tool, 'close') and callable(tool.close):
            await tool.close()


async def test_reconcile_debt_schedule_tool(security_id: str = "MN0LNDB68390"):
    """Test the reconcile_debt_schedule tool action"""
    import traceback
    
    print(f"\n=== Testing reconcile_debt_schedule for security_id={security_id} ===")
    
    # Initialize the tool
    print("Initializing CreditIntelligenceTransformerTool...")
    tool = CreditIntelligenceTransformerTool()
    
    try:
        # Call the reconcile_debt_schedule action directly
        print("\nCalling reconcile_debt_schedule action...")
        print(f"Action: reconcile_debt_schedule, Security ID: {security_id}")
        
        # Get server params for debugging
        params = tool._get_server_params()
        try:
            # Try to access as dict first
            print(f"\nServer params: command={params['command']!r} args={params['args']!r} "
                  f"env={params['env']} cwd={params['cwd']!r} encoding={params['encoding']!r} "
                  f"encoding_error_handler={params['encoding_error_handler']!r}")
        except (TypeError, KeyError):
            # If params is not a dict, try to get attributes
            print("\nServer params:")
            if hasattr(params, 'command'):
                print(f"  Command: {params.command}")
            if hasattr(params, 'args'):
                print(f"  Args: {params.args}")
            if hasattr(params, 'cwd'):
                print(f"  CWD: {params.cwd}")
            if hasattr(params, 'env'):
                print(f"  Env: {params.env}")
            if hasattr(params, 'encoding'):
                print(f"  Encoding: {params.encoding}")
        
        # Make the actual call with error handling
        try:
            result = await tool._arun(
                action="reconcile_debt_schedule",
                security_id=security_id
            )
            
            # Check if result indicates an error
            if isinstance(result, dict) and result.get('status') == 'error':
                print(f"\n❌ Error from reconcile_debt_schedule: {result.get('message', 'Unknown error')}")
                return
                
            # Print the result
            print("\n✅ Reconcile Debt Schedule Result:")
            
            # Handle TextContent objects in the response
            if hasattr(result, 'content') and hasattr(result.content, 'text'):
                # If it's a TextContent object, extract the text
                result_content = result.content.text
                try:
                    # Try to parse as JSON if it looks like JSON
                    if result_content.strip().startswith('{') or result_content.strip().startswith('['):
                        result_data = json.loads(result_content)
                        print(json.dumps(result_data, indent=2))
                    else:
                        print(result_content)
                except json.JSONDecodeError:
                    print(result_content)
            else:
                # Regular result handling
                try:
                    print(json.dumps(result, indent=2, default=str))
                except (TypeError, ValueError) as e:
                    print(f"Result (could not serialize as JSON): {result}")
                    print(f"Error details: {e}")
            
            # Print a summary of short-term and long-term debt by reporting period
            if result and isinstance(result, dict) and "financial_data" in result and len(result["financial_data"]) > 0:
                print("\n📊 Short-term and Long-term Debt by Reporting Period:")
                for period_data in result["financial_data"]:
                    if not isinstance(period_data, dict):
                        continue
                        
                    period = period_data.get("period", period_data.get("reporting_period_end_date", "Unknown Period"))
                    print(f"\n📅 {period}:")
                    
                    # Handle different possible structures
                    bs = period_data.get("balance_sheet", {})
                    if not bs and isinstance(period_data, dict):
                        bs = {k: v for k, v in period_data.items() if isinstance(v, (dict, int, float))}
                    
                    current = bs.get("current_liabilities", {})
                    if not current or not isinstance(current, dict):
                        current = {k: v for k, v in bs.items() if "current" in str(k).lower() and isinstance(v, (int, float))}
                    
                    non_current = bs.get("non_current_liabilities", {})
                    if not non_current or not isinstance(non_current, dict):
                        non_current = {k: v for k, v in bs.items() if "non" in str(k).lower() and "current" in str(k).lower() and isinstance(v, (int, float))}
                    
                    short_term = current.get("short_term_debt") or current.get("short_term_debt_1") or \
                                next((v for k, v in current.items() if "short" in str(k).lower() and "debt" in str(k).lower() and isinstance(v, (int, float))), "N/A")
                    
                    long_term = non_current.get("long_term_debt") or non_current.get("long_term_debt_1") or \
                              next((v for k, v in non_current.items() if "long" in str(k).lower() and "debt" in str(k).lower() and isinstance(v, (int, float))), "N/A")
                    
                    print(f"  Short-term debt: {short_term}")
                    print(f"  Long-term debt: {long_term}")
            
            # Load and display the reconciled data
            from financial_analyst.security_folder_utils import get_security_file
            
            reconciled_file = get_security_file(security_id, "credit_analysis/reconciled_financial_data.json")
            if os.path.exists(reconciled_file):
                with open(reconciled_file, 'r') as f:
                    try:
                        reconciled_data = json.load(f)
                        print("\n✅ Reconciled financial data saved to:", reconciled_file)
                        
                        # Print summary of changes
                        print("\nSummary of debt reconciliation:")
                        for period in reconciled_data.get('mapped_historical_data', []):
                            if not isinstance(period, dict):
                                continue
                                
                            period_end = period.get('reporting_period_end_date')
                            if not period_end:
                                continue
                                
                            bs = period.get('balance_sheet', {})
                            current_liab = bs.get('current_liabilities', {})
                            non_current_liab = bs.get('non_current_liabilities', {})
                            
                            print(f"\nPeriod: {period_end}")
                            print(f"  Short-term debt: {current_liab.get('short_term_debt', 0):,.0f}")
                            print(f"  Long-term debt: {non_current_liab.get('long_term_debt', 0):,.0f}")
                    except json.JSONDecodeError as e:
                        print(f"\n⚠️  Error loading reconciled data: {e}")
                    except Exception as e:
                        print(f"\n⚠️  Error processing reconciled data: {e}")
            
            print("\n✓ Debt schedule reconciliation completed successfully")
            
        except asyncio.exceptions.CancelledError:
            print("\n❌ Operation was cancelled (timeout or cancellation requested)")
        except Exception as e:
            print(f"\n❌ Error calling reconcile_debt_schedule: {e}")
            print("\nStack trace:")
            traceback.print_exc()
            
            # Check for common issues
            if "No module named" in str(e):
                print("\n⚠️  Module import error. Make sure all dependencies are installed.")
            elif "file not found" in str(e).lower():
                print("\n⚠️  File not found. Check if the required data files exist for this security.")
            elif "timeout" in str(e).lower():
                print("\n⚠️  Operation timed out. The server may be taking too long to respond.")
                
            print("\nPlease check the server logs for more details.")
    except Exception as e:
        print(f"\n❌ Unexpected error in test: {e}")
        traceback.print_exc()
    finally:
        await tool.close()

if __name__ == "__main__":
    import asyncio
    import sys
    
    asyncio.run(test_generate_jit_projections_schema_tool())
