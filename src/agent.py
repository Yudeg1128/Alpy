import logging
import yaml
import re
import os
import json
import traceback
import sys
from typing import Dict, Any, List, Union, Optional

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentExecutor, create_react_agent, BaseSingleActionAgent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, HumanMessage
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain.agents.agent import AgentOutputParser
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI

from . import config
from .tools.mcp_bash_tool import MCPBashExecutorTool
from .tools.mcp_python_tool import MCPPythonExecutorTool
from .tools.mcp_filesystem_tool import MCPFileSystemTool
from .tools.mcp_brave_search_tool import MCPBraveSearchTool
from .tools.mcp_media_display_tool import MCPMediaDisplayTool
from .tools.mcp_fetcher_web_tool import MCPFetcherWebTool
from .tools.mcp_puppeteer_tool import MCPPuppeteerTool
from rich.console import Console as RichConsole
from rich.panel import Panel as RichPanel
from rich.markdown import Markdown as RichMarkdown

logger = logging.getLogger(__name__)

ACTION_PATTERN = r"Action\s*:\s*```json\s*([\s\S]+?)\s*```" # More specific to capture JSON block content
FINAL_ANSWER_PATTERN = r"Final Answer\s*:\s*(.*)"
THINK_PATTERN = r"^\s<think>.?</think>\s"

class RichStreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, console: Optional[RichConsole] = None) -> None:
        super().__init__()
        self.trace: List[str] = []
        self._current_llm_output_parts: List[str] = []
        self.console = console
        self.action_count = 0

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        self._current_llm_output_parts = []

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self._current_llm_output_parts.append(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        full_llm_response = "".join(self._current_llm_output_parts).strip()
        self._current_llm_output_parts = [] # Reset

        if not full_llm_response:
            # logger.debug("on_llm_end called with no new tokens.") # Optional debug
            return

        # Add raw LLM output to trace
        self.trace.append(full_llm_response)
        logger.debug(f"RichStreamingCallbackHandler.on_llm_end received: {full_llm_response[:200]}...") # Log what the LLM produces

        # Default title assumes it's a thought or general LLM reasoning
        title = "💡 LLM Thought/Output"
        panel_style = "green" # Default, for thoughts

        action_match = re.search(ACTION_PATTERN, full_llm_response, re.DOTALL | re.IGNORECASE)
        
        if action_match:
            title = "⚙️ LLM Action Proposal"
            panel_style = "cyan" # Brighter for actions
        # We are not trying to identify "Final Answer" here as it's handled by main.py
        # All non-tool-observation output from the LLM will be panelled here.

        if self.console: # Ensure console exists
            self.console.print(
                RichPanel(
                    RichMarkdown(full_llm_response, code_theme="monokai"),
                    title=title,
                    border_style=panel_style,
                    expand=False
                )
            )
            sys.stdout.flush()
        else:
            logger.warning("RichStreamingCallbackHandler.on_llm_end: No console to print to.")

    def on_tool_end(
        self,
        output: str, # output is already a string from the tool
        color: Union[str, None] = None, # Keep for signature compatibility
        observation_prefix: Union[str, None] = None, # Keep
        llm_prefix: Union[str, None] = None, # Keep
        **kwargs: Any,
    ) -> None:
        # 'output' here IS the observation string from the tool.
        observation_text = f"{output.strip()}" # output is already the result string
        self.trace.append(f"Observation: {observation_text}") 
        logger.debug(f"RichStreamingCallbackHandler.on_tool_end received observation: {observation_text[:200]}...")

        if self.console:
            self.console.print(
                RichPanel(
                    # Present observation as pre-formatted text / markdown code block
                    RichMarkdown(f"```text\n{observation_text}\n```", code_theme="monokai"), 
                    title="🔭 Observation",
                    border_style="yellow", # This is a valid Rich style
                    expand=False
                )
            )
            sys.stdout.flush()
        else:
            logger.warning("RichStreamingCallbackHandler.on_tool_end: No console to print to.")

    def get_full_trace(self) -> str:
        return "\n".join(part for part in self.trace if part.strip()).strip()


class CustomReActJsonOutputParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        logger.debug(f"CustomReActJsonOutputParser received text: <<<\\n{text}\\n>>>")

        # Define patterns for block headers. We're looking for the start of these blocks.
        # We use re.IGNORECASE and re.DOTALL for flexibility.
        action_header_pattern = r"Action\s*:"
        final_answer_header_pattern = r"Final Answer\s*:"
        # Observation is a common delimiter that might appear after an action.
        observation_header_pattern = r"Observation\s*:"

        action_header_match = re.search(action_header_pattern, text, re.IGNORECASE | re.DOTALL)
        final_answer_header_match = re.search(final_answer_header_pattern, text, re.IGNORECASE | re.DOTALL)

        action_json_str = None
        parsed_action_successfully = False
        action_block_content_for_error = "Not processed"

        if action_header_match:
            logger.debug(f"Found 'Action:' header at index {action_header_match.start()}.")
            action_content_start_index = action_header_match.end()

            # Determine where the action block content might end.
            # It could end before "Final Answer:", "Observation:", or at the end of the text.
            potential_end_points = [len(text)]
            if final_answer_header_match and final_answer_header_match.start() > action_content_start_index:
                potential_end_points.append(final_answer_header_match.start())
            
            # Search for Observation only *after* the action header's end
            text_after_action_header = text[action_content_start_index:]
            observation_match_in_action_block = re.search(observation_header_pattern, text_after_action_header, re.IGNORECASE | re.DOTALL)
            if observation_match_in_action_block:
                 potential_end_points.append(action_content_start_index + observation_match_in_action_block.start())

            action_content_end_index = min(potential_end_points)
            action_block_text = text[action_content_start_index:action_content_end_index].strip()
            action_block_content_for_error = action_block_text # For error reporting
            logger.debug(f"Extracted potential action block content: <<<\\n{action_block_text}\\n>>>")

            if not action_block_text:
                logger.warning("Action block is empty after stripping.")
            else:
                # Tiered approach to find JSON in the action_block_text:
                # 1. Try to find ```json ... ```
                json_fence_match = re.search(r"```json\s*([\s\S]+?)\s*```", action_block_text, re.DOTALL)
                if json_fence_match:
                    action_json_str = json_fence_match.group(1).strip()
                    logger.debug(f"Found JSON content via ```json fence: '{action_json_str}'")
                else:
                    # 2. Try to find ``` ... ``` (generic code block)
                    generic_fence_match = re.search(r"```\s*([\s\S]+?)\s*```", action_block_text, re.DOTALL)
                    if generic_fence_match:
                        action_json_str = generic_fence_match.group(1).strip()
                        logger.debug(f"Found JSON content via generic ``` fence: '{action_json_str}'")
                    else:
                        # 3. Assume the raw action_block_text (or a part of it) is the JSON.
                        #    Only if it plausibly looks like JSON (starts with { or [).
                        first_bracket = action_block_text.find('[')
                        first_curly = action_block_text.find('{')
                        
                        start_char_pos = -1
                        if first_curly != -1 and first_bracket != -1:
                            start_char_pos = min(first_curly, first_bracket)
                        elif first_curly != -1:
                            start_char_pos = first_curly
                        elif first_bracket != -1:
                            start_char_pos = first_bracket
                        
                        if start_char_pos != -1:
                            # Attempt to isolate the JSON-like structure.
                            # This is a simplified approach; robustly finding the end of a JSON object/array
                            # without full parsing is tricky. We hope the LLM provides relatively clean output here.
                            potential_json_candidate = action_block_text[start_char_pos:]
                            # Simple check if it looks like a complete JSON object or array
                            # by checking start and corresponding end brackets
                            if (potential_json_candidate.startswith('{') and potential_json_candidate.rfind('}') > 0) or \
                               (potential_json_candidate.startswith('[') and potential_json_candidate.rfind(']') > 0):
                                # Try to find the balanced end bracket.
                                # This is still a heuristic. For truly robust, one might need a mini-parser.
                                open_brackets = 0
                                last_char_index = -1
                                expected_closing_bracket = '}' if potential_json_candidate.startswith('{') else ']'
                                opening_bracket = '{' if potential_json_candidate.startswith('{') else '['

                                for i, char in enumerate(potential_json_candidate):
                                    if char == opening_bracket:
                                        open_brackets += 1
                                    elif char == expected_closing_bracket:
                                        open_brackets -= 1
                                    if open_brackets == 0 and (char == expected_closing_bracket):
                                        last_char_index = i
                                        break 
                                
                                if last_char_index != -1:
                                    action_json_str = potential_json_candidate[:last_char_index+1].strip()
                                    logger.debug(f"Attempting to parse raw block content as JSON (heuristic): '{action_json_str}'")
                                else:
                                    logger.debug(f"Raw block content '{potential_json_candidate}' started with '{{' or '[', but couldn't find balanced end.")
                            else:
                                logger.debug(f"Raw block content segment '{potential_json_candidate}' does not look like a complete JSON object/array.")
                        else:
                            logger.debug("No '{{' or '[' found as a likely start of JSON in action block text.")

            if action_json_str:
                try:
                    action_data = json.loads(action_json_str)
                    if not isinstance(action_data, dict) or "action" not in action_data or "action_input" not in action_data:
                        raise OutputParserException(f"Parsed JSON is not a valid action object (e.g., missing 'action' or 'action_input' key, or is not a dict): {action_json_str}")
                    
                    parsed_action_successfully = True
                    return AgentAction(
                        tool=action_data["action"],
                        tool_input=action_data["action_input"],
                        log=text 
                    )
                except json.JSONDecodeError as e:
                    logger.error(f"CustomReActJsonOutputParser: JSONDecodeError for extracted action string: '{action_json_str}'. Error: {e}")
                except OutputParserException as e: # Catch specific re-raise
                    logger.error(f"CustomReActJsonOutputParser: OutputParserException during action processing: {e}")
                    raise e 
                except Exception as e: 
                    logger.error(f"CustomReActJsonOutputParser: Unexpected error processing action: '{action_json_str}'. Error: {e}")
            
            if not parsed_action_successfully:
                 error_message = f"Found 'Action:' header, but could not extract or parse a valid JSON action."
                 if action_json_str: 
                     error_message += f" Extracted candidate string was: '{action_json_str}' but failed validation/parsing."
                 elif action_block_text: 
                     error_message += f" Action block content was: <<<\\n{action_block_text}\\n>>>."
                 else: 
                     error_message += " Action block content was empty or not found after 'Action:' header."
                 error_message += f" Original text: ```{text}```"
                 raise OutputParserException(error_message)

        if final_answer_header_match:
            logger.debug(f"Found 'Final Answer:' header at index {final_answer_header_match.start()}. No valid Action was parsed prior.")
            final_answer_content = text[final_answer_header_match.end():].strip()
            return AgentFinish(return_values={"output": final_answer_content}, log=text)

        logger.warning(f"CustomReActJsonOutputParser: No 'Action:' (that could be successfully parsed) or 'Final Answer:' found in LLM output: {text}")
        raise OutputParserException(f"Could not parse LLM output: No successfully parsable 'Action:' or 'Final Answer:' found in text: ```{text}```")

    @property
    def _type(self) -> str:
        return "custom_react_json_output_parser"


class AlpyAgent:
    """Manages the AI agent's state and interactions using LangChain."""

    def __init__(self, rich_console: Optional[RichConsole] = None):
        self.mode = "agent"  
        logger.info(f"AlpyAgent initializing in '{self.mode}' mode.")

        try:
            prompts_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'prompts.yaml')
            with open(prompts_path, 'r') as f:
                prompts_config = yaml.safe_load(f)
            self.raw_system_prompt_agent = prompts_config.get('system_prompt_react_agent', "You are a helpful ReAct agent. You have access to tools.")
            self.raw_system_prompt_chat = prompts_config.get('system_prompt_chat_mode', "You are a helpful conversational assistant.")
            if not prompts_config.get('system_prompt_react_agent'): 
                logger.warning("system_prompt_react_agent not found in prompts.yaml, using default.")
            if not prompts_config.get('system_prompt_chat_mode'):
                logger.warning("system_prompt_chat_mode not found in prompts.yaml, using default.")
        except FileNotFoundError:
            logger.error(f"Prompts file not found at {prompts_path}. Using basic defaults.")
            self.raw_system_prompt_agent = "You are a helpful ReAct agent. You have access to tools."
            self.raw_system_prompt_chat = "You are a helpful conversational assistant."
        except Exception as e:
            logger.error(f"Error loading prompts: {e}. Using basic defaults.", exc_info=True)
            self.raw_system_prompt_agent = "You are a helpful ReAct agent. Error loading prompts."
            self.raw_system_prompt_chat = "You are a helpful conversational assistant. Error loading prompts."

        self.mcp_bash_tool = MCPBashExecutorTool(instance_name="AlpyBashTool")
        self.mcp_python_tool = MCPPythonExecutorTool(instance_name="AlpyPythonTool")
        self.mcp_filesystem_tool = MCPFileSystemTool(instance_name="AlpyFileSystemTool", allowed_dirs=["/"])
        self.mcp_brave_search_tool = MCPBraveSearchTool(instance_name="AlpyBraveSearchTool")
        self.mcp_media_display_tool = MCPMediaDisplayTool(instance_name="AlpyMediaDisplayTool")
        self.mcp_fetcher_web_tool = MCPFetcherWebTool(instance_name="AlpyFetcherWebTool")
        self.mcp_puppeteer_tool = MCPPuppeteerTool(instance_name="AlpyPuppeteerTool")

        self.tools: List[BaseTool] = [
            self.mcp_bash_tool,
            self.mcp_python_tool,
            self.mcp_filesystem_tool,
            self.mcp_brave_search_tool,
            self.mcp_media_display_tool,
            self.mcp_fetcher_web_tool,
            self.mcp_puppeteer_tool,
        ]
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        logger.info(f"Tools initialized: {[tool.name for tool in self.tools]}")

        final_react_prompt_str = (
            f"{self.raw_system_prompt_agent}\n\n"
            "TOOLS:\n------\n{tools}\n\n"
            "TOOL NAMES: {tool_names}\n\n"
            "CONVERSATION HISTORY:\n{chat_history}\n\n"
            "USER'S INPUT:\n------\n{input}\n\n"
            "SCRATCHPAD (Thought Process):\n---------------------------\n{agent_scratchpad}"
        )
        self.correct_prompt_for_agent = ChatPromptTemplate.from_template(final_react_prompt_str)

        self.output_parser = CustomReActJsonOutputParser()

        self.memory = ConversationBufferWindowMemory(
            k=config.AGENT_MEMORY_WINDOW_SIZE,
            memory_key="chat_history",
            input_key="input",
            output_key="output",
            return_messages=True
        )
        logger.info(f"ConversationBufferWindowMemory initialized with k={config.AGENT_MEMORY_WINDOW_SIZE}")

        self.chat_mode_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.raw_system_prompt_chat),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        logger.info("Chat mode prompt template initialized.")

        self.rich_streaming_callback_handler = RichStreamingCallbackHandler(console=rich_console)

        self.llm: Optional[ChatOpenAI] = None
        self.agent: Optional[Any] = None 
        self.agent_executor: Optional[AgentExecutor] = None
        self.current_llm_model_name: str = "" 
        self.active_provider: str = config.LLM_PROVIDER 

        try:
            self._create_llm_and_agent_components()
            logger.info(f"Initial LLM and agent components created successfully for provider '{self.active_provider}'.")
        except ValueError as e:
            logger.error(f"CRITICAL: Failed to initialize LLM/Agent components during AlpyAgent setup: {e}")
            logger.error("AlpyAgent may not function correctly, especially in 'agent' mode.")
        except Exception as e:
            logger.error(f"UNEXPECTED CRITICAL ERROR during initial LLM/Agent component setup: {e}", exc_info=True)
            logger.error("AlpyAgent is likely in a broken state.")

    def _create_llm_and_agent_components(self, model_name_override: Optional[str] = None):
        logger.info(f"Attempting to create/recreate LLM and agent components...")
        if model_name_override:
            logger.info(f"Model override provided: {model_name_override}")

        chosen_model_name = model_name_override
        if not chosen_model_name:
            if self.active_provider == 'local':
                chosen_model_name = config.ACTIVE_LOCAL_LLM_MODEL
                if not chosen_model_name or chosen_model_name == "default-local-model-not-set":
                    logger.error("No active local model configured in config.py and no override provided.")
                    raise ValueError("Active local model name is not properly configured in config.py.")
            elif self.active_provider == 'openrouter':
                chosen_model_name = config.ACTIVE_OPENROUTER_MODEL
                if not chosen_model_name or chosen_model_name == "default-openrouter-model-not-set":
                    logger.error("No active OpenRouter model configured in config.py and no override provided.")
                    raise ValueError("Active OpenRouter model name is not properly configured in config.py.")
            elif self.active_provider == 'google': # New case for Google
                chosen_model_name = config.ACTIVE_GOOGLE_MODEL
                if not chosen_model_name or chosen_model_name == "default-google-model-not-set": # Assuming you might add a similar placeholder
                    logger.error("No active Google model configured in config.py and no override provided.")
                    raise ValueError("Active Google model name is not properly configured in config.py.")
            else:
                logger.error(f"Unsupported LLM_PROVIDER: {self.active_provider} during component creation.")
                self.current_llm_model_name = "error-unknown-provider"
                self.llm, self.agent, self.agent_executor = None, None, None
                raise ValueError(f"Cannot create LLM components: Unsupported LLM_PROVIDER '{self.active_provider}'.")
        
        self.current_llm_model_name = chosen_model_name
        logger.info(f"Setting up LLM with model: '{self.current_llm_model_name}' for provider: '{self.active_provider}'")

        try:
            # Common parameters for OpenAI-compatible APIs
            openai_compatible_params = {
                "temperature": config.LLM_TEMPERATURE,
                "max_tokens": config.LLM_MAX_TOKENS,
                "top_p": config.LLM_TOP_P,
                "streaming": True,
                "model_name": self.current_llm_model_name,
                "stop": ["\nObservation:"] # Important for ReAct
            }

            if self.active_provider == 'local':
                openai_compatible_params["openai_api_base"] = config.LOCAL_LLM_API_BASE
                openai_compatible_params["openai_api_key"] = config.LOCAL_LLM_API_KEY
                logger.info(f"Local LLM params: API_BASE='{config.LOCAL_LLM_API_BASE}', Model='{self.current_llm_model_name}'")
                self.llm = ChatOpenAI(**openai_compatible_params)
            elif self.active_provider == 'openrouter':
                openai_compatible_params["openai_api_base"] = config.OPENROUTER_API_BASE
                openai_compatible_params["openai_api_key"] = config.OPENROUTER_API_KEY
                default_headers = {}
                if config.OPENROUTER_SITE_URL:
                    default_headers["HTTP-Referer"] = config.OPENROUTER_SITE_URL
                if hasattr(config, 'OPENROUTER_APP_NAME') and config.OPENROUTER_APP_NAME:
                    default_headers["X-Title"] = config.OPENROUTER_APP_NAME
                if default_headers:
                    openai_compatible_params["default_headers"] = default_headers
                logger.info(f"OpenRouter LLM params: API_BASE='{config.OPENROUTER_API_BASE}', Model='{self.current_llm_model_name}'")
                if config.OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_PLACEHOLDER": # Check placeholder
                    logger.warning("OpenRouter API key is a placeholder. Please update it.")
                self.llm = ChatOpenAI(**openai_compatible_params)
            elif self.active_provider == 'google': # New block for Google
                google_llm_params = {
                    "model": self.current_llm_model_name, # Google uses 'model'
                    "google_api_key": config.GOOGLE_API_KEY,
                    "temperature": config.LLM_TEMPERATURE,
                    "max_output_tokens": config.LLM_MAX_TOKENS, # Google uses 'max_output_tokens'
                    "top_p": config.LLM_TOP_P,
                    "top_k": config.LLM_TOP_K, # Google supports top_k
                    "streaming": True,
                    # For Google, `stop_sequences` is the parameter if needed, but Langchain might adapt `stop`.
                    # ReAct's "Observation:" stop might be problematic or handled differently.
                    # Test carefully. If issues, you might need to adjust prompts or remove stop for Google.
                    "stop": ["\nObservation:"] 
                }
                logger.info(f"Google LLM params: Model='{self.current_llm_model_name}'")
                if not config.GOOGLE_API_KEY or config.GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_PLACEHOLDER": # Check placeholder
                    logger.warning("Google API key is not set or is a placeholder. Please update it.")
                self.llm = ChatGoogleGenerativeAI(**google_llm_params)
            
            # Common setup for agent and agent_executor after self.llm is set
            logger.info(f"LangChain {self.llm.__class__.__name__} (re)initialized successfully for provider: {self.active_provider}, model: {self.current_llm_model_name}.")
            if hasattr(self.llm, 'stop'): # Check if the llm instance has stop attribute after init
                 logger.info(f"DEBUG: self.llm.stop configured to: {self.llm.stop}")

            if not all([hasattr(self, attr) and getattr(self,attr) is not None for attr in ['correct_prompt_for_agent', 'tools', 'output_parser', 'memory', 'rich_streaming_callback_handler']]):
                logger.error("One or more required attributes (prompt, tools, parser, memory, callback_handler) not set before agent creation.")
                raise ValueError("Prerequisite attributes for agent creation are missing or None.")

            self.agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=self.correct_prompt_for_agent,
                output_parser=self.output_parser
            )
            logger.info("ReAct agent (re)created successfully.")

            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                memory=self.memory,
                verbose=False, 
                handle_parsing_errors=True, 
                max_iterations=config.AGENT_MAX_ITERATIONS,
                callbacks=[self.rich_streaming_callback_handler]
            )
            logger.info("AgentExecutor (re)created successfully.")

        except AttributeError as e:
            logger.error(f"Config attribute missing during LLM/Agent (re)creation: {e}. Check src/config.py.")
            self.llm, self.agent, self.agent_executor = None, None, None
            self.current_llm_model_name = f"error-config-missing-{e}"
            raise ValueError(f"Failed to (re)initialize LLM/Agent due to missing config: {e}") from e
        except Exception as e:
            logger.error(f"Failed to (re)initialize LLM/Agent: {e}", exc_info=True)
            self.llm, self.agent, self.agent_executor = None, None, None
            self.current_llm_model_name = f"error-initialization-{e}"
            raise ValueError(f"Failed to (re)initialize LLM/Agent components: {e}") from e

    async def switch_llm_model(self, new_model_identifier: str) -> Dict[str, Any]:
        logger.info(f"Attempting to switch LLM model to: '{new_model_identifier}'")
        
        valid_models = []
        if self.active_provider == 'local':
            valid_models = config.AVAILABLE_LOCAL_MODELS
            if not valid_models:
                msg = f"Cannot switch model. No available local models defined in config.py."
                logger.error(msg)
                return {"success": False, "message": msg}
        elif self.active_provider == 'openrouter':
            # For OpenRouter, we often allow any model ID, so validation might be less strict
            # or rely on user knowing valid model IDs.
            # If you want to restrict to config.AVAILABLE_OPENROUTER_MODELS:
            # valid_models = config.AVAILABLE_OPENROUTER_MODELS
            # if not valid_models and new_model_identifier not in config.AVAILABLE_OPENROUTER_MODELS: # Example
            #     msg = f"Model '{new_model_identifier}' is not listed in AVAILABLE_OPENROUTER_MODELS."
            #     logger.warning(msg) # Or return error
            # else: pass
            pass # Currently allows any model string for OpenRouter
        elif self.active_provider == 'google': # New case for Google
            valid_models = config.AVAILABLE_GOOGLE_MODELS
            if not valid_models:
                msg = f"Cannot switch model. No available Google models defined in config.py."
                logger.error(msg)
                return {"success": False, "message": msg}
        else:
            msg = f"Cannot switch model. Unknown active_provider: '{self.active_provider}'"
            logger.error(msg)
            return {"success": False, "message": msg}

        # Perform validation if valid_models list was populated for the provider
        if valid_models and new_model_identifier not in valid_models:
            msg = f"Model '{new_model_identifier}' is not available for provider '{self.active_provider}'. Available models: {valid_models}"
            logger.warning(msg)
            return {"success": False, "message": msg}

        if new_model_identifier == self.current_llm_model_name:
            msg = f"Model '{new_model_identifier}' is already the active model for provider '{self.active_provider}'."
            logger.info(msg)
            return {"success": True, "message": msg}

        logger.info(f"Switching from '{self.current_llm_model_name}' to '{new_model_identifier}'.")
        previous_model_name = self.current_llm_model_name

        try:
            self._create_llm_and_agent_components(model_name_override=new_model_identifier)
            msg = f"Successfully switched model to: '{self.current_llm_model_name}' for provider '{self.active_provider}'."
            logger.info(msg)
            return {"success": True, "message": msg}
        except Exception as e:
            logger.error(f"Failed to switch model to '{new_model_identifier}': {e}. Attempting to restore previous model '{previous_model_name}'.", exc_info=True)
            try:
                self._create_llm_and_agent_components(model_name_override=previous_model_name)
                msg = (f"Failed to switch to '{new_model_identifier}' (Error: {e}). "
                    f"Successfully reverted to previous model: '{self.current_llm_model_name}'.")
                logger.info(f"Successfully reverted to model: '{self.current_llm_model_name}' after failed switch.")
            except Exception as restore_e:
                msg = (f"CRITICAL: Failed to switch to '{new_model_identifier}' (Error: {e}) AND "
                    f"failed to restore previous model '{previous_model_name}' (RestoreError: {restore_e}). "
                    f"Agent might be in an unstable state.")
                logger.critical(msg, exc_info=True)
            return {"success": False, "message": msg}

    async def switch_llm_provider(self, new_provider: str) -> Dict[str, Any]:
        logger.info(f"Attempting to switch LLM provider to: '{new_provider}'")

        if new_provider not in ['local', 'openrouter', 'google']: # Add 'google'
            msg = f"Invalid provider '{new_provider}'. Must be 'local', 'openrouter', or 'google'."
            logger.warning(msg)
            return {"success": False, "message": msg}

        if new_provider == self.active_provider:
            msg = f"Provider '{new_provider}' is already active."
            logger.info(msg)
            return {"success": True, "message": msg}

        logger.info(f"Switching provider from '{self.active_provider}' to '{new_provider}'.")
        previous_provider = self.active_provider
        previous_model = self.current_llm_model_name
        self.active_provider = new_provider

        try:
            # _create_llm_and_agent_components will now pick the default model for the new provider
            self._create_llm_and_agent_components() 
            msg = f"Successfully switched provider to: '{self.active_provider}'. Active model: '{self.current_llm_model_name}'."
            logger.info(msg)
            return {"success": True, "message": msg}
        except Exception as e:
            logger.error(f"Failed to switch provider to '{new_provider}': {e}. Attempting to restore previous provider '{previous_provider}' and model '{previous_model}'.", exc_info=True)
            self.active_provider = previous_provider 
            try:
                self._create_llm_and_agent_components(model_name_override=previous_model)
                msg = (f"Failed to switch to provider '{new_provider}' (Error: {e}). "
                    f"Successfully reverted to previous provider: '{self.active_provider}' and model: '{self.current_llm_model_name}'.")
                logger.info(f"Successfully reverted to provider: '{self.active_provider}' and model: '{self.current_llm_model_name}' after failed switch.")
            except Exception as restore_e:
                msg = (f"CRITICAL: Failed to switch to provider '{new_provider}' (Error: {e}) AND "
                    f"failed to restore previous provider '{previous_provider}' and model '{previous_model}' (RestoreError: {restore_e}). "
                    f"Agent might be in an unstable state.")
                logger.critical(msg, exc_info=True)
            return {"success": False, "message": msg}

    async def set_mode(self, new_mode: str):
        if new_mode in ["agent", "chat"]:
            self.mode = new_mode
            logger.info(f"Alpy mode switched to: {self.mode}")
            return f"Mode switched to {self.mode}."
        else:
            logger.warning(f"Attempted to switch to invalid mode: {new_mode}")
            return f"Invalid mode '{new_mode}'. Valid modes are 'agent' or 'chat'."

    async def get_response(self, user_input: str):
        if self.mode == "agent":
            logger.info(f"Processing user_input in AGENT mode: {user_input[:100]}...")
            
            # self.rich_streaming_callback_handler is already initialized in __init__
            # and passed to the AgentExecutor's callbacks list.
            # We don't need to instantiate it here or pass it explicitly to ainvoke if
            # it's already part of the agent_executor's default callbacks.
            # However, if you want to ensure it's used or add specific callbacks per invoke:
            # callbacks_for_this_run = [self.rich_streaming_callback_handler]
            # For simplicity, assuming it's already in self.agent_executor.callbacks

            try:
                response_data = await self.agent_executor.ainvoke(
                    {"input": user_input}
                    # If you want to be explicit or override default callbacks:
                    # config={"callbacks": [self.rich_streaming_callback_handler]} 
                    # Otherwise, if it's already in agent_executor.callbacks, this is not strictly needed.
                    # For clarity and ensuring it's used, let's include it:
                    , config={"callbacks": [self.rich_streaming_callback_handler]}
                )
                
                final_answer_from_executor = response_data.get('output', "")
                logger.info(f"Agent execution finished. Final answer from executor: '{final_answer_from_executor[:100]}...'")
                
                # The RichStreamingCallbackHandler has already printed thoughts/actions/observations.
                # We only need to return the final answer.
                return final_answer_from_executor.strip() if final_answer_from_executor else "Agent processed the request but didn't produce a final answer."

            except asyncio.CancelledError:
                logger.info("Agent get_response operation was cancelled.")
                raise # Re-raise to allow the caller to know it was cancelled

            except OutputParserException as e: 
                logger.error(f"Output parsing error during agent execution: {e}")
                error_message_str = str(e)
                # The RichStreamingCallbackHandler would have printed thoughts up to the error.
                # We try to extract a final answer if the parser choked on it.
                final_answer_in_error = re.search(FINAL_ANSWER_PATTERN, error_message_str, re.DOTALL)
                if final_answer_in_error:
                    logger.info("Extracted Final Answer from parsing error message.")
                    extracted_answer = final_answer_in_error.group(1).strip()
                    # It's tricky to display this well, main.py expects just the final answer.
                    # The error itself should have been printed by the callback handler or logged.
                    return f"(Note: Parsing error occurred, but attempting to provide extracted answer)\nFinal Answer: {extracted_answer}"
                return "I encountered an issue parsing the response. Check logs for details. The thought process should have been displayed above."
            
            except Exception as e:
                logger.error(f"Error during ReAct AgentExecutor invocation: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Thoughts/actions up to this point should have been printed by the handler.
                return "An unexpected error occurred in agent mode. Check logs for details. The thought process leading to the error should have been displayed above."
        
        elif self.mode == "chat":
            logger.info(f"Processing user_input in CHAT mode: {user_input[:100]}...")
            try:
                chat_history_messages = self.memory.chat_memory.messages
                chat_chain = self.chat_mode_prompt | self.llm
                
                response_message = await chat_chain.ainvoke({
                    "input": user_input,
                    "chat_history": chat_history_messages 
                })
                
                response_content = response_message.content.strip()
                
                self.memory.chat_memory.add_user_message(user_input)
                self.memory.chat_memory.add_ai_message(response_content)

                logger.info(f"Chat mode response: {response_content[:100]}...")
                return response_content

            except asyncio.CancelledError: # Also handle for chat mode if desired
                logger.info("Chat get_response operation was cancelled.")
                raise

            except Exception as e:
                logger.error(f"Error during CHAT mode LLM invocation: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return f"An unexpected error occurred in chat mode: {e}"
        else:
            logger.error(f"Invalid mode '{self.mode}' in get_response.")
            return "Error: Alpy is in an invalid mode."

    async def get_history(self) -> list:
        return self.memory.chat_memory.messages

    async def close(self):
        logger.info("Closing AlpyAgent and its tools...")
        
        tools_to_close = {
            "Bash Tool": self.mcp_bash_tool,
            "Python Tool": self.mcp_python_tool,
            "Filesystem Tool": self.mcp_filesystem_tool,
            "Brave Search Tool": self.mcp_brave_search_tool,
            "Media Display Tool": self.mcp_media_display_tool,
            "Fetcher Web Tool": self.mcp_fetcher_web_tool,
            "Puppeteer Tool": self.mcp_puppeteer_tool,
        }

        for tool_name, tool_instance in tools_to_close.items():
            if hasattr(tool_instance, 'close') and callable(getattr(tool_instance, 'close')):
                try:
                    logger.info(f"Closing {tool_name}...")
                    await tool_instance.close() 
                    logger.info(f"{tool_name} closed successfully.")
                except Exception as e:
                    logger.error(f"Error closing {tool_name}: {e}", exc_info=True)
            else:
                logger.warning(f"{tool_name} does not have a callable close() method.")
        
        logger.info("AlpyAgent tools closed.")

async def main_agent_test(): 
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    agent = None 
    try:
        agent = AlpyAgent()
        print("AlpyAgent (LangChain) initialized. Enter 'quit' to exit.")
        while True:
            try:
                user_input = input("> ") 
                if user_input.lower() == 'quit':
                    break
                response = await agent.get_response(user_input) 
                print(f"Alpy: {response}")
            except EOFError:
                print("\nInput stream closed. Exiting.")
                break
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Fatal Error initializing or running Agent: {e}")
                print(f"Traceback: {traceback.format_exc()}")
    finally:
        if agent:
            await agent.close() 

if  __name__ == '__main__':
    import asyncio
    asyncio.run(main_agent_test()) 