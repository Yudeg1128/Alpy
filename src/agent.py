import logging
import yaml
import re
import os
import json
import traceback
from typing import Dict, Any, List, Union

import httpx
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain.agents.agent import AgentOutputParser
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

import langchain

from . import config
from .tools.bash_tool import bash_tool
from .tools.python_tool import python_tool

logger = logging.getLogger(__name__)

ACTION_PATTERN = r"Action\s*:\s*```json\s*(.*?)\s*```"
FINAL_ANSWER_PATTERN = r"Final Answer\s*:\s*(.*)"
THINK_PATTERN = r"^\s*<think>.*?</think>\s*"

class RichStreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.trace: List[str] = []
        self._current_llm_output_parts: List[str] = []

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        self._current_llm_output_parts = []

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self._current_llm_output_parts.append(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        full_llm_response = "".join(self._current_llm_output_parts).strip()
        if full_llm_response:
            cleaned_llm_response = re.sub(THINK_PATTERN, "", full_llm_response, flags=re.DOTALL | re.IGNORECASE).strip()
            if cleaned_llm_response:
                self.trace.append(cleaned_llm_response)
        self._current_llm_output_parts = []

    def on_tool_end(
        self,
        output: str,
        color: Union[str, None] = None,
        observation_prefix: Union[str, None] = None,
        llm_prefix: Union[str, None] = None,
        **kwargs: Any,
    ) -> None:
        self.trace.append(f"Observation: {output.strip()}")

    def get_full_trace(self) -> str:
        return "\n".join(part for part in self.trace if part.strip()).strip()

class CustomReActJsonOutputParser(AgentOutputParser):
    def parse(self, text: str) -> AgentAction | AgentFinish:
        cleaned_text = re.sub(THINK_PATTERN, "", text.strip(), flags=re.DOTALL | re.IGNORECASE).strip()

        final_answer_match = re.search(FINAL_ANSWER_PATTERN, cleaned_text, re.DOTALL)
        if final_answer_match:
            return AgentFinish({"output": final_answer_match.group(1).strip()}, log=cleaned_text)

        action_match = re.search(ACTION_PATTERN, cleaned_text, re.DOTALL)
        if action_match:
            try:
                action_json_str = action_match.group(1).strip()
                action_data = json.loads(action_json_str)
                tool_name = action_data.get("action")
                tool_input = action_data.get("action_input", "")
                if not tool_name:
                    raise ValueError("Missing 'action' in JSON blob")
                
                return AgentAction(tool=tool_name, tool_input=tool_input, log=cleaned_text)
            except json.JSONDecodeError as e:
                raise OutputParserException(f"Could not parse action JSON: {action_json_str}. Error: {e}") from e
            except ValueError as e:
                 raise OutputParserException(f"Invalid action JSON structure: {action_json_str}. Error: {e}") from e
        
        raise OutputParserException(
            f"Could not parse LLM output: `{cleaned_text}`. "
            f"Expected a JSON action block or a Final Answer."
        )
    
    @property
    def _type(self) -> str:
        return "custom_react_json_output_parser"

class AlpyAgent:
    """Manages the AI agent's state and interactions using LangChain."""

    def __init__(self):
        logger.info("Initializing AlpyAgent with LangChain ReAct tools...")
        raw_system_prompt = ""
        try:
            prompts_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'prompts.yaml')
            with open(prompts_path, 'r') as f:
                prompts_config = yaml.safe_load(f)
            raw_system_prompt = prompts_config.get('system_prompt_react_agent')
            if not raw_system_prompt:
                logger.warning("system_prompt_react_agent not found in prompts.yaml. Using a basic default for storage.")
                raw_system_prompt = "You are a helpful assistant. You have access to tools. Respond using ReAct format with JSON actions."
            logger.info("Raw ReAct system prompt content loaded from YAML (will not be directly used by agent if prompt=None).")
        except (FileNotFoundError, yaml.YAMLError) as e:
            logger.error(f"Error loading prompts: {e}. Using a basic default ReAct prompt string for storage.")
            raw_system_prompt = "You are a helpful assistant. You have access to tools. Respond using ReAct format with JSON actions."

        try:
            custom_client = httpx.Client(
                timeout=httpx.Timeout(60.0, connect=10.0, read=60.0, write=10.0)
            )
            self.llm = ChatOpenAI(
                 openai_api_base=config.LLM_API_BASE,
                 openai_api_key=config.LLM_API_KEY,
                 model_name=config.LLM_MODEL_NAME,
                 temperature=config.LLM_TEMPERATURE,
                 max_tokens=config.LLM_MAX_TOKENS,
                 top_p=config.LLM_TOP_P,
                 streaming=False, 
                 http_client=custom_client,
             )
            logger.info(f"LangChain ChatOpenAI initialized for base_url: {config.LLM_API_BASE} with custom httpx client.")
        except AttributeError as e:
             logger.error(f"Config attribute missing: {e}. Please check src/config.py.")
             raise ValueError("Failed to initialize ChatOpenAI due to missing config.") from e
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI: {e}")
            raise ValueError("Failed to initialize ChatOpenAI.") from e

        self.tools = [bash_tool, python_tool]
        logger.info(f"Tools initialized: {[tool.name for tool in self.tools]}")

        final_react_prompt_str = (
            f"{raw_system_prompt}\n\n"
            "TOOLS:\n------\n{tools}\n\n"
            "TOOL NAMES: {tool_names}\n\n"
            "CONVERSATION HISTORY:\n{chat_history}\n\n"
            "USER'S INPUT:\n------\n{input}\n\n"
            "SCRATCHPAD (Thought/Action/Observation sequence):\n------\n{agent_scratchpad}"
        )

        correct_prompt_for_agent = ChatPromptTemplate.from_template(final_react_prompt_str)

        output_parser = CustomReActJsonOutputParser()

        try:
            self.agent = create_react_agent(
                llm=self.llm, 
                tools=self.tools, 
                prompt=correct_prompt_for_agent, 
                output_parser=output_parser 
            )
            logger.info("ReAct agent created successfully with custom output parser and explicit ReAct prompt.")
        except Exception as e:
            logger.error(f"Failed to create ReAct agent: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ValueError("Failed to create ReAct agent.") from e

        self.memory = ConversationBufferWindowMemory(
            k=config.AGENT_MEMORY_WINDOW_SIZE,
            memory_key="chat_history", 
            return_messages=True 
        )
        logger.info(f"ConversationBufferWindowMemory initialized with k={config.AGENT_MEMORY_WINDOW_SIZE}")

        self.agent_executor = AgentExecutor(
            agent=self.agent, 
            tools=self.tools, 
            memory=self.memory,
            verbose=False, 
            max_iterations=config.AGENT_MAX_ITERATIONS, 
            handle_parsing_errors=True 
        )
        logger.info("AgentExecutor created successfully with ReAct agent.")

    def get_response(self, user_input: str) -> str:
        logger.info(f"Processing user_input with ReAct AgentExecutor: {user_input[:100]}...")
        
        streaming_callback = RichStreamingCallbackHandler()
        full_trace_for_display = ""

        try:
            response_data = self.agent_executor.invoke(
                {"input": user_input},
                config={"callbacks": [streaming_callback]}
            )
            
            full_trace_for_display = streaming_callback.get_full_trace()
            final_answer_from_executor = response_data.get('output', "")

            if not full_trace_for_display.strip() and final_answer_from_executor:
                 full_trace_for_display = f"Final Answer: {final_answer_from_executor}"
            elif final_answer_from_executor and final_answer_from_executor not in full_trace_for_display:
                 if not full_trace_for_display.strip().endswith(final_answer_from_executor.strip()):
                    full_trace_for_display = f"{full_trace_for_display.strip()}\nFinal Answer: {final_answer_from_executor.strip()}".strip()
            
            logger.info(f"Agent trace captured. Final text length: {len(full_trace_for_display)}")
            return full_trace_for_display.strip() if full_trace_for_display.strip() else "Sorry, I couldn't process that fully."

        except OutputParserException as e: 
            logger.error(f"Output parsing error during agent execution: {e}")
            error_message_str = str(e)
            final_answer_in_error = re.search(FINAL_ANSWER_PATTERN, error_message_str, re.DOTALL)
            partial_trace = streaming_callback.get_full_trace()
            if final_answer_in_error:
                logger.info("Extracted Final Answer from parsing error message.")
                extracted_answer = final_answer_in_error.group(1).strip()
                return f"{partial_trace}\nFinal Answer: {extracted_answer}".strip()
            return f"{partial_trace}\nI encountered an issue parsing the response. Details: {e}".strip()
        except Exception as e:
            logger.error(f"Error during ReAct AgentExecutor invocation: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            partial_trace = streaming_callback.get_full_trace()
            return f"{partial_trace}\nAn unexpected error occurred: {e}".strip()

    def get_history(self) -> list:
        return self.memory.chat_memory.messages

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    try:
        agent = AlpyAgent() 
        print("AlpyAgent (LangChain) initialized. Enter 'quit' to exit.")
        while True:
            try:
                user_input = input("> ")
                if user_input.lower() == 'quit':
                    break
                response = agent.get_response(user_input)
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
