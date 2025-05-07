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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, HumanMessage
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain.agents.agent import AgentOutputParser
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

import langchain

from . import config
from .tools.mcp_bash_tool import MCPBashExecutorTool
from .tools.mcp_python_tool import MCPPythonExecutorTool
from .tools.mcp_filesystem_tool import MCPFileSystemTool

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
        self.mode = "agent"  # Default mode
        logger.info(f"AlpyAgent initialized in '{self.mode}' mode.")

        try:
            # Load prompts from YAML
            prompts_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'prompts.yaml')
            with open(prompts_path, 'r') as f:
                prompts_config = yaml.safe_load(f)
            
            self.raw_system_prompt_agent = prompts_config.get('system_prompt_react_agent')
            if not self.raw_system_prompt_agent:
                logger.warning("system_prompt_react_agent not found in prompts.yaml. Using a basic default.")
                self.raw_system_prompt_agent = "You are a helpful ReAct agent. You have access to tools."
            
            self.raw_system_prompt_chat = prompts_config.get('system_prompt_chat_mode')
            if not self.raw_system_prompt_chat:
                logger.warning("system_prompt_chat_mode not found in prompts.yaml. Using a basic default.")
                self.raw_system_prompt_chat = "You are a helpful conversational assistant."

        except FileNotFoundError:
            logger.error(f"Prompts file not found at {prompts_path}")
            # Fallback to basic prompts if file not found
            self.raw_system_prompt_agent = "You are a helpful ReAct agent. You have access to tools."
            self.raw_system_prompt_chat = "You are a helpful conversational assistant."
        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
            self.raw_system_prompt_agent = "You are a helpful ReAct agent. You have access to tools. Error loading prompts."
            self.raw_system_prompt_chat = "You are a helpful conversational assistant. Error loading prompts."

        # Instantiate new tools and store them as instance attributes
        self.mcp_bash_tool = MCPBashExecutorTool(instance_name="AlpyBashTool")
        self.mcp_python_tool = MCPPythonExecutorTool(instance_name="AlpyPythonTool")
        self.mcp_filesystem_tool = MCPFileSystemTool(
            instance_name="AlpyFileSystemTool",
            allowed_dirs=["/"] # Defaulting to current working directory
        )

        self.tools: List[BaseTool] = [self.mcp_bash_tool, self.mcp_python_tool, self.mcp_filesystem_tool]
        self.tools_by_name = {tool.name: tool for tool in self.tools}

        logger.info(f"Tools initialized: {[tool.name for tool in self.tools]}")

        # Initialize LLM and tools after loading prompts
        try:
            custom_client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=10.0, read=60.0, write=10.0)
            )
            self.llm = ChatOpenAI(
                 openai_api_base=config.LLM_API_BASE,
                 openai_api_key=config.LLM_API_KEY,
                 model_name=config.LLM_MODEL_NAME,
                 temperature=config.LLM_TEMPERATURE,
                 max_tokens=config.LLM_MAX_TOKENS,
                 top_p=config.LLM_TOP_P,
                 streaming=False
                 # http_client=custom_client, # REMOVE THIS LINE
             )
            logger.info(f"LangChain ChatOpenAI initialized for base_url: {config.LLM_API_BASE}.") # Removed "with custom httpx client"
        except AttributeError as e:
             logger.error(f"Config attribute missing: {e}. Please check src/config.py.")
             raise ValueError("Failed to initialize ChatOpenAI due to missing config.") from e
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI: {e}")
            raise ValueError("Failed to initialize ChatOpenAI.") from e

        # -- Agent Mode Setup --
        # Prompt for ReAct agent
        final_react_prompt_str = (
            f"{self.raw_system_prompt_agent}\n\n"
            "TOOLS:\n------\n{tools}\n\n"
            "TOOL NAMES: {tool_names}\n\n"
            "CONVERSATION HISTORY:\n{chat_history}\n\n"
            "USER'S INPUT:\n------\n{input}\n\n"
            "SCRATCHPAD (Thought Process):\n---------------------------\n{agent_scratchpad}"
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
            logger.error(f"Error creating ReAct agent: {e}")
            logger.error(traceback.format_exc())
            raise ValueError("Failed to create ReAct agent.") from e

        self.memory = ConversationBufferWindowMemory(
            k=config.AGENT_MEMORY_WINDOW_SIZE,
            memory_key="chat_history",
            input_key="input",
            output_key="output", # Langchain expects this for certain memory types
            return_messages=True # Important for ReAct prompt
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

        # -- Chat Mode Setup --
        # Simpler prompt for chat mode
        # MEMORY: Use tuples for from_messages for variable substitution
        self.chat_mode_prompt = ChatPromptTemplate.from_messages([
            ("system", self.raw_system_prompt_chat),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        logger.info("Chat mode prompt template created.")

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
            
            streaming_callback = RichStreamingCallbackHandler()
            full_trace_for_display = ""

            try:
                response_data = await self.agent_executor.ainvoke(
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
                return full_trace_for_display.strip() if full_trace_for_display.strip() else "Sorry, I couldn't process that fully (agent mode)."
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
                return f"{partial_trace}\nAn unexpected error occurred in agent mode: {e}".strip()
        
        elif self.mode == "chat":
            logger.info(f"Processing user_input in CHAT mode: {user_input[:100]}...")
            try:
                # Get chat history from memory
                chat_history_messages = self.memory.chat_memory.messages
                # The memory provides AIMessage, HumanMessage. Ensure they're passed correctly.
                # chat_mode_prompt expects a list of BaseMessage or tuples.

                # Construct the chain for chat mode
                chat_chain = self.chat_mode_prompt | self.llm
                
                response_message = await chat_chain.ainvoke({
                    "input": user_input,
                    "chat_history": chat_history_messages # Pass the actual BaseMessage objects
                })
                
                # The response_message from ChatOpenAI is an AIMessage content string
                response_content = response_message.content.strip()
                
                # Manually add interaction to memory for chat mode
                # self.memory.save_context({"input": user_input}, {"output": response_content}) # This is for AgentExecutor memory.
                # For direct memory update with BaseMessages:
                self.memory.chat_memory.add_user_message(user_input)
                self.memory.chat_memory.add_ai_message(response_content)

                logger.info(f"Chat mode response: {response_content[:100]}...")
                return response_content
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
        """Closes all managed tools and cleans up resources."""
        logger.info("Closing AlpyAgent and its tools...")
        
        tools_to_close = {
            "Bash Tool": self.mcp_bash_tool,
            "Python Tool": self.mcp_python_tool,
            "Filesystem Tool": self.mcp_filesystem_tool,
        }

        for tool_name, tool_instance in tools_to_close.items():
            if hasattr(tool_instance, 'close') and callable(getattr(tool_instance, 'close')):
                try:
                    logger.info(f"Closing {tool_name}...")
                    await tool_instance.close() # Added await
                    logger.info(f"{tool_name} closed successfully.")
                except Exception as e:
                    logger.error(f"Error closing {tool_name}: {e}", exc_info=True)
            else:
                logger.warning(f"{tool_name} does not have a callable close() method.")
        
        logger.info("AlpyAgent tools closed.")

async def main_agent_test(): # Created async main for testing
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    agent = None # Initialize agent to None for finally block
    try:
        agent = AlpyAgent() 
        print("AlpyAgent (LangChain) initialized. Enter 'quit' to exit.")
        while True:
            try:
                user_input = input("> ") # input() is blocking, will run in default executor
                if user_input.lower() == 'quit':
                    break
                response = await agent.get_response(user_input) # Added await
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
            await agent.close() # Added await

if __name__ == '__main__':
    import asyncio
    asyncio.run(main_agent_test()) # Changed to run async main
