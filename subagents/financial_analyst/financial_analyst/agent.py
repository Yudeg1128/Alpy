import asyncio
from .memory import FinancialAnalystMemory
from .tasks import handle_task
from tools.pdf_to_image_tool import PDFToImageTool
from tools.vector_ops_tool import VectorOpsTool
from tools.image_embedder_tool import ImageEmbedderTool
from tools.image_parser_tool import ImageParserTool
import os
import yaml
from . import config

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ConversationBufferWindowMemory

from rich.console import Console as RichConsole
from rich.panel import Panel as RichPanel
from rich.markdown import Markdown as RichMarkdown
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser as CustomReActJsonOutputParser
import re
import logging
from langchain_core.outputs import LLMResult
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
ACTION_PATTERN = r"Action\s*:\s*```json\s*([\s\S]+?)\s*```"

class RichStreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, console: Optional[RichConsole] = None) -> None:
        super().__init__()
        self.trace: List[str] = []
        self._current_llm_output_parts: List[str] = []
        self.console = console
        self.action_count = 0

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        self._current_llm_output_parts = []

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self._current_llm_output_parts.append(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        full_llm_response = "".join(self._current_llm_output_parts).strip()
        self._current_llm_output_parts = []
        if not full_llm_response:
            return
        self.trace.append(full_llm_response)
        logger.debug(f"RichStreamingCallbackHandler.on_llm_end received: {full_llm_response[:200]}...")
        title = "💡 LLM Thought/Output"
        panel_style = "green"
        action_match = re.search(ACTION_PATTERN, full_llm_response, re.DOTALL | re.IGNORECASE)
        if action_match:
            title = "⚙️ LLM Action Proposal"
            panel_style = "cyan"
        if self.console:
            self.console.print(RichPanel(full_llm_response, title=title, border_style=panel_style))

class FinancialAnalystAgent:
    def __init__(self, memory_path='memory.json', rich_console=None):
        self.memory = FinancialAnalystMemory(memory_path)
        self.task_queue = []
        self.running_tasks = {}
        self._last_truncated_thought_process = None

        # --- TOOLS ---
        self.pdf_to_image_tool = PDFToImageTool()
        self.vector_ops_tool = VectorOpsTool()
        self.image_embedder_tool = ImageEmbedderTool()
        self.image_parser_tool = ImageParserTool()
        self.tools = [self.pdf_to_image_tool, self.vector_ops_tool, self.image_embedder_tool, self.image_parser_tool]
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        logger.info(f"Tools initialized: {[tool.name for tool in self.tools]}")

        # --- PROMPT LOAD ---
        try:
            prompts_path = os.path.join(os.path.dirname(__file__), 'prompts.yaml')
            with open(prompts_path, 'r') as f:
                prompts_config = yaml.safe_load(f)
            self.raw_system_prompt = prompts_config.get('system_prompt_react_agent_financial_analyst', "You are a financial analyst agent. Use tools when needed.")
        except FileNotFoundError:
            logger.error(f"Prompts file not found at {prompts_path}. Using basic default.")
            self.raw_system_prompt = "You are a financial analyst agent. Use tools when needed."
        except Exception as e:
            logger.error(f"Error loading prompts: {e}. Using basic default.", exc_info=True)
            self.raw_system_prompt = "You are a financial analyst agent. Error loading prompts."

        final_react_prompt_str = (
            f"{self.raw_system_prompt}\n\n"
            "TOOLS:\n------\n{tools}\n\n"
            "TOOL NAMES: {tool_names}\n\n"
            "CONVERSATION HISTORY:\n{chat_history}\n\n"
            "USER'S INPUT:\n------\n{input}\n\n"
            "SCRATCHPAD (Thought Process):\n---------------------------\n{agent_scratchpad}"
        )
        self.correct_prompt_for_agent = ChatPromptTemplate.from_template(final_react_prompt_str)

        # --- MEMORY ---
        self.langchain_memory = ConversationBufferWindowMemory(
            k=6, # match AlpyAgent or config
            memory_key="chat_history",
            input_key="input",
            output_key="output",
            return_messages=True
        )
        logger.info(f"ConversationBufferWindowMemory initialized with k={config.AGENT_MEMORY_WINDOW_SIZE}")

        self.rich_streaming_callback_handler = RichStreamingCallbackHandler(console=rich_console)
        self.output_parser = CustomReActJsonOutputParser()

        # Initialize LLM and agent
        try:
            google_llm_params = {
                "model": config.GOOGLE_MODEL,
                "google_api_key": config.GOOGLE_API_KEY,
                "temperature": config.LLM_TEMPERATURE,
                "max_output_tokens": config.LLM_MAX_TOKENS,
                "top_p": config.LLM_TOP_P,
            }
            logger.info(f"Initializing Google LLM with model={config.GOOGLE_MODEL}")
            if not config.GOOGLE_API_KEY or config.GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_PLACEHOLDER":
                logger.warning("Google API key is not set or is a placeholder. Please update it.")
            
            self.llm = ChatGoogleGenerativeAI(**google_llm_params)
            logger.info("LangChain ChatGoogleGenerativeAI initialized successfully")

            if not all([hasattr(self, attr) and getattr(self,attr) is not None for attr in ['correct_prompt_for_agent', 'tools', 'output_parser', 'langchain_memory', 'rich_streaming_callback_handler']]):
                logger.error("One or more required attributes (prompt, tools, parser, memory, callback_handler) not set before agent creation.")
                raise ValueError("Prerequisite attributes for agent creation are missing or None.")

            self.agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=self.correct_prompt_for_agent,
                output_parser=self.output_parser
            )
            logger.info("ReAct agent created successfully.")

            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                memory=self.langchain_memory,
                verbose=False,
                handle_parsing_errors=True,
                max_iterations=config.AGENT_MAX_ITERATIONS,
                callbacks=[self.rich_streaming_callback_handler],
                return_intermediate_steps=True
            )
            logger.info("AgentExecutor created successfully.")

        except AttributeError as e:
            logger.error(f"Config attribute missing during LLM/Agent creation: {e}. Check config.py.")
            self.llm, self.agent, self.agent_executor = None, None, None
            raise ValueError(f"Failed to initialize LLM/Agent due to missing config: {e}") from e
        except Exception as e:
            logger.error(f"Failed to initialize LLM/Agent: {e}", exc_info=True)
            self.llm, self.agent, self.agent_executor = None, None, None
            raise ValueError(f"Failed to initialize LLM/Agent components: {e}") from e

    async def delegate_task(self, task_data):
        task_id = self.memory.add_task(task_data)
        coro = self._run_task(task_id, task_data)
        task = asyncio.create_task(coro)
        self.running_tasks[task_id] = task
        return task_id

    async def _run_task(self, task_id, task_data):
        result = await handle_task(task_data)
        self.memory.complete_task(task_id, result)
        del self.running_tasks[task_id]

    def get_task_report(self, task_id):
        return self.memory.get_task_report(task_id)

    def list_tasks(self):
        return self.memory.list_tasks()

    async def cleanup(self):
        """Clean up resources when shutting down."""
        logger.info("Cleaning up agent resources...")
        for tool in self.tools:
            try:
                await tool.close()
                logger.info(f"Closed tool: {tool.name}")
            except Exception as e:
                logger.error(f"Error closing tool {tool.name}: {e}")
        logger.info("Agent cleanup complete")

async def main():
    print("FinancialAnalystAgent (LLM+tools) started. Type 'quit' to exit.")
    agent = FinancialAnalystAgent()
    chat_history = []
    while True:
        try:
            user_input = input("Enter your financial analysis request (or 'quit'): ")
            if user_input.lower() == 'quit':
                await agent.cleanup()
                break

            result = await agent.agent_executor.ainvoke({
                "input": user_input,
                "chat_history": chat_history
            })

            if isinstance(result, dict) and 'output' in result:
                output = result['output']
                print(f"Agent: {output}")
                chat_history.append(HumanMessage(content=user_input))
                chat_history.append(AIMessage(content=output))
            else:
                logger.error(f"Unexpected result format: {result}")
                print("Agent encountered an error. Please try again.")

        except Exception as e:
            logger.error(f"Error during agent execution: {e}", exc_info=True)
            print("Agent encountered an error. Please try again.")
            continue

if __name__ == "__main__":
    asyncio.run(main())
