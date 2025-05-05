import logging
import yaml
import re
import traceback
from typing import Dict, Any

import openai
import httpx
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from . import config

logger = logging.getLogger(__name__)

class AlpyAgent:
    """Manages the AI agent's state and interactions using LangChain."""

    def __init__(self):
        """Initializes the AlpyAgent."""
        logger.info("Initializing AlpyAgent with LangChain...")
        # Load prompts
        try:
            # Ensure the path is correct relative to where the script is run from
            # If running 'python -m src.main' from Alpy root, 'prompts/...' is correct.
            with open("prompts/prompts.yaml", 'r') as f:
                prompts = yaml.safe_load(f)
            self.system_prompt = prompts['system_prompt']
            logger.debug("Prompts loaded successfully.")
        except FileNotFoundError:
             logger.error("prompts/prompts.yaml not found. Using default system prompt.")
             self.system_prompt = "You are a helpful assistant."
        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
            self.system_prompt = "You are a helpful assistant." # Fallback

        # --- LangChain LLM Initialization --- 
        try:
            # --- Configure httpx Client --- 
            # Explicitly create an httpx client with longer timeouts
            custom_client = httpx.Client(
                timeout=httpx.Timeout(60.0, connect=10.0, read=60.0, write=10.0)
                # Adjust timeouts as needed
            )

            # --- Restore ChatOpenAI with custom client and parameters ---
            self.llm = ChatOpenAI(
                 openai_api_base=config.LLM_API_BASE,
                 openai_api_key=config.LLM_API_KEY, # Pass the dummy key
                 model_name=config.LLM_MODEL_NAME, # Model identifier
                 temperature=config.LLM_TEMPERATURE,
                 max_tokens=config.LLM_MAX_TOKENS,
                 top_p=config.LLM_TOP_P,
                 model_kwargs={ # Non-standard OpenAI params
                 },
                 streaming=False,
                 http_client=custom_client # Pass the custom client
             )

            logger.info(f"LangChain ChatOpenAI initialized for base_url: {config.LLM_API_BASE} with custom httpx client.")

        except AttributeError as e:
             logger.error(f"Config attribute missing: {e}. Please check src/config.py.")
             raise ValueError("Failed to initialize ChatOpenAI due to missing config.") from e
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ValueError("Failed to initialize ChatOpenAI.") from e

        # --- LangChain Memory --- (Return messages=True for prompt template)
        self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

        # --- LangChain Prompt Template --- (Using tuples per memory)
        # Ensure 'history' matches memory_key and '{input}' is used
        self.prompt_template = ChatPromptTemplate.from_messages([
            ('system', self.system_prompt),
            MessagesPlaceholder(variable_name="history"), # Use placeholder for memory
            ('human', '{input}')
        ])

        # --- LangChain LCEL Chain --- 
        # Chain structure: Load history -> Add input -> Format prompt -> Call LLM -> Parse output
        self.chain = (
            # 1. Log initial input
            RunnableLambda(lambda x: logger.debug(f"Chain input: {x}") or x)
            |
            # Load history from memory
            RunnablePassthrough.assign(
                history=RunnableLambda(self._load_memory)
            )
            # 2. Log input + history before prompt
            | RunnableLambda(lambda x: logger.debug(f"Data for prompt: {x}") or x)
            | self.prompt_template
            # 3. Log prompt object itself
            | RunnableLambda(lambda x: logger.debug(f"Formatted prompt messages: {x}") or x)
            | self.llm
            # 4. Log LLM output
            | RunnableLambda(lambda x: logger.debug(f"Raw ChatOpenAI output: {x}") or x)
            | StrOutputParser()
        )

        logger.info("AlpyAgent (LangChain) initialized successfully.")

    def get_response(self, user_input: str) -> str:
         """Processes user input using the LangChain chain and manages memory."""
         logger.info(f"Processing user input: {user_input}")
 
         # # Log memory state *before* invocation -- Temporarily commented out for debugging
         # try:
         #     current_history = self.memory.load_memory_variables({}).get('history', [])
         #     logger.debug(f"Memory state BEFORE chain invocation: {current_history}")
         # except Exception as e:
         #     logger.error(f"Error loading memory before invocation: {e}")
 
         try:
             logger.debug("--- Preparing to invoke LangChain chain --- ")
             # Input to the chain should match the expected variables ('input')
             response = self.chain.invoke({"input": user_input})
             logger.debug("--- LangChain chain invocation complete --- ")
            #  logger.info(f"<<< RAW RESPONSE RECEIVED FROM CHAIN: {response} >>>")

             # Save context (input and response) to memory
             # Note: invoke doesn't automatically save. We need to do it manually.
             # For LCEL chains, memory is typically managed *before* the chain
             # invocation (loading) and *after* (saving).
             # The RunnablePassthrough handles loading. Let's save here.
             self.memory.save_context({"input": user_input}, {"output": response})
             logger.debug(f"Saved context to memory. Input: '{user_input}', Output: '{response[:50]}...'" )

             return response.strip()

         except Exception as e:
             logger.error(f"Error during LangChain chain invocation: {e}")
             logger.error(f"Traceback: {traceback.format_exc()}")
             # Consider returning a user-friendly error message
             return "Sorry, I encountered an error processing your request."

    def _load_memory(self, inputs: Dict[str, Any]) -> list:
        """Loads memory variables and logs them."""
        logger.debug(f"_load_memory received inputs: {inputs}")
        memory_vars = self.memory.load_memory_variables(inputs)
        logger.debug(f"_load_memory retrieved: {memory_vars}")
        return memory_vars.get('history', [])

    def get_history(self):
        """Returns the conversation history from memory."""
        # Directly access the buffer from the memory object
        return self.memory.buffer_as_messages # Or buffer_as_str based on need

# Example Usage (if run directly - for basic testing)
# Note: Ensure config.py is accessible and correct
if __name__ == '__main__':
    # Configure logging for standalone testing
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    try:
        # Make sure config is loaded if running standalone
        # This might require adjustments depending on your project structure
        # If config relies on being run via `python -m src.main`, this won't work directly
        # For now, assuming config is loaded correctly when agent is initialized
        agent = AlpyAgent() 
        print("AlpyAgent (LangChain) initialized. Enter 'quit' to exit.")
        while True:
            try:
                user_input = input("> ")
                if user_input.lower() == 'quit':
                    break
                response = agent.get_response(user_input)
                print(f"Alpy: {response}")
                # Optional: print history
                # history = agent.get_history()
                # print(f"\n--- History ({len(history)} turns) ---\n{history}\n---------------------")
            except EOFError:
                print("\nInput stream closed. Exiting.")
                break # Exit cleanly if input stream is closed
            except KeyboardInterrupt:
                print("\nExiting...")
                break
    except Exception as e:
        print(f"Fatal Error initializing or running Agent: {e}")
        print(f"Traceback: {traceback.format_exc()}")
