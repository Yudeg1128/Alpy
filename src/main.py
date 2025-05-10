# src/main.py
import sys
import logging
import re
import json
import asyncio # Make sure asyncio is imported
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
# import langchain # Not directly used in main, but good if type hinting needs it
from rich.logging import RichHandler 
from typing import Optional

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from . import config
from .agent import AlpyAgent

logging.basicConfig(
    level=config.LOG_LEVEL, 
    format="%(message)s", 
    datefmt="[%X]", 
    handlers=[RichHandler(rich_tracebacks=True, show_path=False, keywords=[])]
)
logger = logging.getLogger(__name__)

async def amain():
    logger.info(f"Starting Alpy...")
    alpy_agent: Optional[AlpyAgent] = None # Type hint for clarity
    console = Console() # Define console here to pass to agent
    try:
        alpy_agent = AlpyAgent(rich_console=console) # Pass console to AlpyAgent
    except Exception as e:
        logger.error(f"Failed to initialize AlpyAgent: {e}", exc_info=True) # Add exc_info
        print(f"Fatal Error: Could not initialize Alpy. Check logs.", file=sys.stderr)
        sys.exit(1)
    
    welcome_panel = Panel(
        Text("Welcome to Alpy! Ask me anything.\nType 'exit' or 'quit' to end.", justify="center"),
        title="✨ Alpy AI Assistant ✨",
        border_style="bold green",
        padding=(1, 2)
    )
    console.print(welcome_panel)

    # Main interaction loop
    while True:
        try:
            prompt_text = Text("You: ", style="bold bright_blue")
            # Use asyncio.to_thread for blocking input to keep the event loop responsive
            user_input = await asyncio.to_thread(console.input, prompt_text)
            
            if user_input.lower() in ["exit", "quit"]:
                logger.info("User requested exit.")
                break # Exit the while loop
            
            if not user_input:
                continue

            if user_input.lower().startswith("/mode "):
                parts = user_input.lower().split(maxsplit=2)
                if len(parts) == 2:
                    new_mode = parts[1]
                    mode_set_response = await alpy_agent.set_mode(new_mode)
                    console.print(Text(mode_set_response, style="yellow"))
                else:
                    console.print(Text("Invalid /mode command. Use /mode agent or /mode chat.", style="red"))
                continue 

            # Handle /model command for switching LLMs
            if user_input.lower().startswith("/model "):
                parts = user_input.split(maxsplit=1) # split into '/model' and 'model_identifier'
                if len(parts) == 2 and parts[1].strip():
                    new_model_identifier = parts[1].strip()
                    logger.info(f"User requested model switch to: '{new_model_identifier}'")
                    if alpy_agent:
                        switch_result = await alpy_agent.switch_llm_model(new_model_identifier)
                        if switch_result.get("success"):
                            console.print(Panel(Text(f"✅ {switch_result.get('message')}", style="bold green"), title="Model Switch Success", border_style="green"))
                            console.print(Text(f"Current provider: {config.LLM_PROVIDER}, Active model: {alpy_agent.current_llm_model_name}", style="dim"))
                        else:
                            console.print(Panel(Text(f"❌ {switch_result.get('message')}", style="bold red"), title="Model Switch Failed", border_style="red"))
                    else:
                        console.print(Text("Agent not initialized, cannot switch model.", style="red"))
                else:
                    console.print(Text("Invalid /model command. Usage: /model <model_identifier>", style="red"))
                    # Optionally list available models
                    if config.LLM_PROVIDER == 'local':
                        console.print(Text(f"Available local models: {config.AVAILABLE_LOCAL_MODELS}", style="yellow"))
                    elif config.LLM_PROVIDER == 'openrouter':
                        console.print(Text(f"Available OpenRouter models: {config.AVAILABLE_OPENROUTER_MODELS}", style="yellow"))
                continue # Go to next iteration of while loop

            # Handle /provider command for switching LLM providers
            if user_input.lower().startswith("/provider "):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 2 and parts[1].strip():
                    new_provider_name = parts[1].strip().lower()
                    logger.info(f"User requested provider switch to: '{new_provider_name}'")
                    if alpy_agent:
                        switch_result = await alpy_agent.switch_llm_provider(new_provider_name)
                        if switch_result.get("success"):
                            console.print(Panel(Text(f"✅ {switch_result.get('message')}", style="bold green"), title="Provider Switch Success", border_style="green"))
                            # The message from switch_llm_provider already includes new model and provider
                        else:
                            console.print(Panel(Text(f"❌ {switch_result.get('message')}", style="bold red"), title="Provider Switch Failed", border_style="red"))
                    else:
                        console.print(Text("Agent not initialized, cannot switch provider.", style="red"))
                else:
                    console.print(Text("Invalid /provider command. Usage: /provider <local|openrouter>", style="red"))
                continue # Go to next iteration of while loop

            response_content = await alpy_agent.get_response(user_input) # This should be the final answer string

            if alpy_agent.mode == "agent":
                # The RichStreamingCallbackHandler now prints thoughts/actions/observations live.
                # So, we don't need to print the full_trace separately here.
                final_answer_text = response_content.strip() if response_content else "Agent did not produce a final answer."

                alpy_panel_title = f"Alpy (Agent Mode) - Final Answer"
                alpy_panel = Panel(
                    Markdown(final_answer_text),
                    title=alpy_panel_title,
                    title_align="left", border_style="magenta", padding=(0,1), expand=True
                )
                console.print(alpy_panel)

            elif alpy_agent.mode == "chat":
                chat_response_text = response_content.strip() if response_content else "No response in chat mode."
                alpy_panel_title = f"Alpy (Chat Mode)"
                alpy_panel = Panel(
                    Markdown(chat_response_text),
                    title=alpy_panel_title,
                    title_align="left", border_style="magenta", padding=(0,1), expand=False
                )
                console.print(alpy_panel)
            else: 
                error_text = "Error: Alpy is in an unrecognized mode."
                logger.error(f"Alpy in unrecognized mode: {alpy_agent.mode}")
                console.print(Text(error_text, style="bold red"))

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received during interaction. Exiting loop.")
            print("\nExiting...")
            break # Exit the while loop
        except EOFError: 
             logger.info("EOFError received during interaction. Exiting loop.")
             print("\nExiting...")
             break # Exit the while loop
        except Exception as e:
            logger.exception("An unexpected error occurred in the main interaction loop.")
            console.print_exception(show_locals=False)
            # Optionally, you might want to 'continue' here to allow further attempts,
            # or 'break' to exit the application on any error.
            # For robustness, 'continue' might be better unless error is critical.
            # For now, let it fall through, and the loop will continue. If fatal, it should be caught higher.

    # This block is executed AFTER the while loop has finished (due to break).
    if alpy_agent: # Check if agent was successfully initialized
        logger.info("Alpy shutting down. Closing agent and its resources...")
        try:
            await alpy_agent.close()
            logger.info("AlpyAgent and tools closed successfully.")
        except Exception as e_close:
            logger.error(f"Error during AlpyAgent close: {e_close}", exc_info=True)
    else:
        logger.info("Alpy shutting down. Agent was not initialized.")

if __name__ == "__main__":
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        # This catches Ctrl+C if it happens before amain() loop, or if amain() itself re-raises it.
        logger.info("Application terminated by KeyboardInterrupt at top level.")
    except Exception as e_global:
        logger.critical(f"Global unhandled exception in Alpy: {e_global}", exc_info=True)
    finally:
        logger.info("Alpy application has finished execution.")