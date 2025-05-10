# src/main.py
import sys
import logging
import re
import json
import asyncio 
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.logging import RichHandler 
from rich.theme import Theme
from typing import Optional

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from . import config
from .agent import AlpyAgent

# Define a custom theme for log levels
log_theme = Theme({
    "logging.level.spam": "dim blue",
    "logging.level.debug": "dim blue", 
    "logging.level.info": "green",
    "logging.level.warning": "yellow",
    "logging.level.error": "bold red",
    "logging.level.critical": "bold red blink"
})

# Create a console with the custom theme
rich_console_for_logging = Console(theme=log_theme)

logging.basicConfig(
    level=config.LOG_LEVEL, 
    format="%(message)s", 
    datefmt="[%X]", 
    handlers=[RichHandler(
        console=rich_console_for_logging, 
        rich_tracebacks=True, 
        show_path=False, 
        keywords=[]
    )]
)
logger = logging.getLogger(__name__)

async def amain():
    logger.info(f"Starting Alpy...")
    alpy_agent: Optional[AlpyAgent] = None 
    # Use a new console for UI, separate from the one used for logging to avoid theme conflicts if any
    ui_console = Console() 
    try:
        alpy_agent = AlpyAgent(rich_console=ui_console) 
    except Exception as e:
        logger.error(f"Failed to initialize AlpyAgent: {e}", exc_info=True) 
        print(f"Fatal Error: Could not initialize Alpy. Check logs.", file=sys.stderr)
        sys.exit(1)
    
    welcome_panel = Panel(
        Text("Welcome to Alpy! Ask me anything.\nType 'exit' or 'quit' to end.", justify="center"),
        title=" Alpy AI Assistant ",
        border_style="bold green",
        padding=(1, 2)
    )
    ui_console.print(welcome_panel)

    # Main interaction loop
    while True:
        current_agent_task = None 
        try:
            prompt_text = Text("You: ", style="bold bright_blue")
            user_input = "" 

            try:
                user_input = await asyncio.to_thread(ui_console.input, prompt_text)
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt at input prompt. Exiting application.")
                ui_console.print("\nExiting Alpy...")
                break 
            except EOFError:
                logger.info("EOFError at input prompt. Exiting application.")
                ui_console.print("\nExiting Alpy...")
                break 
            
            if user_input.lower() in ["exit", "quit"]:
                logger.info("User requested exit.")
                break 

            if not user_input:
                continue

            # Handle commands like /mode, /model, /provider (these are quick)
            if user_input.lower().startswith("/mode "):
                parts = user_input.lower().split(maxsplit=2)
                if len(parts) == 2:
                    new_mode = parts[1]
                    mode_set_response = await alpy_agent.set_mode(new_mode)
                    ui_console.print(Text(mode_set_response, style="yellow"))
                else:
                    ui_console.print(Text("Invalid /mode command. Use /mode agent or /mode chat.", style="red"))
                continue 

            if user_input.lower().startswith("/model "):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 2 and parts[1].strip():
                    new_model_identifier = parts[1].strip()
                    logger.info(f"User requested model switch to: '{new_model_identifier}'")
                    if alpy_agent:
                        switch_result = await alpy_agent.switch_llm_model(new_model_identifier)
                        if switch_result.get("success"):
                            ui_console.print(Panel(Text(f" {switch_result.get('message')}", style="bold green"), title="Model Switch Success", border_style="green"))
                            ui_console.print(Text(f"Current provider: {alpy_agent.active_provider}, Active model: {alpy_agent.current_llm_model_name}", style="dim")) 
                        else:
                            ui_console.print(Panel(Text(f" {switch_result.get('message')}", style="bold red"), title="Model Switch Failed", border_style="red"))
                    else:
                        ui_console.print(Text("Agent not initialized, cannot switch model.", style="red"))
                else:
                    ui_console.print(Text("Invalid /model command. Usage: /model <model_identifier>", style="red"))
                    if alpy_agent: 
                        current_provider = alpy_agent.active_provider
                        if current_provider == 'local':
                            ui_console.print(Text(f"Available local models: {config.AVAILABLE_LOCAL_MODELS}", style="yellow"))
                        elif current_provider == 'openrouter':
                            ui_console.print(Text(f"Available OpenRouter models: {config.AVAILABLE_OPENROUTER_MODELS}", style="yellow"))
                        elif current_provider == 'google':
                             ui_console.print(Text(f"Available Google models: {config.AVAILABLE_GOOGLE_MODELS}", style="yellow"))
                continue

            if user_input.lower().startswith("/provider "):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 2 and parts[1].strip():
                    new_provider_name = parts[1].strip().lower()
                    logger.info(f"User requested provider switch to: '{new_provider_name}'")
                    if alpy_agent:
                        switch_result = await alpy_agent.switch_llm_provider(new_provider_name)
                        if switch_result.get("success"):
                            ui_console.print(Panel(Text(f" {switch_result.get('message')}", style="bold green"), title="Provider Switch Success", border_style="green"))
                        else:
                            ui_console.print(Panel(Text(f" {switch_result.get('message')}", style="bold red"), title="Provider Switch Failed", border_style="red"))
                    else:
                        ui_console.print(Text("Agent not initialized, cannot switch provider.", style="red"))
                else:
                    ui_console.print(Text("Invalid /provider command. Usage: /provider <local|openrouter|google>", style="red")) 
                continue

            # If it's not a command or exit, it's a prompt for the agent
            response_content = None
            try:
                logger.info("Agent is thinking... (Press Ctrl+C to attempt to cancel)")
                current_agent_task = asyncio.create_task(alpy_agent.get_response(user_input))
                response_content = await current_agent_task
            except asyncio.CancelledError: 
                logger.info("Agent task was cancelled by an external request (not KeyboardInterrupt here).")
                if alpy_agent:
                    ui_console.print(Panel(Text("", justify="center"), title="Operation Cancelled", border_style="yellow"))
                response_content = None 
            except KeyboardInterrupt: 
                logger.warning("KeyboardInterrupt received during active agent operation.")
                if current_agent_task and not current_agent_task.done():
                    logger.info("Cancelling the active agent task...")
                    current_agent_task.cancel()
                    try:
                        await current_agent_task 
                    except asyncio.CancelledError:
                        logger.info("Agent task successfully processed cancellation due to KeyboardInterrupt.")
                    ui_console.print(Panel(Text("", justify="center"), title="Operation Interrupted", border_style="yellow"))
                else: 
                    ui_console.print(Panel(Text("", justify="center"), title="Operation Interrupted", border_style="yellow"))
                response_content = None 
            
            if response_content is not None:
                if alpy_agent.mode == "agent":
                    final_answer_text = response_content.strip() 
                    alpy_panel_title = f"Alpy (Agent Mode) - Final Answer"
                    alpy_panel = Panel(
                        Markdown(final_answer_text),
                        title=alpy_panel_title,
                        title_align="left", border_style="magenta", padding=(0,1), expand=True
                    )
                    ui_console.print(alpy_panel)
                elif alpy_agent.mode == "chat":
                    chat_response_text = response_content.strip() 
                    alpy_panel_title = f"Alpy (Chat Mode)"
                    alpy_panel = Panel(
                        Markdown(chat_response_text),
                        title=alpy_panel_title,
                        title_align="left", border_style="magenta", padding=(0,1), expand=False
                    )
                    ui_console.print(alpy_panel)
                else: 
                    error_text = "Error: Alpy is in an unrecognized mode."
                    logger.error(f"Alpy in unrecognized mode: {alpy_agent.mode}")
                    ui_console.print(Text(error_text, style="bold red"))
            # If response_content is None (due to cancellation), loop continues to next prompt

        except KeyboardInterrupt: # Catches KI if it happens outside input or agent task handling.
            logger.info("General KeyboardInterrupt in main loop. Exiting.")
            ui_console.print("\nExiting Alpy...")
            break 
        except EOFError: 
             logger.info("General EOFError in main loop. Exiting.")
             ui_console.print("\nExiting Alpy...")
             break 
        except Exception as e:
            logger.exception("An unexpected error occurred in the main interaction loop.")
            if hasattr(ui_console, 'print_exception') and callable(ui_console.print_exception):
                 ui_console.print_exception(show_locals=False)
            else:
                # Fallback if ui_console is not a Rich Console or doesn't have print_exception
                import traceback
                traceback.print_exc()

    if alpy_agent: 
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
        logger.info("Application terminated by KeyboardInterrupt at top level.")
    except Exception as e_global:
        logger.critical(f"Global unhandled exception in Alpy: {e_global}", exc_info=True)
    finally:
        logger.info("Alpy application has finished execution.")