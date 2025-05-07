# src/main.py
import sys
import logging
import re
import asyncio # Make sure asyncio is imported
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
# import langchain # Not directly used in main, but good if type hinting needs it
from rich.logging import RichHandler 

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

FINAL_ANSWER_PATTERN_MAIN = r"Final Answer\s*:\s*(.*)"

async def amain():
    logger.info(f"Starting Alpy...")
    alpy_agent: Optional[AlpyAgent] = None # Type hint for clarity
    try:
        alpy_agent = AlpyAgent()
    except Exception as e:
        logger.error(f"Failed to initialize AlpyAgent: {e}", exc_info=True) # Add exc_info
        print(f"Fatal Error: Could not initialize Alpy. Check logs.", file=sys.stderr)
        sys.exit(1)
    
    console = Console()
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

            response_content = await alpy_agent.get_response(user_input)

            if alpy_agent.mode == "agent":
                reasoning_steps_str = ""
                final_answer_str = ""
                final_answer_match = re.search(FINAL_ANSWER_PATTERN_MAIN, response_content, re.DOTALL | re.IGNORECASE)
                if final_answer_match:
                    final_answer_str = final_answer_match.group(1).strip()
                    reasoning_steps_str = response_content[:final_answer_match.start()].strip()
                else:
                    final_answer_str = response_content.strip()
                if reasoning_steps_str:
                    console.print(Text(reasoning_steps_str, style="dim italic"))
                display_text = final_answer_str if final_answer_str else "No specific answer identified (agent mode)."
            elif alpy_agent.mode == "chat":
                display_text = response_content.strip() if response_content.strip() else "No specific answer identified (chat mode)."
            else: 
                display_text = "Error: Alpy is in an unrecognized mode."
                logger.error(f"Alpy in unrecognized mode: {alpy_agent.mode}")

            alpy_panel = Panel(
                Text(display_text, style="cyan"),
                title=f"Alpy ({alpy_agent.mode} mode)",
                title_align="left", border_style="magenta", padding=(0, 1)
            )
            console.print(alpy_panel)

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