# src/main.py
import sys
import logging
import re
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import langchain 
from rich.logging import RichHandler 

# Ensure src is in the path if running as a script
# (Not strictly needed when running with `python -m src.main`)
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from . import config
from .agent import AlpyAgent

# Configure Rich Logging
logging.basicConfig(
    level=config.LOG_LEVEL, 
    format="%(message)s", 
    datefmt="[%X]", 
    handlers=[RichHandler(rich_tracebacks=True, show_path=False, keywords=[])]
)
logger = logging.getLogger(__name__)

# Adjust RichHandler's default styles for log levels if needed
# For example, to make INFO dim:
# RichHandler.KEYWORDS = RichHandler.KEYWORDS + [("INFO", "dim")] 
# A better way is to customize the theme or use a custom Formatter if direct style mapping is hard.
# For now, the default RichHandler will at least make it look nicer than plain basicConfig.
# To specifically dim INFO logs from specific loggers, you might need more granular control or filters.

# Regex to find the Final Answer block
# This should match the FINAL_ANSWER_PATTERN from agent.py
FINAL_ANSWER_PATTERN_MAIN = r"Final Answer\s*:\s*(.*)"


def main():
    logger.info(f"Starting Alpy...")
    try:
        alpy_agent = AlpyAgent()
    except Exception as e:
        logger.error(f"Failed to initialize AlpyAgent: {e}")
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

    while True:
        try:
            prompt_text = Text("You: ", style="bold bright_blue")
            user_input = console.input(prompt_text)
            if user_input.lower() in ["exit", "quit"]:
                logger.info("User requested exit.")
                break
            
            if not user_input:
                continue

            # Response from agent is the full trace
            full_trace = alpy_agent.get_response(user_input)

            reasoning_steps_str = ""
            final_answer_str = ""

            # Parse the full_trace to separate reasoning from final answer
            final_answer_match = re.search(FINAL_ANSWER_PATTERN_MAIN, full_trace, re.DOTALL | re.IGNORECASE)

            if final_answer_match:
                final_answer_str = final_answer_match.group(1).strip()
                # Reasoning steps are everything before the match for Final Answer
                reasoning_steps_str = full_trace[:final_answer_match.start()].strip()
            else:
                # If no explicit "Final Answer:", the whole trace might be the response or an error message
                # For now, assume the whole trace is the "final answer" in this case, or it's an error handled by get_response.
                final_answer_str = full_trace.strip()
                # No separate reasoning steps if no Final Answer block is found.
                reasoning_steps_str = ""
            
            # Display reasoning steps (if any) with a dim style, outside the panel
            if reasoning_steps_str:
                console.print(Text(reasoning_steps_str, style="dim italic"))

            # The final answer goes into the panel
            alpy_panel = Panel(
                Text(final_answer_str if final_answer_str else "No specific answer identified.", style="cyan"),
                title="Alpy",
                title_align="left",
                border_style="magenta",
                padding=(0, 1)
            )
            console.print(alpy_panel)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Exiting.")
            print("\nExiting...")
            break
        except EOFError: 
             logger.info("EOFError received. Exiting.")
             print("\nExiting...")
             break
        except Exception as e:
            logger.exception("An unexpected error occurred in the main loop.")
            console.print_exception(show_locals=False)
            # print(f"An error occurred: {e}") 

    logger.info(f"Alpy shutting down.")

if __name__ == "__main__":
    main()
