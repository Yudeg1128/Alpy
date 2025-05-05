# src/main.py
import sys
import logging

# Ensure src is in the path if running as a script
# (Not strictly needed when running with `python -m src.main`)
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from . import config
from .agent import AlpyAgent

# Configure logging (can be more sophisticated, e.g., file logging)
logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

def main():
    logger.info(f"Starting Alpy...")
    try:
        alpy_agent = AlpyAgent()
    except Exception as e:
        logger.error(f"Failed to initialize AlpyAgent: {e}")
        print(f"Fatal Error: Could not initialize Alpy. Check logs.", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nWelcome to Alpy! Type 'exit' or 'quit' to end.")

    while True:
        try:
            user_input = input("> ")
            if user_input.lower() in ["exit", "quit"]:
                logger.info("User requested exit.")
                break
            
            if not user_input:
                continue

            # Process the request through the agent
            response = alpy_agent.get_response(user_input)
            
            # Print the agent's response
            print(f"Alpy: {response}")

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Exiting.")
            print("\nExiting...")
            break
        except EOFError: # Handle Ctrl+D
             logger.info("EOFError received. Exiting.")
             print("\nExiting...")
             break
        except Exception as e:
            logger.exception("An unexpected error occurred in the main loop.")
            print(f"An error occurred: {e}")
            # Optionally, decide whether to continue or exit on errors

    logger.info(f"Alpy shutting down.")

if __name__ == "__main__":
    main()
