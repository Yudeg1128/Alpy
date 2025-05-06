# src/config.py

import os

# --- LLM Configuration (using llama.cpp server API) ---

# Base URL for the OpenAI-compatible API provided by the llama.cpp server
LLM_API_BASE = "http://127.0.0.1:8080/v1"

# API Key (not strictly needed for local llama.cpp, but required by ChatOpenAI)
# Can be set via environment variable or hardcoded here (use dummy value for local)
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "dummy-key")

# Model name (Optional, helps identify the model being used via API)
LLM_MODEL_NAME = "qwen3-8b-gguf"

# Type of LLM connection/provider
LLM_TYPE = os.getenv('LLM_TYPE', 'llama-cpp')

# --- Generation Parameters ---
# These parameters are passed to the API
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', 0.7))
LLM_TOP_P = float(os.getenv('LLM_TOP_P', 0.8))
LLM_TOP_K = int(os.getenv('LLM_TOP_K', 20))
LLM_REPEAT_PENALTY = float(os.getenv('LLM_REPEAT_PENALTY', 1.1))
LLM_MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', 1024))

# --- Agent Configuration ---
AGENT_MEMORY_WINDOW_SIZE = int(os.getenv('AGENT_MEMORY_WINDOW_SIZE', 5)) # Number of past interactions to keep in memory
AGENT_MAX_ITERATIONS = int(os.getenv('AGENT_MAX_ITERATIONS', 7))       # Max steps for ReAct agent before stopping
# (Add any agent-specific configs here later)

# --- Logging ---
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# --- Prompt Templates (Load from file) ---
# (We can add logic here later to load prompts if needed)
PROMPT_TEMPLATES_PATH = "prompts/prompts.yaml"

verbose = os.getenv('LLAMA_VERBOSE', 'True').lower() == 'true'

# --- MCP Server Configuration ---
BASH_MCP_SERVER_ADDRESS = os.getenv('BASH_MCP_SERVER_ADDRESS', 'localhost:50051')
PYTHON_MCP_SERVER_ADDRESS = os.getenv('PYTHON_MCP_SERVER_ADDRESS', 'localhost:50052')

print(f"Config loaded. API Base: {LLM_API_BASE}")
