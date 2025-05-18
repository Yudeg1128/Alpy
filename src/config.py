# src/config.py

import os

# --- LLM Provider Configuration ---
# Choose 'local' for llama.cpp server or 'openrouter' for OpenRouter.ai
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'google') # 'local' or 'openrouter'

# --- Local LLM Configuration (e.g., llama.cpp server API) ---
LOCAL_LLM_API_BASE = os.getenv("LOCAL_LLM_API_BASE", "http://127.0.0.1:8080/v1")
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "dummy-key") # Typically not needed for local
AVAILABLE_LOCAL_MODELS = [
    "Qwen3-0.6B-abliterated-iq3_xxs.gguf",
    "Qwen3-4B-abliterated-q6_k_m.gguf", 
    "Qwen3-8B-abliterated-iq2_xxs.gguf",
    "Qwen3-8B-abliterated-q4_k_m.gguf",
    "Goekdeniz-Guelmez_Josiefied-Qwen3-8B-abliterated-v1-Q2_K.gguf",
    "Llama-3.2-1B-Instruct-abliterated.Q2_K.gguf",
    # Add other local model names here, e.g., "llama3-8b-instruct-gguf"
]
ACTIVE_LOCAL_LLM_MODEL = os.getenv(
    "ACTIVE_LOCAL_LLM_MODEL", 
    AVAILABLE_LOCAL_MODELS[0] if AVAILABLE_LOCAL_MODELS else "default-local-model-not-set"
)

# --- OpenRouter.ai LLM Configuration ---
OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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") # Replace with your actual key
AVAILABLE_OPENROUTER_MODELS = [
    "rekaai/reka-flash-3:free",
    # User will provide other model IDs directly via /model command
]
ACTIVE_OPENROUTER_MODEL = os.getenv(
    "ACTIVE_OPENROUTER_MODEL", 
    AVAILABLE_OPENROUTER_MODELS[0] if AVAILABLE_OPENROUTER_MODELS else "rekaai/reka-flash-3:free" # Fallback if list is somehow empty
)
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "https://your-site-url.com") # Recommended: Your app's website
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "Alpy") # Recommended: Your app's name

# --- Google API Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyBN8f_HMB291iGskfw2oePbF9-W_5kIZfo")
AVAILABLE_GOOGLE_MODELS = [
    'gemini-2.0-flash-lite', 'gemini-2.5-flash-preview-04-17', 'gemini-2.5-pro-preview-05-06', 'gemini-2.0-flash',     
    'gemini-2.0-flash-preview-image-generation', 'gemini-1.5-flash',  
    'gemini-1.5-flash-8b', 'gemini-1.5-pro', 'gemini-embedding-exp', 'text-embedding-004',     
    'veo-2.0-generate-001', 'gemini-2.0-flash-live-001'
]
ACTIVE_GOOGLE_MODEL = os.getenv("ACTIVE_GOOGLE_MODEL", AVAILABLE_GOOGLE_MODELS[0])

# --- Generation Parameters (Common for both providers) ---
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', 0.7))
LLM_TOP_P = float(os.getenv('LLM_TOP_P', 0.8))
LLM_TOP_K = int(os.getenv('LLM_TOP_K', 20))
LLM_REPEAT_PENALTY = float(os.getenv('LLM_REPEAT_PENALTY', 1.1))
LLM_MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', 1024))

# --- Agent Configuration ---
AGENT_MEMORY_WINDOW_SIZE = int(os.getenv('AGENT_MEMORY_WINDOW_SIZE', 5)) # Number of past interactions to keep in memory
AGENT_MAX_ITERATIONS = int(os.getenv('AGENT_MAX_ITERATIONS', 20))       # Max steps for ReAct agent before stopping
# (Add any agent-specific configs here later)

# --- Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# --- Prompt Templates (Load from file) ---
PROMPT_TEMPLATES_PATH = "prompts/prompts.yaml"

# --- MCP Server Configuration ---
BASH_MCP_SERVER_ADDRESS = os.getenv('BASH_MCP_SERVER_ADDRESS', 'localhost:50051')
PYTHON_MCP_SERVER_ADDRESS = os.getenv('PYTHON_MCP_SERVER_ADDRESS', 'localhost:50052')

print(f"Initializing Alpy with LLM Provider: {LLM_PROVIDER}")

BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY", "BSAZcb1kxyIgyqKkA7hj-OohX02D765")

# Update the print statement to reflect the new structure
if LLM_PROVIDER == 'local':
    print(f"  Local LLM API Base: {LOCAL_LLM_API_BASE}")
    print(f"  Available Local Models: {AVAILABLE_LOCAL_MODELS}")
    print(f"  Active Local Model: {ACTIVE_LOCAL_LLM_MODEL}")
elif LLM_PROVIDER == 'openrouter':
    print(f"  OpenRouter API Base: {OPENROUTER_API_BASE}")
    print(f"  Available OpenRouter Models: {AVAILABLE_OPENROUTER_MODELS}")
    print(f"  Active OpenRouter Model: {ACTIVE_OPENROUTER_MODEL}")
    # Obscure the API key for printing, showing only last 4 chars if long enough
    api_key_display = OPENROUTER_API_KEY
    if len(api_key_display) > 7 and api_key_display != "YOUR_OPENROUTER_API_KEY_PLACEHOLDER":
        api_key_display = "********" + api_key_display[-4:]
    elif api_key_display == "YOUR_OPENROUTER_API_KEY_PLACEHOLDER":
        api_key_display = "YOUR_OPENROUTER_API_KEY_PLACEHOLDER (not set)"
    print(f"  OpenRouter API Key: {api_key_display}")
elif LLM_PROVIDER == 'google':
    print(f"  Google API Key: {GOOGLE_API_KEY}")
    print(f"  Available Google Models: {AVAILABLE_GOOGLE_MODELS}")
    print(f"  Active Google Model: {ACTIVE_GOOGLE_MODEL}")


# dy@emipmongolia.com
# MidasHYB#1

