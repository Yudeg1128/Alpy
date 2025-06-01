# src/config.py

import os
from pathlib import Path

# --- General Configuration ---
LOG_DIR = os.getenv('LOG_DIR', 'logs')

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
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyDhpY9Egn3NxbWQfi3Ttp4h2_N1ImPv1l4")
AVAILABLE_GOOGLE_MODELS = [
    'gemini-2.0-flash-lite', 'gemini-2.5-flash-preview-04-17', 'gemini-2.5-pro-preview-05-06', 'gemini-2.0-flash',     
    'gemini-2.0-flash-preview-image-generation', 'gemini-1.5-flash',  
    'gemini-1.5-flash-8b', 'gemini-1.5-pro', 'gemini-embedding-exp', 'text-embedding-004',     
    'veo-2.0-generate-001', 'gemini-2.0-flash-live-001'
]
ACTIVE_GOOGLE_MODEL = os.getenv("ACTIVE_GOOGLE_MODEL", AVAILABLE_GOOGLE_MODELS[0])

# --- Gemini API Call Control (used by _InternalGeminiLLMService in PhaseAOrchestrator) ---
# Values based on Gemini API documentation for free tier Flash models (e.g., gemini-2.0-flash)
# RPM: Requests Per Minute
# TPM: Tokens Per Minute (prompt + response)
# RPD: Requests Per Day
# Concurrent requests are also limited by the API (e.g., ~3 for free tier).

GEMINI_CONCURRENT_REQUEST_LIMIT = int(os.getenv("GEMINI_CONCURRENT_REQUEST_LIMIT", 1))  # More conservative parallel limit
GEMINI_RPM_LIMIT = int(os.getenv("GEMINI_RPM_LIMIT", 15))  # More conservative RPM to avoid rate limits
GEMINI_TPM_LIMIT = int(os.getenv("GEMINI_TPM_LIMIT", 1000000)) # More conservative TPM buffer
GEMINI_RPD_LIMIT = int(os.getenv("GEMINI_RPD_LIMIT", 1500))    # More conservative RPD buffer

GEMINI_INTER_REQUEST_DELAY_SECONDS = float(os.getenv("GEMINI_INTER_REQUEST_DELAY_SECONDS", 2.0)) # Increased delay between requests
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", 3)) # Reduced max retries
GEMINI_INITIAL_RETRY_DELAY_SECONDS = float(os.getenv("GEMINI_INITIAL_RETRY_DELAY_SECONDS", 10.0)) # Increased initial retry delay

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

# --- RAG Configuration ---
RAG_EMBEDDING_MODEL_NAME = os.getenv("RAG_EMBEDDING_MODEL_NAME", "text-embedding-004") # Or a newer Gemini embedding model
RAG_EMBEDDING_DIM = int(os.getenv("RAG_EMBEDDING_DIM", 768)) # Dimension for text-embedding-004
RAG_VECTOR_STORE_BASE_PATH = os.getenv("RAG_VECTOR_STORE_BASE_PATH", str(Path(__file__).resolve().parent.parent / "data" / "vector_stores")) # Points to Alpy/data/vector_stores
RAG_CHUNK_OVERLAP_PERCENTAGE = float(os.getenv("RAG_CHUNK_OVERLAP_PERCENTAGE", 0.1)) # 10% overlap
RAG_MIN_CHUNK_SIZE_CHARS = int(os.getenv("RAG_MIN_CHUNK_SIZE_CHARS", 100)) # Minimum characters for a chunk to be considered useful
RAG_FAIL_IF_NO_CHUNKS = os.getenv("RAG_FAIL_IF_NO_CHUNKS", "True").lower() == "true"
RAG_CHUNK_STRATEGY = os.getenv("RAG_CHUNK_STRATEGY", "semantic_markdown") # Options: 'simple', 'semantic_markdown'
# RAG_CHUNK_OVERLAP (character based) is now calculated dynamically in PhaseAOrchestrator using RAG_CHUNK_SIZE and RAG_CHUNK_OVERLAP_PERCENTAGE
RAG_NUM_RETRIEVED_CHUNKS = int(os.getenv("RAG_NUM_RETRIEVED_CHUNKS", 5))
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", 1000)) # Chars for text chunker
OCR_ENABLED = os.getenv("OCR_ENABLED", "True").lower() == "true" # Whether to use OCR for PDFs
RAG_MAX_CONTEXT_LENGTH = int(os.getenv("RAG_MAX_CONTEXT_LENGTH", 4000)) # Max characters for combined RAG context

# Ensure the RAG_VECTOR_STORE_BASE_PATH directory exists
Path(RAG_VECTOR_STORE_BASE_PATH).mkdir(parents=True, exist_ok=True)

# --- Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
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
