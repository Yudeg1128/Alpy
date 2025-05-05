# Alpy Project Plan

## 1. Goal

Develop Alpy, a command-line AI assistant capable of:
*   Leveraging a locally running Large Language Model (LLM) using `Mungert/Qwen3-8B-abliterated-Q4_K_M.gguf` served via `llama.cpp`'s OpenAI-compatible API server.
*   Integrating with locally running MCP (Model Context Protocol) servers (from [github.com/modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers)) to perform device control tasks (e.g., bash commands, Python execution).
*   Using `langchain-openai` for LLM interaction and conversation management.

## 2. Core Components

*   **Alpy Python Application:** The main CLI interface (`src/main.py`), agent logic (`src/agent.py`), configuration (`src/config.py`), and prompts (`prompts/prompts.yaml`). Uses `langchain-openai` to interact with the LLM server.
*   **Local LLM Server:** A separate `llama.cpp` server process running the specified GGUF model, exposing an OpenAI-compatible API endpoint (e.g., `http://127.0.0.1:8080/v1`).
*   **MCP Server(s):** Separate processes running specific MCP servers (e.g., `mcp_bash_server`) for tool execution.

## 3. Phases

### Phase 1: Core Alpy Setup (LLM Integration) - COMPLETE

*   Create project structure (`src/`, `prompts/`).
*   Add core dependencies (`langchain-openai`, `python-dotenv`, `pyyaml`, etc. - see `requirements.txt`).
*   Configure LLM connection details (`src/config.py`) for the `llama.cpp` server API.
*   Define system prompt (`prompts/prompts.yaml`).
*   Initialize `ChatOpenAI` and basic conversational memory (`ConversationBufferMemory`) in `src/agent.py` using LangChain.
*   Construct LCEL chain (`RunnablePassthrough`, `ChatPromptTemplate`, `ChatOpenAI`, `StrOutputParser`).
*   Implement basic CLI loop in `src/main.py` for user interaction.
*   Test basic conversation and memory.
*   **Key Change:** Shifted from direct `llama-cpp-python` integration or `qwen-agent` to using the `llama.cpp` server API + `langchain-openai` for robustness and standard framework usage.

### Phase 2: MCP Server Tool Integration (Using LangChain Tools)

*   **Goal:** Enable Alpy to use external tools (like a bash shell) via MCP, orchestrated by LangChain.
*   **Steps:**
    1.  **Setup & Run MCP Server:** Start a specific MCP server (e.g., `mcp_bash_server` from the `modelcontextprotocol/servers` repo) in a separate terminal/process. Note its address (e.g., `http://localhost:50051`).
    2.  **Define LangChain Tool:** In `src/tools.py` (new file), create a LangChain `Tool` or `BaseTool` subclass (e.g., `BashMCPTool`).
        *   This tool's `_run` method will encapsulate the logic to send a request (using `httpx` or `requests`) to the running MCP server's API endpoint (e.g., `/command`) and return the result.
        *   Define the tool's `name` and `description` clearly for the LLM.
    3.  **Integrate Tool(s) into Agent:** Modify `src/agent.py`:
        *   Import the defined tool(s).
        *   **Option A (Simple Invocation):** Keep the current conversational chain. In `get_response`, after getting the LLM response, parse it to see if it's asking to use a tool. If so, manually invoke the tool and potentially feed the result back to the LLM in a subsequent turn.
        *   **Option B (LangChain Agent):** Re-architect the agent to use a LangChain Agent Executor (e.g., `create_openai_tools_agent`). This involves:
            *   Passing the LLM and the list of tools to the agent constructor.
            *   Modifying the prompt template to be suitable for an agent (LangChain Hub has examples).
            *   Invoking the `agent_executor` instead of the simple chain.
            *   The agent executor handles parsing the LLM's tool requests, calling the tools, and feeding back results automatically.
    4.  **Refine Prompts:** Adjust the system prompt in `prompts/prompts.yaml` to explicitly instruct the LLM on the available tools, their purpose, and how to request their use (especially if using Option A).
    5.  **Test Tool Usage:** Verify that Alpy can correctly understand a request requiring a tool (e.g., "list files in the current directory"), invoke the `BashMCPTool`, and present the result.

### Phase 3: Expansion & Refinement

*   Add more MCP tools (Python execution, web search, etc.).
*   Improve error handling for tool failures or unavailable servers.
*   Implement confirmation steps for potentially risky tool actions.
*   Enhance the CLI presentation (`src/main.py`).
*   Explore advanced memory strategies or RAG if needed.
