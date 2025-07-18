# Alpy - Your Local AI Assistant

Alpy is a command-line AI assistant designed to run locally on your machine, leveraging the power of local Large Language Models (LLMs) through `llama.cpp` and the flexibility of the `LangChain` framework.

## ✨ Features

*   **Local LLM Interaction**: Connects to a locally running `llama-server` (compatible with OpenAI API standards) to process your requests.
*   **Dual Interaction Modes**:
    *   **Agent Mode**: A ReAct-style agent capable of thinking step-by-step and utilizing tools to perform actions (e.g., running shell commands, executing Python code, interacting with MCP-based tools).
    *   **Chat Mode**: A straightforward conversational interface for direct interaction with the LLM.
*   **Tool Integration**: Equipped with built-in tools and designed for extension via MCP:
    *   `bash_executor`: Executes non-interactive bash commands.
    *   `python_executor`: Executes Python code snippets.
    *   **Model Context Protocol (MCP) Tools**: Actively integrates with external tools running as MCP servers, allowing for a diverse and expandable set of capabilities (see `MCP_GUIDE.md` and `mcp_servers/` directory).
*   **Rich CLI Experience**: Uses the `rich` library for an enhanced and user-friendly command-line interface.
*   **Configurable**: Settings for LLM, agent behavior, and prompts are managed through configuration files.
*   **Conversation Memory**: Remembers recent parts of the conversation for contextual interactions.

## 🚀 Getting Started

### Prerequisites

*   **Python Environment**: Python 3.x with `pyenv` (recommended for managing Python versions).
*   **llama.cpp**: A compiled `llama.cpp` build with the `llama-server` executable. The `start_alpy.sh` script expects this to be in a specific location (configurable in the script).
*   **LLM Model**: A GGUF-formatted LLM model compatible with `llama.cpp`. The path to this model is configured in `start_alpy.sh`.
*   **Required Python Packages**: Listed in `requirements.txt`.

### Setup

1.  **Clone the Repository (if applicable)**:
    ```bash
    # git clone <repository-url>
    # cd Alpy
    ```
2.  **Configure `llama.cpp` and Model Paths**:
    *   Edit `start_alpy.sh` and update the following variables to match your setup:
        *   `LLAMA_CPP_BUILD_DIR`: Path to your `llama.cpp/build` directory.
        *   `MODEL_PATH`: Full path to your `.gguf` LLM model file.
        *   `PYENV_ENV_NAME`: The name of your pyenv environment for this project (e.g., `alpy_env`).
3.  **Set up Python Environment (using pyenv)**:
    ```bash
    pyenv install <python-version-specified-in-.python-version-or-your-choice>
    pyenv virtualenv <python-version> <your-pyenv-env-name>
    pyenv local <your-pyenv-env-name>
    ```
4.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Running Alpy

Execute the startup script:

```bash
./start_alpy.sh
```

This script will:
1.  Initialize `pyenv`.
2.  Start the `llama-server` in the background, logging its output to `llama_server.log`.
3.  Open a new `gnome-terminal` window running the Alpy application.

Upon closing the Alpy terminal window, the script will attempt to stop the `llama-server` process.

## ⌨️ Usage

Once Alpy is running in the new terminal, you'll see a welcome message.

*   **Chatting**: Simply type your questions or requests and press Enter.
*   **Switching Modes**:
    *   To switch to agent mode: `/mode agent`
    *   To switch to chat mode: `/mode chat`
*   **Exiting**: Type `exit` or `quit` and press Enter, or press `Ctrl+C`.

In **Agent Mode**, Alpy might show its internal thought process or actions before providing a final answer. It will use its tools if your request requires them (e.g., "list files in the current directory" might trigger the `bash_executor`).

## 🛠️ Configuration

*   **`src/config.py`**: Contains core configurations such as:
    *   `LLM_API_BASE`, `LLM_MODEL_NAME`
    *   LLM generation parameters (temperature, max tokens, etc.)
    *   Agent settings (memory window size, max iterations)
    *   Paths to MCP servers (for future tool extensions).
*   **`prompts/prompts.yaml`**: Defines the system prompts used to instruct the LLM in different modes (general, agent, chat). This file is crucial for shaping Alpy's behavior and persona.
*   **`start_alpy.sh`**: Contains paths for `llama.cpp` and the LLM model, as well as the `pyenv` environment name.

## ⚙️ Key Components

*   **`start_alpy.sh`**: Main startup script for launching `llama-server` and the Alpy application.
*   **`src/`**: Source code for Alpy.
    *   **`main.py`**: CLI entry point.
    *   **`agent.py`**: Core agent logic, LangChain and tool integration.
    *   **`config.py`**: Configuration.
    *   **`financial_modeling/`**: Financial modeling orchestrators and utilities.
    *   **`otcmn_interaction/`**: Handlers for OTCMN (Over-the-Counter Market Network) interactions.
    *   **`tools/`**: Built-in and MCP tool implementations (e.g., `otcmn_scraper_tool.py`, `mcp_*_tool.py`, `download_file_tool.py`).
*   **`prompts/`**: Prompt YAMLs for LLMs (`prompts.yaml`, `financial_modeling_prompts.yaml`).
*   **`requirements.txt`**: Python dependencies.
*   **`models/`**: GGUF-formatted LLM models.
*   **`mcp_servers/`**: MCP server implementations (e.g., `bash_server.py`, `brave_server.py`, `document_parser_server.py`, etc.).
*   **`fund/`**: Financial model instructions and schemas.
*   **`logs/`**: Log files for various components and tests.
*   **`otcmn_tool_test_output/`**: Output from OTCMN tool tests.
*   **`PROJECT_STRUCTURE.md`**: Additional details on project organization.
*   **`MCP_GUIDE.md`**: Guide for MCP tool/server development.
*   **`FINANCIAL_MODEL_GUIDE.md`**: Guide for financial modeling features.
*   **Other files**: `.gitignore`, `.python-version`, and various documentation and log files.

This structure provides a clear separation between core application logic, tool/server implementations, configuration, prompts, logs, models, and documentation.

## 🤖 Tech Stack

*   **Python**
*   **LangChain**: Framework for developing applications powered by language models.
*   **llama.cpp**: For running LLMs locally via `llama-server`.
*   **OpenAI API (compatible)**: Used by LangChain to communicate with the `llama-server`.
*   **Rich**: For creating rich text and beautiful formatting in the terminal.
*   **PyYAML**: For parsing YAML configuration files (prompts).
*   **pyenv**: For Python version management.

## 🔧 Potential Future Development

*   Continued expansion and refinement of Model Context Protocol (MCP) server integrations and available tools.
*   Enhanced tool management and discovery within Alpy.
*   More advanced error handling and user feedback mechanisms across all components.
*   User interface improvements, potentially exploring GUI options in the long term.
