# Alpy Project Structure

```
Alpy/
├── .git/
├── .gitignore
├── .python-version
├── MCP_GUIDE.md
├── PROJECT_STRUCTURE.md
├── README.md
├── fund/
│   ├── financial_model_instructions.json
│   └── model_guide.md
├── logs/
│   ├── .keep
│   ├── fetcher_mcp_npx_server.log
│   ├── llama_server.log
│   ├── playwright_mcp_client_sees_playwright_server.py.log
│   ├── playwright_py_server.log
│   └── puppeteer_mcp_npx_server.log
├── managed_fs_root/
├── mcp_servers/
│   ├── bash_server.py
│   ├── brave_server.py
│   ├── filesystem_server/
│   ├── media_display_server.py
│   ├── playwright_server.py
│   ├── python_server.py
│   └── requirements.txt
├── models/
│   ├── Goekdeniz-Guelmez_Josiefied-Qwen3-8B-abliterated-v1-Q2_K.gguf
│   ├── Llama-3.2-1B-Instruct-abliterated.Q2_K.gguf
│   ├── Qwen3-0.6B-abliterated-iq3_xxs.gguf
│   ├── Qwen3-4B-abliterated-q6_k_m.gguf
│   ├── Qwen3-8B-abliterated-iq2_xxs.gguf
│   └── Qwen3-8B-abliterated-q4_k_m.gguf
├── prompts/
│   └── prompts.yaml
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── agent.py
│   ├── config.py
│   ├── main.py
│   └── tools/
├── start_alpy.sh
├── test_otcmn_scraper_output_tool/
│   ├── documents/
│   └── tables/
├── wstest.py


```

## Key Directories:
- `fund/`: Financial modeling resources and guides
- `logs/`: All runtime and server log files (centralized)
- `mcp_servers/`: Model Context Protocol server implementations (Python and subfolders)
- `models/`: Local GGUF model storage
- `prompts/`: LLM prompt configurations (prompts.yaml)
- `src/`: Main application source code (including agent.py, config.py, tools/)
- `test_otcmn_scraper_output_tool/`: Test data for OTCMN scraper tools

Note: Run `tree` command for complete recursive listing.
