# Alpy Project Structure

```
Alpy/
├── .git/
├── .gitignore
├── .python-version
├── MCP_GUIDE.md
├── PROJECT_STRUCTURE.md
├── README.md
├── FINANCIAL_MODEL_GUIDE.md
├── fund/
│   ├── financial_model_instructions.json
│   ├── financial_model_schema.json
│   └── model_guide.md
├── logs/
│   ├── .keep
│   ├── docparser_mcp_client_sees_document_parser_server.py.log
│   ├── fetcher_mcp_npx_server.log
│   ├── financial_modeling_orchestrator_test.log
│   ├── financial_modeling_orchestrator_v2_1_test.log
│   ├── llama_server.log
│   ├── playwright_mcp_client_sees_playwright_server.py.log
│   ├── playwright_py_server.log
│   └── puppeteer_mcp_npx_server.log
├── managed_fs_root/
├── mcp_servers/
│   ├── bash_server.py
│   ├── brave_server.py
│   ├── document_parser_server.py
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
├── otcmn_tool_test_output/
├── prompts/
│   ├── financial_modeling_prompts.yaml
│   └── prompts.yaml
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── agent.py
│   ├── config.py
│   ├── main.py
│   ├── financial_modeling/
│   │   ├── __init__.py
│   │   ├── phase_a_orchestrator.py
│   │   ├── quality_checker.py
│   │   └── utils.py
│   ├── otcmn_interaction/
│   │   ├── __init__.py
│   │   ├── common.py
│   │   ├── detail_page_handler.py
│   │   ├── interactor.py
│   │   └── listing_page_handler.py
│   └── tools/
│       ├── download_file_tool.py
│       ├── financial_phase_a_tool.py
│       ├── mcp_bash_tool.py
│       ├── mcp_brave_search_tool.py
│       ├── mcp_document_parser_tool.py
│       ├── mcp_fetcher_web_tool.py
│       ├── mcp_filesystem_tool.py
│       ├── mcp_media_display_tool.py
│       ├── mcp_playwright_tool.py
│       ├── mcp_puppeteer_tool.py
│       ├── mcp_python_tool.py
│       └── otcmn_scraper_tool.py
├── start_alpy.sh
├── wstest.py
├── requirements.txt
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
