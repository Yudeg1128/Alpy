# Alpy Project Structure

```
Alpy/
├── .git/
├── .gitignore
├── .python-version
├── MCP_GUIDE.md
├── README.md
├── llama_server.log
├── managed_fs_root/
├── mcp_servers/
│   └── [11 server files]
├── models/
├── prompts/
│   └── prompts.yaml
├── requirements.txt
├── src/
│   ├── agent.py
│   ├── config.py
│   └── [7 other files]
├── start_alpy.sh
└── test_filesystem_data/
```

## Key Directories:
- `mcp_servers/`: Contains 11 Model Context Protocol server implementations
- `src/`: Main application source code (9 files including agent.py and config.py)
- `prompts/`: LLM prompt configurations (prompts.yaml)
- `models/`: Local GGUF model storage

Note: Run `tree` command for complete recursive listing.
