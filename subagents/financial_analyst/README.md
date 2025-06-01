# Financial Analyst Subagent

This subagent is an autonomous, persistent background worker for financial analysis tasks. It maintains its own memory, task queue, and reporting. It is fully isolated from the main agent and can be run in its own virtualenv.

## Structure
- `financial_analyst/agent.py`: Subagent entrypoint and orchestration
- `financial_analyst/memory.py`: Persistent memory and task/result store
- `financial_analyst/tasks.py`: Task definitions and handlers
- `financial_analyst/config.py`: Configurations

## Usage
- Install dependencies: `pip install -r requirements.txt`
- Run: `python -m financial_analyst.agent`
