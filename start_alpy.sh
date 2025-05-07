#!/bin/bash

# Explicitly initialize pyenv for non-interactive shells
if [ -d "$HOME/.pyenv/bin" ]; then
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv init -)"
fi

# Script to start llama-server and the Alpy application

# --- Configuration ---
LLAMA_CPP_BUILD_DIR="/home/me/CascadeProjects/llama.cpp/build"
LLAMA_SERVER_EXEC="${LLAMA_CPP_BUILD_DIR}/bin/llama-server"
MODEL_PATH="/home/me/CascadeProjects/Alpy/models/Qwen3-4B-abliterated-q6_k_m.gguf"
ALPY_PROJECT_DIR="/home/me/CascadeProjects/Alpy"
PYENV_ENV_NAME="alpy_env"

LLAMA_SERVER_LOG="${ALPY_PROJECT_DIR}/llama_server.log"
ALPY_APP_LOG="${ALPY_PROJECT_DIR}/alpy_app.log"

# --- Check if executables exist ---
if [ ! -f "${LLAMA_SERVER_EXEC}" ]; then
    echo "Error: llama-server executable not found at ${LLAMA_SERVER_EXEC}"
    exit 1
fi

if ! command -v pyenv &> /dev/null; then
    echo "Error: pyenv command not found. Make sure pyenv is installed and initialized."
    exit 1
fi

# --- Start llama-server ---
echo "Starting llama-server in the background..."
echo "Log file: ${LLAMA_SERVER_LOG}"
cd "${LLAMA_CPP_BUILD_DIR}" || exit 1 # Change to server directory
# Start in background, redirect stdout and stderr to log file
"${LLAMA_SERVER_EXEC}" -m "${MODEL_PATH}" --chat-template chatml -ngl -1 -c 4096 --host 127.0.0.1 --port 8080 > "${LLAMA_SERVER_LOG}" 2>&1 &
LLAMA_SERVER_PID=$!
echo "llama-server started with PID: ${LLAMA_SERVER_PID}"

# Give the server a moment to start up
sleep 2

# --- Start Alpy Application in a new terminal ---
echo "Launching Alpy application in a new gnome-terminal window..."
# The gnome-terminal command replaces the direct execution below
# cd "${ALPY_PROJECT_DIR}" || exit 1 # This is now handled by --working-directory
# pyenv exec python -m src.main # This is now run inside gnome-terminal

# Use single quotes for bash -c to simplify internal quoting.
gnome-terminal --working-directory="${ALPY_PROJECT_DIR}" \
               --title="Alpy Assistant" \
               -- bash -c '
                   echo "Starting Alpy application using pyenv environment \"${PYENV_ENV_NAME}\" ..."
                   pyenv exec python -m src.main
                   # --- Cleanup after Alpy finishes ---
                   echo
                   echo "----------------------------------------"
                   echo "Alpy application finished or was interrupted."
                   echo "Attempting to stop all llama-server processes using pkill..."
                   pkill -f llama-server
                   sleep 0.5 # Brief pause to allow processes to terminate
                   # Check if any remain and force kill if necessary
                   if pgrep -f llama-server > /dev/null; then
                       echo "Some llama-server processes might still be running. Attempting force kill..."
                       pkill -9 -f llama-server
                   fi
                   echo "Stop command sent."
                   echo "----------------------------------------"
                   # Execute an interactive bash session to keep the terminal open
                   exec bash -i
               ' bash

# The original script continues here, potentially finishing before the gnome-terminal
echo "---"
echo "New terminal launched. Original script finished."
echo "Background llama-server PID: ${LLAMA_SERVER_PID}"
echo "Log file: ${LLAMA_SERVER_LOG}"
