#!/bin/bash
set -e

ENV_DIR="env"

echo "Using START_DATE = $START_DATE"
echo "Using END_DATE = $END_DATE"

# --- Check/Create Virtual Environment ---
if [[ ! -d "$ENV_DIR" ]]; then
    echo "Virtual environment not found. Creating it in '$ENV_DIR'..."
    python3 -m venv "$ENV_DIR"

    echo "Virtual environment created. Activating it and installing requirements..."
    source "$ENV_DIR/bin/activate"

    if [[ -f "requirements.txt" ]]; then
        pip install --upgrade pip
        pip install -r "requirements.txt"
    else
        echo "WARNING: requirements.txt not found. Continuing without installing packages."
    fi
else
    echo "Virtual environment already exists."

    if [[ -z "$VIRTUAL_ENV" ]]; then
        echo "Activating virtual environment..."
        source "$ENV_DIR/bin/activate"
    else
        echo "Virtual environment already active: $VIRTUAL_ENV"
    fi
fi


printf "\n\n\nCREATE DB\n"
python3 01_create_schema.py

printf "\n\n\nPOPULATE DB\n"
python3 02_populate_db.py

printf "\n\n\nCREATE INDEX FOR RAG\n"
python3 03_create_rag_index.py

