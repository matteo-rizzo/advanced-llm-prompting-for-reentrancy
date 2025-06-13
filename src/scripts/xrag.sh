#!/bin/bash

# Define paths
VENV_PATH=".venv"
PYTHON_SCRIPT="src/scripts/xrag.py"

# Ensure the virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "[ERROR] Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Ensure the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "[ERROR] Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Add the 'src' directory to PYTHONPATH
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Modes to run
MODES=("cfg")

# OpenAI Model Names (modify these as needed)
MODELS=("gemini-2.5-flash-preview-05-20")

# Dataset path template
DATASET_PATHS=("cv_splits/cv_split_3/{}")

# Loop through each dataset path
for DATASET_PATH in "${DATASET_PATHS[@]}"; do

  # Loop through each OpenAI model
  for MODEL in "${MODELS[@]}"; do
      echo "[INFO] Using LLM model: $MODEL"

      # Loop through each mode
      for MODE in "${MODES[@]}"; do
          echo "[INFO] Running contract analysis in mode: $MODE..."

          # Run the Python script with model name argument
          python3 "$PYTHON_SCRIPT" \
              --dataset-path "$DATASET_PATH" \
              --mode "$MODE" \
              --model-name "$MODEL" \
              --k 3 \

          # Log completion

          echo "[INFO] Finished processing mode: $MODE with model: $MODEL."
          echo "------------------------------------------------------"
      done
  done
done

echo "[INFO] All modes executed successfully!"
