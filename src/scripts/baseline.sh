#!/bin/bash
#
# Runs the baseline (non-RAG) model evaluation pipeline.
# This script iterates through all cross-validation splits for a given set of models.
#
# Usage:
#   ./src/scripts/baseline.sh [MODEL_NAME_1] [MODEL_NAME_2] ...
#
# Examples:
#   # Run evaluation for the 'o3-mini' model on all CV splits
#   ./src/scripts/baseline.sh o3-mini
#
#   # Run evaluation for 'o3-mini' and 'gpt-4o' on all CV splits
#   ./src/scripts/baseline.sh o3-mini gpt-4o
#
# If no models are provided, it will run a default set.

# --- Configuration ---
VENV_PATH=".venv"
PYTHON_SCRIPT="src/scripts/baseline.py"
NUM_CV_SPLITS=5 # Total number of cross-validation splits in the dataset
DEFAULT_MODELS=("o3-mini" "gpt-4o") # Models to run if none are provided as arguments

# --- Script Logic ---
set -e # Exit immediately if a command exits with a non-zero status.

# --- Functions ---
print_usage() {
    echo "Usage: $0 [MODEL_NAME_1] [MODEL_NAME_2] ..."
    echo "Runs the baseline evaluation for the specified models across all $NUM_CV_SPLITS cross-validation splits."
    echo "If no models are specified, it defaults to: ${DEFAULT_MODELS[*]}"
}

# --- Argument Parsing ---
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    print_usage
    exit 0
fi

MODELS_TO_RUN=("$@")
if [ ${#MODELS_TO_RUN[@]} -eq 0 ]; then
    echo "[INFO] No models specified. Running default models: ${DEFAULT_MODELS[*]}"
    MODELS_TO_RUN=("${DEFAULT_MODELS[@]}")
fi

# --- Pre-run Checks ---
# Ensure the virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "[ERROR] Virtual environment not found at '$VENV_PATH'. Please create it first."
    exit 1
fi

# Ensure the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "[ERROR] Python script not found at '$PYTHON_SCRIPT'."
    exit 1
fi

# --- Execution ---
# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Add the 'src' directory to PYTHONPATH to resolve local module imports
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

echo "[INFO] Starting baseline evaluation for models: ${MODELS_TO_RUN[*]}"

# Loop through each cross-validation split
for i in $(seq 1 $NUM_CV_SPLITS); do
    DATASET_PATH="cv_splits/cv_split_${i}/test"
    echo "======================================================"
    echo "[INFO] Processing Cross-Validation Split ${i}/${NUM_CV_SPLITS} (Path: $DATASET_PATH)"
    echo "======================================================"

    # Loop through each specified OpenAI model
    for MODEL in "${MODELS_TO_RUN[@]}"; do
        echo "[INFO] Running model: $MODEL"

        # Run the Python script with model name argument
        python3 "$PYTHON_SCRIPT" \
            --dataset-path "$DATASET_PATH" \
            --model-name "$MODEL"

        echo "[INFO] Finished processing with model: $MODEL."
        echo "------------------------------------------------------"
    done
done

echo "[SUCCESS] All baseline evaluations completed successfully!"