#!/bin/bash
#
# Runs the main Structurally-Aware RAG pipeline for classification and explanation.
#
# This script iterates through specified models and cross-validation splits,
# applying a given RAG mode (e.g., cfg, ast) and k-value.
#
# Usage:
#   ./src/scripts/xrag.sh [OPTIONS] [MODEL_NAME_1] [MODEL_NAME_2] ...
#
# Options:
#   --mode [cfg|ast|ast_cfg]  The RAG retrieval mode to use (default: cfg).
#   --k K_VALUE               The number of contracts to retrieve (default: 3).
#   --split [1-5|all]         The CV split to run on (default: all).
#   -h, --help                Show this help message.
#
# Examples:
#   # Run the full pipeline with default settings (cfg, k=3, all splits, default models)
#   ./src/scripts/xrag.sh
#
#   # Run using 'ast' mode with k=5 for the 'o4-mini' model on all splits
#   ./src/scripts/xrag.sh --mode ast --k 5 o4-mini
#
#   # Run using 'cfg' mode for 'gpt-4o' on only the 2nd CV split
#   ./src/scripts/xrag.sh --split 2 gpt-4o

# --- Configuration ---
VENV_PATH=".venv"
PYTHON_SCRIPT="src/scripts/xrag.py"
DEFAULT_MODELS=("o4-mini" "gpt-4o")
DEFAULT_MODE="cfg"
DEFAULT_K=3
DEFAULT_SPLIT="all"
NUM_CV_SPLITS=5

# --- Script Logic ---
set -e # Exit immediately if a command exits with a non-zero status.

# --- Functions ---
print_usage() {
    echo "Usage: $0 [OPTIONS] [MODEL_NAME_1] [MODEL_NAME_2] ..."
    echo "Runs the main Structurally-Aware RAG pipeline."
    echo ""
    echo "Options:"
    echo "  --mode [cfg|ast|ast_cfg]  The RAG retrieval mode (default: $DEFAULT_MODE)."
    echo "  --k K_VALUE               The number of contracts to retrieve (default: $DEFAULT_K)."
    echo "  --split [1-5|all]         The CV split to run, or 'all' (default: $DEFAULT_SPLIT)."
    echo "  -h, --help                Show this help message."
    echo ""
    echo "If no models are specified, it defaults to: ${DEFAULT_MODELS[*]}"
}

# --- Argument Parsing ---
MODE="$DEFAULT_MODE"
K="$DEFAULT_K"
SPLIT_ARG="$DEFAULT_SPLIT"
MODELS_TO_RUN=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_usage
            exit 0
            ;;
        --mode) MODE="$2"; shift 2 ;;
        --k) K="$2"; shift 2 ;;
        --split) SPLIT_ARG="$2"; shift 2 ;;
        *)
            MODELS_TO_RUN+=("$1") # Save positional arg
            shift # Past argument
            ;;
    esac
done

if [ ${#MODELS_TO_RUN[@]} -eq 0 ]; then
    echo "[INFO] No models specified. Running default models: ${DEFAULT_MODELS[*]}"
    MODELS_TO_RUN=("${DEFAULT_MODELS[@]}")
fi

# Determine which splits to run
SPLITS_TO_RUN=()
if [[ "$SPLIT_ARG" == "all" ]]; then
    SPLITS_TO_RUN=($(seq 1 $NUM_CV_SPLITS))
else
    SPLITS_TO_RUN=($SPLIT_ARG)
fi

# --- Pre-run Checks ---
if [ ! -d "$VENV_PATH" ]; then echo "[ERROR] Virtual environment not found at '$VENV_PATH'."; exit 1; fi
if [ ! -f "$PYTHON_SCRIPT" ]; then echo "[ERROR] Python script not found at '$PYTHON_SCRIPT'."; exit 1; fi

# --- Execution ---
source "$VENV_PATH/bin/activate"
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

echo "[INFO] Starting RAG pipeline with settings:"
echo "  - Mode: $MODE"
echo "  - K-Value: $K"
echo "  - Models: ${MODELS_TO_RUN[*]}"
echo "  - Splits: ${SPLITS_TO_RUN[*]}"

for split_num in "${SPLITS_TO_RUN[@]}"; do
    DATASET_PATH="cv_splits/cv_split_${split_num}/{}" # Placeholder for train/test
    echo "======================================================"
    echo "[INFO] Processing Cross-Validation Split ${split_num}/${NUM_CV_SPLITS}"
    echo "======================================================"

    for MODEL in "${MODELS_TO_RUN[@]}"; do
        echo "[INFO] Running model: $MODEL"
        python3 "$PYTHON_SCRIPT" \
            --dataset-path "$DATASET_PATH" \
            --mode "$MODE" \
            --model-name "$MODEL" \
            --k "$K"

        echo "[INFO] Finished processing mode: $MODE with model: $MODEL on split $split_num."
        echo "------------------------------------------------------"
    done
done

echo "[SUCCESS] RAG pipeline completed successfully!"