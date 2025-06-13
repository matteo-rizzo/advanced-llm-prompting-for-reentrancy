import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from src.classes.utils.DebugLogger import DebugLogger
from src.classes.utils.EnvLoader import EnvLoader
from src.classes.xrag.LLMHandler import LLMHandler

# Load environment configuration.
try:
    EnvLoader(env_dir="src").load_env_files()
except Exception as e:
    print(f"Error loading environment configuration: {e}")
    exit(1)

logger = DebugLogger()
LOGS_BASE_DIR = Path("logs", "explanation_eval")


def load_and_filter_contracts(path_to_contracts: Path) -> list[Path]:
    """
    Loads and filters Solidity contract file paths with better error handling.
    """
    if not path_to_contracts.is_dir():
        logger.error(f"Directory not found: {path_to_contracts}")
        return []
    try:
        files = [f for f in path_to_contracts.iterdir() if f.is_file() and f.name.endswith(".sol")]
        return sorted(files)
    except OSError as e:
        logger.error(f"Error accessing directory {path_to_contracts}: {e}")
        return []


def process_contract(path_to_file: Path, gt_category: str, llm: LLMHandler, log_dir: Path) -> bool | dict[
    str, Any | None]:
    """
    Processes a single contract file with improved error handling.
    Returns True if classification is correct, False otherwise.
    """
    filename = path_to_file.name
    try:
        contract_content = path_to_file.read_text(encoding='latin-1')
    except UnicodeDecodeError as e:
        logger.error(f"Error decoding file {path_to_file} with latin-1: {e}. Trying utf-8.")
        try:
            contract_content = path_to_file.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Error reading file {path_to_file}: {e}")
            return False
    except Exception as e:
        logger.error(f"Error reading file {path_to_file}: {e}")
        return False

    logger.debug(f"Processing file: {filename}")

    filename = filename.replace(".sol", ".json")
    contract_id = filename.replace(".json", "")
    cv = f"cv_{path_to_file.parts[1][-1]}"
    model_name = os.environ["MODEL_TO_EVAL"]
    path_to_explanation = f"explanations/prompting/baseline/{cv}/{model_name}/{gt_category}/{filename}"
    #path_to_explanation = f"explanations/prompting/rag_cot/{cv}/{model_name}/cfg/{gt_category}/{contract_id}/classification.json"
    explanation_text_to_evaluate = json.load(open(path_to_explanation))["explanation"]

    try:
        llm_response = llm.eval_explanation(contract_content, explanation_text_to_evaluate)
        if llm_response:
            answer = llm_response
        else:
            logger.error(f"Empty LLM response for file {filename}: {llm_response}")
            return False
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON response for file {filename}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error generating completion for file {filename}: {e}")
        return False

    output_path = log_dir / f"{filename.split('.')[0]}.json"
    try:
        output_path.write_text(json.dumps(answer, indent=4, ensure_ascii=True), encoding='utf-8')
    except OSError as e:
        logger.error(f"Error writing output file {output_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error writing output file {output_path}: {e}")

    return answer


def evaluate(path_to_contracts: Path, model_name: str = None) -> None:
    """
    Evaluates Solidity contracts in the given directory with better error handling.
    """
    if not path_to_contracts.is_dir():
        logger.error(f"Directory not found: {path_to_contracts}")
        return

    gt_category = path_to_contracts.name
    log_dir = LOGS_BASE_DIR / f"{model_name}_{time.strftime('%Y%m%d')}" / f"{gt_category}"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Error creating log directory {log_dir}: {e}")
        return

    solidity_files = load_and_filter_contracts(path_to_contracts)
    total_files = len(solidity_files)

    if total_files == 0:
        logger.warning(f"No Solidity (.sol) files found in {path_to_contracts}.")
        return

    logger.info(f"Evaluating explanation of {model_name} on {total_files} files from category: {gt_category}")
    logger.info(f"Results will be logged at: {log_dir}")

    llm = LLMHandler()
    correct = 0
    for index, path_to_file in enumerate(solidity_files, start=1):
        results = process_contract(path_to_file, gt_category, llm, log_dir)
        if not results:
            logger.warning(f"Unable to evaluate file {path_to_file}")
        else:
            logger.debug(f"Evaluated file {path_to_file}: {results}")
        logger.info(f"Processed {index}/{total_files} files.")


def main(args) -> None:
    """
    Main function to evaluate test datasets with improved error handling.
    """
    dataset_path = Path(args.dataset_path)
    path_to_reentrant = dataset_path / "source" / "reentrant"
    path_to_safe = dataset_path / "source" / "safe"

    os.environ["MODEL_NAME"] = args.evaluator
    os.environ["MODEL_TO_EVAL"] = args.model_name

    evaluate(path_to_reentrant, args.model_name)
    evaluate(path_to_safe, args.model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Contract Analysis CLI for analyzing manually verified contracts' explanations.")
    parser.add_argument("--dataset-path", type=str, default="dataset", help="Base path for the dataset.")
    parser.add_argument("--model-name", type=str, required=True, help="OpenAI or Google model name to evaluate.")
    parser.add_argument("--evaluator", type=str, default="o4-mini", help="Evaluator OpenAI or Google model name.")
    args = parser.parse_args()
    main(args)
