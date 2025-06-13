import json
import os
import time
from typing import List, Optional, Any, Dict

from llama_index.core.llms import LLM
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.openai import OpenAI

from src.classes.utils.DebugLogger import DebugLogger
from src.prompts import RAG_PROMPT_TMPL, BASELINE_PROMPT_TMPL, \
    ANALYZE_RAG_COT_CONTEXT_SAFE_TMPL, BASELINE_COT_STEP1_TRIAGE_PROMPT_TMPL, \
    BASELINE_COT_STEP2_ANALYZE_FUNC_PROMPT_TMPL, \
    BASELINE_COT_STEP3_INTERACTION_PROMPT_TMPL, BASELINE_COT_STEP4_SYNTHESIS_PROMPT_TMPL, RAG_COT_PROMPT_TMPL, \
    ANALYZE_RAG_COT_CONTEXT_REENTRANT_TMPL, EVALUATE_EXPLANATION_PROMPT_TMPL
from src.pydantic_models import ContractAnalysis, Step1Output, Step2Output, Step3Output, ExplanationEvaluation


class LLMHandler:
    def __init__(self):
        self.logger = DebugLogger()
        model_name = os.getenv("MODEL_NAME")
        if not model_name:
            raise ValueError("MODEL_NAME environment variable not set.")
        self.logger.debug(f"Initializing LLMHandler with model '{model_name}'...")

        # Initialize a single LLM instance
        # LlamaIndex's LLM interface is unified, use it for both structured and unstructured calls
        if model_name in ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-2.5-flash-preview-05-20']:
            # Ensure you have GOOGLE_API_KEY set in your environment
            self.llm: LLM = GoogleGenAI(model_name=model_name, temperature=0.0)
        elif model_name in ['gpt-4o', 'gpt-4.1', 'o3-mini', 'o4-mini', "o3"]:
            # Ensure you have OPENAI_API_KEY set in your environment
            self.llm: LLM = OpenAI(model=model_name, temperature=0.0)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    def _retry_request(self, func, *args, max_retries=5, initial_wait=4, **kwargs):
        """
        Handles API rate limits and transient errors by retrying with exponential backoff.

        :param func: The function to retry (e.g., self.llm.complete, self.llm.structured_predict).
        :param args: Positional arguments for the function.
        :param max_retries: Maximum number of retries before failing.
        :param initial_wait: Initial wait time (seconds) before retrying.
        :param kwargs: Keyword arguments for the function.
        :return: The function's result or None if it fails after retries.
        :raises: The last exception if max retries are reached.
        """
        wait_time = initial_wait
        last_exception = None
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                # Check for specific API error types if needed (e.g., RateLimitError)
                self.logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {wait_time:.2f} seconds..."
                )
                # Optional: Log raw output here if possible, though difficult with structured_predict
                # if isinstance(e, (json.JSONDecodeError, pydantic.ValidationError)):
                #    self.logger.error("Potential malformed JSON output from LLM.")
                #    # Consider logging the prompt that caused the error
                #    # prompt_arg = args[1] if len(args) > 1 else kwargs.get('prompt') # Fragile way to get prompt
                #    # if prompt_arg: self.logger.warning(f"Prompt causing error: {prompt_arg}")

            time.sleep(wait_time)
            wait_time *= 2  # Exponential backoff

        self.logger.error(f"Max retries ({max_retries}) reached. Request failed.")
        if last_exception:
            raise last_exception  # Re-raise the last encountered exception
        return None  # Should not be reached if an exception occurred

    def analyze_contract(
            self,
            contract_source: str,
            similar_contexts: Optional[List[str]] = None,
            use_cot: bool = False
    ) -> Optional[dict[str, Any]]:
        """
        Classifies the security status of an input contract using the configured LLM.

        Uses structured_predict to directly output an ContractAnalysis object.

        :param contract_source: Source code of the input contract.
        :param similar_contexts: Optional list of security analyses of similar contracts.
        :return: Json representation of the ContractAnalysis object or None if analysis fails after retries.
        """
        self.logger.info("Analyzing input contract for classification.")

        if similar_contexts:
            prompt_template = RAG_COT_PROMPT_TMPL if use_cot else RAG_PROMPT_TMPL
            template_vars = {
                "contract_source": contract_source,
                "similar_contexts_str": "\n---\n".join(similar_contexts)  # Join contexts nicely
            }
            self.logger.debug("Using prompt template with context.")
        else:
            prompt_template = BASELINE_PROMPT_TMPL
            template_vars = {
                "contract_source": contract_source
            }
            self.logger.debug("Using prompt template without context.")

        try:
            # structured_predict takes the Pydantic class, the prompt template, and kwargs for formatting
            result = self._retry_request(
                self.llm.structured_predict,  # Pass the method itself
                ContractAnalysis,  # output_cls (as positional arg for _retry_request)
                prompt=prompt_template,  # Use keyword arg for clarity
                **template_vars  # Variables for the template
            )
            return result.model_dump_json()
        except Exception as e:
            self.logger.error(f"Failed to analyze contract after retries: {e}")
            return None

    def eval_explanation(
            self,
            contract_source: str,
            explanation: str
    ) -> Optional[dict[str, Any]]:
        """
        Evaluate the input contract's explanation using the configured LLM.

        :param contract_source: Source code of the input contract.
        :param explanation: Contract's explanation.
        :return: Json representation of the ExplanationEvaluation object or None if analysis fails after retries.
        """
        self.logger.info("Analyzing input contract for explanation.")

        prompt_template = EVALUATE_EXPLANATION_PROMPT_TMPL
        template_vars = {
            "contract_source": contract_source,
            "explanation_text_to_evaluate": explanation
        }

        try:
            # structured_predict takes the Pydantic class, the prompt template, and kwargs for formatting
            result = self._retry_request(
                self.llm.structured_predict,  # Pass the method itself
                ExplanationEvaluation,  # output_cls (as positional arg for _retry_request)
                prompt=prompt_template,  # Use keyword arg for clarity
                **template_vars  # Variables for the template
            )
            return result.model_dump_json()
        except Exception as e:
            self.logger.error(f"Failed to analyze contract after retries: {e}")
            return None

    def analyze_similar_contract(
            self,
            similar_source_code: str,
            label: str
    ) -> Optional[str]:
        """
        Analyzes a similar contract to generate descriptive text about its
        reentrancy status (why it's safe or how it's vulnerable).

        Uses the standard 'complete' method for free-form text generation.

        :param similar_source_code: Source code of the similar contract.
        :param label: Known label ('safe' or 'reentrant') of the contract.
        :return: Text analysis from the LLM or None if analysis fails after retries.
        """
        self.logger.info(f"Analyzing similar contract (Label: {label}).")
        label_lower = label.lower()

        if label_lower == "reentrant":
            prompt_template = ANALYZE_RAG_COT_CONTEXT_REENTRANT_TMPL
            template_vars = {"similar_source_code": similar_source_code}
        elif label_lower == "safe":
            prompt_template = ANALYZE_RAG_COT_CONTEXT_SAFE_TMPL
            template_vars = {"similar_source_code": similar_source_code}
        else:
            raise ValueError("Unknown label '{label}'.")

        # Format the prompt using the selected template and variables
        formatted_prompt = prompt_template.format(**template_vars)

        try:
            # Use the standard complete method for text generation
            response = self._retry_request(
                self.llm.complete,  # Pass the method itself
                formatted_prompt  # Pass the formatted prompt string
            )
            # .complete returns a CompletionResponse object, get the text
            return response.text + "\n\n --- source code: \n\n" + similar_source_code if response else None
        except Exception as e:
            self.logger.error(f"Failed to analyze similar contract after retries: {e}")
            return None

    # --- Chain-of-Thought (CoT) Methods Start Here ---

    def _cot_step1_triage(self, contract_source: str) -> List[str]:
        """[CoT Step 1] Identifies all functions with external calls."""
        self.logger.debug("CoT Step 1: Triage - Identifying functions with external calls.")
        prompt_template = BASELINE_COT_STEP1_TRIAGE_PROMPT_TMPL
        template_vars = {
            "contract_source": contract_source
        }

        try:
            response = self._retry_request(
                self.llm.structured_predict,  # Pass the method itself
                Step1Output,  # output_cls (as positional arg for _retry_request)
                prompt=prompt_template,
                **template_vars
            )
            if not response: return []
            data = json.loads(response.model_dump_json())
            functions = data.get("functions_to_analyze", [])
            self.logger.info(f"CoT Step 1: Found {len(functions)} functions to analyze: {functions}")
            return functions
        except (json.JSONDecodeError, AttributeError) as e:
            self.logger.error(f"CoT Step 1 (Triage): Failed to decode JSON response. Error: {e}")
            return []

    def _cot_step2_analyze_functions(self, contract_source: str, functions_to_analyze: List[str]) -> List[
        Dict[str, Any]]:
        """[CoT Step 2] Performs a deep-dive analysis on each identified function."""
        self.logger.debug(f"CoT Step 2: Analyzing {len(functions_to_analyze)} functions individually.")
        all_analyses = []
        for func_name in functions_to_analyze:
            prompt_template = BASELINE_COT_STEP2_ANALYZE_FUNC_PROMPT_TMPL
            template_vars = {
                "contract_source": contract_source,
                "function_name": func_name
            }
            try:
                response = self._retry_request(
                    self.llm.structured_predict,  # Pass the method itself
                    Step2Output,  # output_cls (as positional arg for _retry_request)
                    prompt=prompt_template,
                    **template_vars
                )
                if response:
                    analysis = json.loads(response.model_dump_json())
                    all_analyses.append(analysis)
            except (json.JSONDecodeError, AttributeError) as e:
                self.logger.error(f"CoT Step 2: Failed to decode JSON for function '{func_name}'. Error: {e}")
        return all_analyses

    def _cot_step3_analyze_interactions(self, contract_source: str, function_analyses: List[Dict[str, Any]]) -> Dict[
        str, Any]:
        """[CoT Step 3] Analyzes for cross-function reentrancy exploits."""
        self.logger.debug("CoT Step 3: Analyzing for cross-function exploits.")
        analyses_json_str = json.dumps(function_analyses, indent=2)
        prompt_template = BASELINE_COT_STEP3_INTERACTION_PROMPT_TMPL
        template_vars = {
            "contract_source": contract_source,
            "analyses_json_str": analyses_json_str
        }
        try:
            response = self._retry_request(
                self.llm.structured_predict,  # Pass the method itself
                Step3Output,  # output_cls (as positional arg for _retry_request)
                prompt=prompt_template,
                **template_vars
            )
            return json.loads(response.model_dump_json()) if response else {}
        except (json.JSONDecodeError, AttributeError) as e:
            self.logger.error(f"CoT Step 3 (Interaction Analysis): Failed to decode JSON. Error: {e}")
            return {}

    def _cot_step4_synthesize_report(self, contract_source: str, function_analyses: List[Dict[str, Any]],
                                     interaction_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """[CoT Step 4] Consolidates all findings into a final report."""
        self.logger.debug("CoT Step 4: Synthesizing the final report.")
        func_analyses_str = json.dumps(function_analyses, indent=2)
        inter_analysis_str = json.dumps(interaction_analysis, indent=2)
        prompt_template = BASELINE_COT_STEP4_SYNTHESIS_PROMPT_TMPL
        template_vars = {
            "contract_source": contract_source,
            "func_analyses_str": func_analyses_str,
            "inter_analysis_str": inter_analysis_str
        }

        try:
            response = self._retry_request(
                self.llm.structured_predict,  # Pass the method itself
                ContractAnalysis,  # output_cls (as positional arg for _retry_request)
                prompt=prompt_template,
                **template_vars
            )
            return response.model_dump_json() if response else {}
        except (json.JSONDecodeError, AttributeError) as e:
            self.logger.error(f"CoT Step 4 (Synthesis): Failed to decode final JSON report. Error: {e}")
            return {}

    def analyze_contract_with_cot(self, contract_source: str) -> str:
        """
        Orchestrates the full multi-step CoT audit of a smart contract.
        """
        self.logger.info("Starting contract analysis with Chain-of-Thought (CoT) workflow.")

        # Step 1: Triage
        functions_to_analyze = self._cot_step1_triage(contract_source)
        if not functions_to_analyze:
            self.logger.warning("CoT workflow terminated: No functions with external calls found.")
            # Return a 'Safe' report if no external calls are found.
            return ContractAnalysis(
                classification="Safe",
                explanation="The contract is classified as 'Safe' because the initial triage found no functions containing external calls, which is the primary vector for reentrancy attacks."
            ).model_dump_json()

        # Step 2: Per-Function Analysis
        function_analyses = self._cot_step2_analyze_functions(contract_source, functions_to_analyze)

        # Step 3: Cross-Function Analysis
        interaction_analysis = self._cot_step3_analyze_interactions(contract_source, function_analyses)

        # Step 4: Final Synthesis
        final_report = self._cot_step4_synthesize_report(contract_source, function_analyses, interaction_analysis)

        self.logger.info("CoT analysis workflow completed.")
        return final_report
