from typing import List, Literal

from pydantic import BaseModel, Field, conint


class ContractAnalysis(BaseModel):
    classification: str = Field(
        ...,
        description="The classification label indicating whether the contract is 'reentrant' or 'safe'."
    )
    explanation: str = Field(
        ...,
        description="A detailed explanation for the classification, citing patterns in the contract as evidence."
    )


# --- Model for BASELINE_COT_STEP1_TRIAGE_PROMPT_TMPL ---

class Step1Output(BaseModel):
    """
    Validates the output of the first step (triage), which identifies all
    functions containing external calls within a Solidity contract.
    """
    functions_to_analyze: List[str] = Field(
        ...,
        description="An array of function names that contain an external call."
    )


# --- Models for BASELINE_COT_STEP2_ANALYZE_FUNC_PROMPT_TMPL ---

class FunctionAnalysis(BaseModel):
    """
    A detailed security analysis for a single function regarding
    reentrancy vulnerabilities.
    """
    has_external_call: bool = Field(
        ...,
        description="True if the function contains an external call."
    )
    follows_cei_pattern: bool = Field(
        ...,
        description="True if state changes occur before external calls (Checks-Effects-Interactions)."
    )
    has_reentrancy_guard: bool = Field(
        ...,
        description="True if the function is protected by a reentrancy guard like a 'nonReentrant' modifier."
    )
    is_vulnerable_in_isolation: bool = Field(
        ...,
        description="True if re-entering this specific function could cause harm, assuming it's the only public/external entry point."
    )
    reasoning: str = Field(
        ...,
        description="A brief explanation supporting the analysis findings for the function, citing relevant line numbers."
    )


class Step2Output(BaseModel):
    """
    Validates the output of the second step (function analysis), which provides
    a detailed reentrancy analysis for a single contract function.
    """
    function_name: str = Field(
        ...,
        description="The name of the function being analyzed."
    )
    analysis: FunctionAnalysis


# --- Models for BASELINE_COT_STEP3_INTERACTION_PROMPT_TMPL ---

class CrossFunctionAnalysis(BaseModel):
    """
    Analysis of potential cross-function reentrancy attack vectors.
    """
    is_exploitable: bool = Field(
        ...,
        description="True if a plausible cross-function reentrancy attack path exists."
    )
    exploit_scenario: str = Field(
        ...,
        description="A step-by-step description of the attack path. If no path exists, this will state that none were identified."
    )


class Step3Output(BaseModel):
    """
    Validates the output of the third step (interaction analysis), which
    determines if a cross-function reentrancy attack is possible.
    """
    cross_function_analysis: CrossFunctionAnalysis


# --- Model for BASELINE_COT_STEP4_SYNTHESIS_PROMPT_TMPL ---

class Step4Output(BaseModel):
    """
    Validates the final synthesis report, providing a conclusive assessment
    of the contract's vulnerability to reentrancy.
    """
    classification: Literal["Reentrant", "Safe"] = Field(
        ...,
        description="The final classification of the contract. 'Reentrant' if any vulnerability was found, otherwise 'Safe'."
    )
    explanation: str = Field(
        ...,
        description="A valid, escaped JSON string that summarizes the overall findings from the analysis."
    )


# --- Explanation Evaluation ---

class CriterionEvaluation(BaseModel):
    """
    Represents the evaluation of a single criterion (Correctness, Informativeness, Pertinence).
    """
    rating: conint(ge=1, le=5) = Field(
        ...,
        description="Quantitative rating on a scale of 1 to 5 (1=Poor, 5=Excellent)."
    )
    justification: str = Field(
        ...,
        description="Detailed justification for the rating, citing specific examples from both the Explanation Text and the Source Code."
    )


class ExplanationEvaluation(BaseModel):
    """
    Contains the evaluation results for all three criteria.
    """
    correctness: CriterionEvaluation = Field(
        ...,
        description="Evaluation of the factual accuracy of the explanation."
    )
    informativeness: CriterionEvaluation = Field(
        ...,
        description="Evaluation of the extent to which the explanation furnished meaningful insights."
    )
    pertinence: CriterionEvaluation = Field(
        ...,
        description="Evaluation of the relevance and specificity of the information provided to reentrancy."
    )
