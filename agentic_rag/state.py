from operator import add
from typing import Annotated, TypedDict, Literal, Optional

class RetrievalStep(TypedDict):
    step_number: int
    sub_query: str
    retrieved_chunks: list[dict]
    raw_scores: list[float]
    normalised_scores: list[float]  # based on rank/placement
    latency_ms: float

class ReasoningStep(TypedDict):
    step_number: int
    decision: Literal["retrieve_more", "refine_query", "generate"]
    reasoning_trace: str            # justification
    refine_reason: Optional[str]    # only for refine_query

class AgentState(TypedDict):
    # ids
    question_id: str
    config_id: Literal["naive", "advanced", "hybrid"]
    hop_count: int                  # GROUND TRUTH

    # working data
    original_query: str
    sub_queries: Annotated[list[str], add]
    retrieval_steps: Annotated[list[RetrievalStep], add]
    reasoning_steps: Annotated[list[ReasoningStep], add]
    accumulated_context: Annotated[str, add]

    # control
    step_counter: int
    reasoning_counter: int
    retrieval_counter: int
    refine_reason: Optional[str]    # reasoning -> decomposition

    # output
    final_answer: Optional[str]
    status: Literal["running", "completed", "failed", "insufficient_context"]
    error: Optional[str]

if __name__=="__main__":
    obj = ReasoningStep(refine_reason="hi")
    obj1 = AgentState(reasoning_steps=[obj])
    print(obj1["reasoning_steps"][-1].get("refine_reason"))