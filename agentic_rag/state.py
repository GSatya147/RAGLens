from typing import Annotated, TypedDict, Literal, Optional

from langgraph.graph.message import add_messages

class RetrievalStep(TypedDict):
    step_number: int
    sub_query: str
    retrieved_chunks: list[dict]
    raw_scores: list[float]
    noramlised_scores: list[float]  # based on rank/placement
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
    sub_queries: Annotated[list[str], add_messages]
    retrieval_steps: Annotated[list[RetrievalStep], add_messages]
    reasoning_steps: Annotated[list[ReasoningStep], add_messages]
    accumulated_context: str

    # control
    step_counter: int
    refine_reason: Optional[str]    # reasoning -> decomposition

    # output
    final_answer: Optional[str]
    status: Literal["running", "completed", "failed", "insufficient_context"]
    error: Optional[str]