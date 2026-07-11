import json
import time

from langfuse import observe

from client.client import Client
from agentic_rag.state import AgentState, ReasoningStep, RetrievalStep
from agentic_rag.helpers.decomposition_context import build_decomposition_messages
from agentic_rag.helpers.reasoning_context import build_reasoning_messages
from agentic_rag.helpers.final_context import build_generation_messages
from agentic_rag.helpers.normaliser import normalize_by_rank
from retrieval.pipeline import run_pipeline

client = Client()

@observe(name="query_decomposition")
def query_decomposition_node(state: AgentState) -> dict:
    refine_reason = state["refine_reason"] if state["refine_reason"] else None
    context = build_decomposition_messages(original_query=state["original_query"], step_counter=state["step_counter"], accumulated_context=state["accumulated_context"], refine_reason=refine_reason, previous_sub_queries=state["sub_queries"])
    response = client.generate_response(context, temperature=0)

    return {"sub_queries" : [response["content"]], "step_counter": state["step_counter"] + 1}

@observe(name="retrieval_node")
def retrieval_node(state: AgentState):
    t0 = time.time()
    nodes = run_pipeline(user_query=state["sub_queries"][-1], retriever=state["config_id"])
    t1 = time.time()

    seen_chunk_ids: set = set(state.get("seen_chunk_ids", []))

    retrieved_chunks: list[dict] = []
    scores: list[float] = []
    new_context_pieces: list[str] = []
    newly_seen_ids: list = []

    for node in nodes:
        chunk_id = node.node.metadata.get("chunk_id")
        retrieved_chunks.append({"text": node.node.text, "metadata": node.node.metadata})
        scores.append(node.score)

        # Only append new chunks to the context passed to reasoning/generation.
        if chunk_id not in seen_chunk_ids:
            new_context_pieces.append(f"{node.node.text}\n\n")
            newly_seen_ids.append(chunk_id)

    accumulated_context: str = "".join(new_context_pieces)

    field = RetrievalStep(
        step_number=state["retrieval_counter"] + 1,
        sub_query=state["sub_queries"][-1],
        retrieved_chunks=retrieved_chunks,
        raw_scores=scores,
        normalised_scores=normalize_by_rank(scores),
        latency_ms=1000*(t1-t0)
    )

    return {
        "retrieval_steps": [field],
        "accumulated_context": accumulated_context,
        "seen_chunk_ids": newly_seen_ids,
        "retrieval_counter": state["retrieval_counter"] + 1,
        "step_counter": state["step_counter"] + 1
    }

@observe(name="reasoning_node")
def reasoning_node(state: AgentState) -> dict:
    context = build_reasoning_messages(original_query=state["original_query"], accumulated_context=state["accumulated_context"], step_counter=state["step_counter"])
    response = client.generate_response(context, temperature=0)
    try:
        content = json.loads(response["content"])
    except json.JSONDecodeError as e:
        content = {"decision": "generate", "reasoning": f"JSON parse failure: {e}", "refine_reason": None}

    field = ReasoningStep(
        step_number     = state["reasoning_counter"] + 1,
        decision        = content["decision"],
        reasoning_trace = content["reasoning"],
        refine_reason   = content.get("refine_reason")
    )

    return {"reasoning_steps" : [field], "refine_reason" : content.get("refine_reason"), "reasoning_counter" : state["reasoning_counter"] + 1, "step_counter" : state["step_counter"] + 1} 

@observe(name="final_generation")
def generation_node(state: AgentState):
    context = build_generation_messages(original_query=state["original_query"], accumulated_context=state["accumulated_context"])
    response = client.generate_response(context, temperature=0)

    return {"final_answer" : response["content"], "status": "insufficient_context" if response["content"].strip().lower() == "insufficient context" else "completed", "step_counter" : state["step_counter"] + 1}

@observe(name="routing_logic")
def routing_logic(state: AgentState):
    decision: str = state["reasoning_steps"][-1]["decision"]

    if state["reasoning_counter"] > state["hop_count"] + 2:
        return "generation_node"

    if decision.lower() == "retrieve_more":
        return "query_decomposition_node"

    elif decision.lower() == "refine_query":
        return "query_decomposition_node"

    elif decision.lower() == "generate":
        return "generation_node"

    else:
        return "generation_node"

