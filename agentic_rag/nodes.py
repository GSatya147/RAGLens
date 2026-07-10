import time

from langfuse import observe

from client.client import Client
from agentic_rag.state import AgentState, ReasoningStep, RetrievalStep
from agentic_rag.helpers.decomposition_context import build_decomposition_messages
from agentic_rag.helpers.reasoning_context import build_reasoning_messages
from agentic_rag.helpers.final_context import build_generation_messages
from retrieval.pipeline import run_pipeline

client = Client()
@observe(name="query_decomposition")
def query_decomposition_node(state: AgentState) -> dict:
    refine_reason = state["refine_reason"] if state["refine_reason"] else None
    context = build_decomposition_messages(original_query=state["original_query"], step_counter=state["step_counter"], accumulated_context=state["accumulated_context"], refine_reason=refine_reason)
    response = client.generate_response(context, temperature=0)

    return {"sub_queries" : [response["content"]], "step_counter": state["step_counter"] + 1}

@observe(name="retrieval_node")
def retrieval_node(state: AgentState):
    t0 = time.time()
    nodes = run_pipeline(user_query=state["sub_queries"][-1], retriever=state["config_id"])
    t1= time.time()

    retrieved_chunks: list[dict] = []
    scores: list[float] = []
    accumulated_context: str = ""
    for node in nodes:
        accumulated_context += f"{node.node.text}\n\n"
        retrieved_chunks.append({"text": node.node.text, "metadata": node.node.metadata})
        scores.append(node.score)

    field = RetrievalStep(
        step_number=state["retrieval_counter"] + 1,
        sub_query=state["sub_queries"][-1],
        retrieved_chunks=retrieved_chunks,
        raw_scores=scores,
        noramlised_scores=scores,
        latency_ms=t1-t0
    )
    
    return {"retrieval_steps" : [field],  "accumulated_context" : accumulated_context, "retrieval_counter" : state["retrieval_counter"] + 1, "step_counter" : state["step_counter"] + 1}

@observe(name="reasoning_node")
def reasoning_node(state: AgentState) -> dict:
    context = build_reasoning_messages(original_query=state["original_query"], accumulated_context=state["accumulated_context"], step_counter=state["step_counter"])
    response = client.generate_response(context, temperature=0)
    content = response["content"]

    field = ReasoningStep(
        step_number     = state["reasoning_counter"] + 1,
        decision        = content["decision"],
        reasoning_trace = content["reasoning"],
        refine_reason   = content.get("refine_reason")
    )

    return {"reasoning_steps" : [field], "refine_reason" : content["refine_reason"], "reasoning_counter" : state["reasoning_counter"] + 1, "step_counter" : state["step_counter"] + 1} 

@observe(name="final_generation")
def generation_node(state: AgentState):
    context = build_generation_messages(original_query=state["original_query"], accumulated_context=state["accumulated_context"])
    response = client.generate_response(context, temperature=0)

    return {"final_answer" : response["content"], "status": "insufficient_context" if response["content"].strip().lower() == "insufficient context" else "completed", "step_counter" : state["step_counter"] + 1}

@observe(name="routing_logic")
def routing_logic(state: AgentState):
    decision: str = state["reasoning_steps"][-1]["decision"]

    if state["reasoning_counter"] > state["hop_count"] + 2:
        return "generate_node"

    if decision.lower() == "retrieve_more":
        return "retrieval_node"

    elif decision.lower() == "refine_query":
        return "query_decomposition_node"

    elif decision.lower() == "generate":
        return "generation_node"

    else:
        return "generation_node"

