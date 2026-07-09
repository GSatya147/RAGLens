from langfuse import observe

from client.client import Client
from agentic_rag.state import AgentState
from agentic_rag.helpers.decomposition_context import decomposition_context_builder

client = Client()
@observe(name="query_decomposition")
def query_decomposition_node(state: AgentState) -> dict:
    context = decomposition_context_builder()
    response = client.generate_response(context, temperature=0)

    return {"sub_queries" : [response["content"]], "step_counter": state["step_counter"] + 1}

@observe(name="retrieval_node")
def retrieval_node(state: AgentState):
    pass

@observe(name="reasoning_node")
def reasoning_node(state: AgentState):
    pass

@observe(name="final_generation")
def generation_node(state: AgentState):
    pass

@observe(name="routing_logic")
def routing_logic(state: AgentState):
    decision: str = state["reasoning_steps"].decision

    if state["limiter"] > state["hop_count"] + 2:
        return "generate_node"

    if decision.lower() == "retrieve_more":
        return "retrieval_node"

    elif decision.lower() == "refine_query":
        return "query_decomposition_node"

    elif decision.lower() == "generate":
        return "generation_node"

    else:
        return "reasoning_node"

