from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from agentic_rag.state import AgentState
from agentic_rag.nodes import query_decomposition_node, retrieval_node, reasoning_node, generation_node, routing_logic

class AgentGraph:
    def __init__(self):
        self.graph = StateGraph(AgentState)
        self.checkpointer = InMemorySaver()

        # add nodes
        self.graph.add_node("query_decomposition_node", query_decomposition_node)
        self.graph.add_node("retrieval_node", retrieval_node)
        self.graph.add_node("reasoning_node", reasoning_node)
        self.graph.add_node("generation_node", generation_node)

        # add edges
        self.graph.add_edge(START, "query_decomposition_node")
        self.graph.add_edge("query_decomposition_node", "retrieval_node")
        self.graph.add_edge("retrieval_node", "reasoning_node")
        self.graph.add_edge("generation_node", END)
        
        # add conditional edges
        self.graph.add_conditional_edges("reasoning_node", routing_logic)

        self.graph = self.graph.compile(checkpointer=self.checkpointer)

    def invoke_graph(self, initial_state: dict, question_id: str) -> dict:
        config = {
            "configurable" : {
                "thread_id" : f"thread_{question_id}"
            }
        }

        try:
            self.graph.invoke(initial_state, config=config)

        except Exception as e:
            partial = self.graph.get_state(config=config)
            result = dict(partial.values)
            result["status"] = "failed"
            result["error"] = {"exception_type": type(e).__name__, "message": str(e)}
            return result

        return dict(self.graph.get_state(config=config).values)
        