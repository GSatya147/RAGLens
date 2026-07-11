from agentic_rag.graph import AgentGraph
from agentic_rag.tracer import assemble_trace, write_trace
from configs.config import TRACE_DIR

def run_agent_pipeline(question_id: str, hop_count: int, user_query: str, config: str):
    initial_state = {
        # ids 
        "question_id": question_id,
        "config_id": config,          
        "hop_count": hop_count,                

        # working data
        "original_query": user_query,
        "sub_queries": [],
        "retrieval_steps": [],
        "reasoning_steps": [],
        "accumulated_context": "",

        # control
        "step_counter": 0,
        "reasoning_counter": 0,
        "retrieval_counter": 0,
        "refine_reason": None,

        # output
        "final_answer": None,
        "status": "running",
        "error": None,
    }
    graph = AgentGraph()
    result = graph.invoke_graph(initial_state=initial_state, question_id=initial_state["question_id"])

    trace: dict = assemble_trace(final_state=result)
    path = write_trace(trace=trace, output_dir=TRACE_DIR)
    return result

if __name__=="__main__":
    result = run_agent_pipeline("What date saw the writing of the song where the devil went down to the state where WDXQ is located?")
    print(result)