import json, os

def assemble_trace(final_state: dict) -> dict:
    total_retrieval_latency_ms = sum(
        step["latency_ms"] for step in final_state["retrieval_steps"]
    )

    return {
        "question_id": final_state["question_id"],
        "config_id": final_state["config_id"],
        "hop_count": final_state["hop_count"],
        "original_query": final_state["original_query"],
        "sub_queries": final_state["sub_queries"],
        "retrieval_steps": final_state["retrieval_steps"],
        "reasoning_steps": final_state["reasoning_steps"],
        "final_answer": final_state["final_answer"],
        "total_retrieval_latency_ms": total_retrieval_latency_ms,
        "status": final_state["status"],
        "error": final_state["error"],
    }

def write_trace(trace: dict, output_dir: str) -> str:
    filename = f"traces_{trace['config_id']}.jsonl"
    path = os.path.join(output_dir, filename)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    return path

def read_trace(path: str) -> dict:
    traces = {}
    with open(path) as f:
        for line in f:
            t = json.loads(line)
            traces[t["question_id"]] = t 
    
    return traces