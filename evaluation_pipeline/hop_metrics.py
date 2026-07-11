def compute_hop_metrics(trace: dict) -> list[dict]:
    results = []

    for step in trace["retrieval_steps"]:
        top_3 = step["retrieved_chunks"][:3]

        supporting_flags = [
            c["metadata"]["is_supporting"] and c["metadata"]["question_id"] == trace["question_id"]
            for c in top_3
        ]
        num_supporting = sum(supporting_flags)

        rank = next(
            (i + 1 for i, is_sup in enumerate(supporting_flags) if is_sup),
            None,
        )

        results.append({
            "step_number": step["step_number"],
            "context_precision_at_3": num_supporting / 3,
            "context_recall": 1.0 if num_supporting > 0 else 0.0,
            "supporting_chunk_rank": rank,
        })

    return results
