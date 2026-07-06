import numpy as np

def mmr_select(query_embedding, candidates, lambda_param=0.5, top_k=5):
    "lambda * sim(query, doc) - (1-lambda) * max_sim(doc, already_selected)"
    query_vec = np.array(query_embedding)
    selected = []
    remaining = candidates.copy()

    while remaining and len(selected) < top_k:
        best_score = -float("inf")
        best_candidate = None

        for cand in remaining:
            cand_vec = np.array(cand["embedding"])
            relevance = np.dot(query_vec, cand_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(cand_vec)
            )
            if selected:
                diversity_penalty = max(
                    np.dot(cand_vec, np.array(s["embedding"])) /
                    (np.linalg.norm(cand_vec) * np.linalg.norm(s["embedding"]))
                    for s in selected
                )
            else:
                diversity_penalty = 0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty

            if mmr_score > best_score:
                best_score = mmr_score
                best_candidate = cand

        selected.append(best_candidate)
        remaining.remove(best_candidate)

    return selected