

def rrf_fuse(dense_results: list[dict], sparse_results: list[dict], k: int = 60) -> list[dict]:
    scores = {}
    combined_lookup = {}

    for rank, row in enumerate(dense_results):
        cid = row["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        combined_lookup[cid] = row

    for rank, row in enumerate(sparse_results):
        cid = row["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        combined_lookup[cid] = row

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{**combined_lookup[cid], "rrf_score": score} for cid, score in fused]