
def normalize_by_rank(raw_scores: list[float]) -> list[float]:
    """
    Rank 1 (best) -> 1.0
    Rank N (worst of this set) -> close to 0, but never exactly 0
    """
    n = len(raw_scores)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    return [1.0 - (rank / (n - 1)) for rank in range(n)]
