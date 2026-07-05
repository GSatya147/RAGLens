def dedupe_chunks(chunks_list: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for c in chunks_list:
        key = (c["question_id"], c["content"])
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    dropped = len(chunks_list) - len(deduped)
    print(f"Dropped {dropped} intra-question duplicate(s)")
    return deduped
