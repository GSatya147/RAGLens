from rank_bm25 import BM25Okapi

from db.db_client import execute_query

def build_bm25_index():
    all_chunks = execute_query(sql="SELECT chunk_id, question_id, hop_count, is_supporting, content FROM chunks;")
    tokenized_corpus = [c["content"].lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, all_chunks
