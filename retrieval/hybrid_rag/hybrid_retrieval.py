from typing import List
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode

from db.db_client import execute_query
from retrieval.query_embedder import query_embedder
from retrieval.advanced_rag.reranker import Reranker
from retrieval.hybrid_rag.rrf import rrf_fuse
from retrieval.hybrid_rag.bm25 import build_bm25_index

class HybridRetriever(BaseRetriever):
    """Config 3: dense + BM25 (Postgres tsvector) fused via RRF -> cross-encoder rerank -> top 5."""

    def __init__(self, dense_top_k=10, sparse_top_k=10, final_top_k=5, bm25=None, bm25_corpus=None):
        self._dense_top_k = dense_top_k
        self._sparse_top_k = sparse_top_k
        self._final_top_k = final_top_k
        self.reranker_obj = Reranker()
        self._bm25, self._bm25_corpus = (bm25, bm25_corpus) if bm25 else build_bm25_index()
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_text = query_bundle.query_str

        # 1. Dense retrieval (same as Naive/Advanced)
        query_embedding = query_embedder(query_text)

        dense_sql = """
            SELECT chunk_id, question_id, hop_count, is_supporting, content,
                   1 - (embedding <=> %s::vector) AS score
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        dense_results = execute_query(
            sql=dense_sql, params=(query_embedding, query_embedding, self._dense_top_k)
        )

        # 2. Sparse 
        tokenized_query = query_text.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[::-1][:self._sparse_top_k]
        sparse_results = [
            {**self._bm25_corpus[i], "score": scores[i]} for i in top_indices
        ]

        # 3. RRF fusion, combine both ranked lists by position, not raw score
        fused = rrf_fuse(dense_results, sparse_results)

        # 4. Cross-encoder rerank on the fused candidate set -> final top 5
        candidate_texts = [c["content"] for c in fused]
        rerank_result = self.reranker_obj.reranker(query_text=query_text, candidate_texts=candidate_texts, top_k=self._final_top_k)

        nodes_with_scores = []
        for r in rerank_result.results:
            original = fused[r.index]
            node = TextNode(
                text=original["content"],
                metadata={
                    "chunk_id": original["chunk_id"],
                    "question_id": original["question_id"],
                    "hop_count": original["hop_count"],
                    "is_supporting": original["is_supporting"],
                }
            )
            nodes_with_scores.append(NodeWithScore(node=node, score=r.relevance_score))

        return nodes_with_scores

if __name__=="__main__":
    obj = HybridRetriever()
    nodes = obj.retrieve("What date saw the writing of the song where the devil went down to the state where WDXQ is located?")

    for node in nodes:
        print(f"text: {node.node.text}, score: {node.score}\n")