import numpy as np
from typing import List
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
import voyageai

from db.db_client import execute_query
from retrieval.advanced_rag.mmr import mmr_select
from retrieval.advanced_rag.reranker import Reranker
from retrieval.query_embedder import query_embedder


class AdvancedRetriever(BaseRetriever):
    """Config 2: dense (top_k=10) -> MMR diversity selection -> cross-encoder rerank -> top 5."""

    def __init__(self, dense_top_k=10, mmr_lambda=0.5, final_top_k=5):
        self._dense_top_k = dense_top_k
        self._mmr_lambda = mmr_lambda
        self.top_k = final_top_k
        self.reranker_obj = Reranker()
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_text = query_bundle.query_str

        query_embedding = query_embedder(query_text)

        # 2. Dense retrieval — pull WIDER candidate pool (top 10), fetch embedding
        #    too since MMR needs candidate vectors, not just text
        sql = """
            SELECT chunk_id, question_id, hop_count, is_supporting, content, embedding,
                   1 - (embedding <=> %s::vector) AS similarity_score
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        rows = execute_query(
            sql=sql, params=(query_embedding, query_embedding, self._dense_top_k)
        )

        # 3. MMR — reorder/select for diversity from the 10 candidates
        mmr_selected = mmr_select(
            query_embedding, rows, lambda_param=self._mmr_lambda, top_k=self._dense_top_k
        )
        # NOTE: top_k=self._dense_top_k here means MMR reorders all 10 rather than
        # cutting down early reranker gets the full diversified set to make the
        # final relevance call. Change to a smaller number if you want MMR itself
        # to trim before reranking.

        # 4. Cross-encoder rerank on the MMR-ordered candidates -> final top 5
        candidate_texts = [c["content"] for c in mmr_selected]
        
        rerank_result = self.reranker_obj.reranker(query_text=query_text, candidate_texts=candidate_texts, top_k=self.top_k)

        # 5. Wrap final reranked results into NodeWithScore
        nodes_with_scores = []
        for r in rerank_result.results:
            original = mmr_selected[r.index]
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
    obj = AdvancedRetriever()
    nodes = obj.retrieve("What date saw the writing of the song where the devil went down to the state where WDXQ is located?")
    for node in nodes:
        print(f"text: {node.node.text}, score: {node.score}\n")