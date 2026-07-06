from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.retrievers import BaseRetriever

from db.db_client import execute_query
from retrieval.query_embedder import query_embedder

class NaiveRetriever(BaseRetriever):
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        super().__init__()
    
    def _retrieve(self, query_bundle: QueryBundle):
        query_str = query_bundle.query_str
        query_embedding = query_embedder(user_query=query_str)

        sql = """
            SELECT chunk_id, question_id, hop_count, is_supporting, content, 1 - (embedding <=> %s::vector) AS similarity_score
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """

        rows = execute_query(sql=sql, params=(query_embedding, query_embedding, self.top_k))

        nodes_with_score: list[NodeWithScore] = []
        for row in rows:
            node = TextNode(
                text=row["content"],
                metadata={
                    "chunk_id"      : row["chunk_id"],
                    "question_id"   : row["question_id"],
                    "hop_count"     : row["hop_count"],
                    "is_supporting" : row["is_supporting"], 
                }
            )

            nodes_with_score.append(NodeWithScore(node=node, score=row["similarity_score"]))
        
        return nodes_with_score
    
if __name__=="__main__":
    obj = NaiveRetriever()
    nodes = obj.retrieve("What date saw the writing of the song where the devil went down to the state where WDXQ is located?")
    for node in nodes:
        print(f"text: {node.node.text}, score: {node.score}\n")
