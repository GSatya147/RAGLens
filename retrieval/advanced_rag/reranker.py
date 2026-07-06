import voyageai

from configs.config import VOYAGE_API_KEY

vo_client = voyageai.Client(api_key=VOYAGE_API_KEY)

class Reranker():
    def __init__(self):
        pass

    def reranker(self, query_text, candidate_texts, top_k):
        try:
            result = vo_client.rerank(
                query=query_text,
                documents=candidate_texts,
                model="rerank-2.5",
                top_k=top_k
            )

            return result
        except Exception as e:
            print(e)