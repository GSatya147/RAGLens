import voyageai
from langfuse import observe

from configs.config import VOYAGE_API_KEY, EMBEDDING_MODEL

try:
    vo_client = voyageai.Client(api_key=VOYAGE_API_KEY)
except Exception as e:
    print(e)

@observe(as_type="embedding")
def query_embedder(user_query: str) -> list[float]:
    try:
        embedding = vo_client.embed(texts=user_query, model=EMBEDDING_MODEL, output_dimension=1024, input_type="query").embeddings
        return embedding[0]
    except Exception as e:
        print(e)

if __name__=="__main__":
    print(query_embedder("hello?"))