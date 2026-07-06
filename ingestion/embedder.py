import voyageai

from configs.config import VOYAGE_API_KEY, EMBEDDING_MODEL, BATCH_SIZE
from ingestion.chunker import corpus_chunker
from ingestion.data_sampler import data_sampler
from ingestion.corpus_extractor import corpus_extractor

try:
    vo_client = voyageai.Client(api_key=VOYAGE_API_KEY)
except Exception as e:
    print(e)

class Embedder():
    def __init__(self):
        pass

    def modify_chunk_list(self, chunk_list: list[dict], embeddings: list[float]) -> list[dict]:
        for chunk, embedding in zip(chunk_list, embeddings):
            chunk["embedding"] = embedding

        return chunk_list

    def corpus_embedder(self, chunk_list: list[dict]) -> list[dict]:
        running_tokens: int = 0
        batch: list[str] = []
        embeddings: list[float] = []
        try:
            for chunk in chunk_list:
                if running_tokens + chunk["token_count"] >= BATCH_SIZE:
                    embeddings+= vo_client.embed(texts=batch, model=EMBEDDING_MODEL, output_dimension=1024, input_type="document").embeddings
                    running_tokens = 0
                    batch = []
                
                running_tokens+=chunk["token_count"]
                batch.append(chunk["content"])
            
            if batch:
                embeddings += vo_client.embed(texts=batch, model=EMBEDDING_MODEL, output_dimension=1024).embeddings
            
            modified = self.modify_chunk_list(chunk_list=chunk_list, embeddings=embeddings)
            return modified
                
        except Exception as e:
            print(e)

if __name__=="__main__":
    sampled = data_sampler() 
    flattened = corpus_extractor(df=sampled)

    chunks = corpus_chunker(paragraph_list=flattened)

    obj = Embedder()
    modified_chunks = obj.corpus_embedder(chunk_list=chunks)
    print(len(modified_chunks))
    print(modified_chunks[0])
