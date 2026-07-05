from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from sentence_transformers  import SentenceTransformer

from configs.config import MAX_CHUNK_OVERLAP, MAX_CHUNK_SIZE, EMBEDDING_TOKENIZER 
from ingestion.data_sampler import data_sampler
from ingestion.corpus_extractor import corpus_extractor

splitter = SentenceSplitter(
    chunk_size=MAX_CHUNK_SIZE,
    chunk_overlap=MAX_CHUNK_OVERLAP,
)
try:
    model = SentenceTransformer(EMBEDDING_TOKENIZER, truncate_dim=1024)
    Tokenizer = model.tokenizer
except Exception as e:
    print(e)

def count_tokens(text: str, tokenizer=Tokenizer) -> int:
    try:
        return int(len(tokenizer.encode(text)))
    except Exception as e:
        print(e)

def document_builder(entry: dict) -> Document:
    return Document(
        text = entry.get("paragraph_text"),
        metadata = {
            "question_id"   : entry.get("question_id"),
            "is_supporting" : entry.get("is_supporting"),
            "title"         : entry.get("title"),
            "paragraph_id"  : entry.get("paragraph_id"),
            "hop_count"     : entry.get("hop_count"),
            "embedding"     : None
        }
    )

def corpus_chunker(paragraph_list: list[dict]) -> list[dict]:
    documents: list[Document] = [document_builder(entry) for entry in paragraph_list]

    nodes: list[Document] = splitter.get_nodes_from_documents(documents=documents)

    chunk_list=[]
    for node in nodes:
        if node.text:
            chunk_list.append({
                "question_id"   : node.metadata.get("question_id"),
                "hop_count"     : node.metadata.get("hop_count"),
                "is_supporting" : node.metadata.get("is_supporting"),
                "title"         : node.metadata.get("title"),
                "content"       : node.text,
                "token_count"   : count_tokens(node.text),
                "embedding"     : None,
            })
    return chunk_list

if __name__=="__main__":
    sampled = data_sampler() 
    flattened = corpus_extractor(df=sampled)

    chunks = corpus_chunker(paragraph_list=flattened)
    print(len(chunks))
    print(chunks[0])
    