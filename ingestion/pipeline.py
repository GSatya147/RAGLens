from ingestion.chunker import corpus_chunker
from ingestion.data_sampler import data_sampler
from ingestion.corpus_extractor import corpus_extractor
from ingestion.embedder import Embedder
from ingestion.writer import insert_chunks, insert_questions, get_questions_db, get_chunks_db, clear_db
from ingestion.dedup_corpus import dedupe_chunks

def pipeline_run():
    clear_db()

    sampled = data_sampler() 
    flattened = corpus_extractor(df=sampled)

    insert_questions(df=sampled)

    chunks = corpus_chunker(paragraph_list=flattened)

    obj = Embedder()
    modified_chunks = obj.corpus_embedder(chunk_list=chunks)

    chunks_list = dedupe_chunks(modified_chunks)
    insert_chunks(chunks_list)

    question_rows = get_questions_db()
    print("question db: ", question_rows[0])
    print("question len: ", len(question_rows))

    chunk_rows = get_chunks_db()
    print("chunk db: ", chunk_rows[0])
    print("chunk len: ", len(chunk_rows))

if __name__=="__main__":
    pipeline_run()
    

