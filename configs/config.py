import os

from dotenv import load_dotenv

load_dotenv()

DB_HOST             = os.getenv("DB_HOST")
DB_NAME             = os.getenv("DB_NAME")
DB_PASSWORD         = os.getenv("DB_PASSWORD")
DB_PORT             = os.getenv("DB_PORT")
DB_USER             = os.getenv("DB_USER")

DATA_FILE_PATH      = "./data/musique_ans_v1.0_dev.jsonl"
OUPUT_SAMPLE_DATA   = "./data/musique_270.jsonl"

EMBEDDING_TOKENIZER = "voyageai/voyage-4-nano"
EMBEDDING_MODEL     = "voyage-4-lite"

VOYAGE_API_KEY      = os.getenv("VOYAGE_API_KEY")
DEEPSEEK_API_KEY    = os.getenv("DEEPSEEK_API_KEY")

MAX_CHUNK_SIZE      = 350
MAX_CHUNK_OVERLAP   = int(0.15*MAX_CHUNK_SIZE)
BATCH_SIZE          = 25000 # cap 32k