import json

import pandas as pd
from pandas import DataFrame

from ingestion.data_sampler import data_sampler
from configs.config import OUPUT_SAMPLE_DATA

def corpus_exctractor(df: DataFrame) -> list[dict]:
    df = pd.read_json(OUPUT_SAMPLE_DATA, lines=True, encoding="utf=8")

    flattened = []
    with open("./data/raw_extracted.jsonl", "w", encoding="utf-8") as f:
        for id, row in df.iterrows():
            for i in row["paragraphs"]:
                field: dict = {
                    "question_id"   : row["id"],
                    "paragraph_text": i.get("paragraph_text"),
                    "is_supporting" : i.get("is_supporting"),
                    "title"         : i.get("title"),
                    "paragraph_id"  : i.get("idx")
                }
                flattened.append(field)
                f.write(json.dumps(field) + "\n")

    return flattened

if __name__=="__main__":
    sampled: DataFrame = data_sampler()
    flattened: list[dict] = corpus_exctractor(df=sampled)
    print(len(flattened))
    print(flattened[0])

    