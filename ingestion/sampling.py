import pandas as pd
from pandas import DataFrame

from configs.config import DATA_FILE_PATH, OUPUT_SAMPLE_DATA

def data_loader(file_path: str) -> DataFrame:
    data = pd.read_json(file_path, lines=True, encoding="utf-8")
    return data

def data_sanity_check(data: DataFrame) -> None:
    print(data.shape)
    print(data.columns)
    print(data.iloc[0])

def data_sampler(data: DataFrame) -> DataFrame:
    # extract hop number from "2hop"...
    data["hop_count"] = data["id"].str.extract(r"^(\d+)hop").astype(int)

    grouped = data.groupby("hop_count")

    sampled = grouped.sample(n=90, random_state=42)

    try:
        sampled.to_json(OUPUT_SAMPLE_DATA, orient="records", lines=True, force_ascii=False)
    except Exception as e:
        print(e)

    return sampled

def sample_sanity_check(sample: DataFrame) -> None:
    print(sample["hop_count"].value_counts())
    print(sample["id"].duplicated().sum())

if __name__=="__main__":
    df = data_loader(file_path=DATA_FILE_PATH)
    data_sanity_check(data=df)
    sampled = data_sampler(data=df)
    sample_sanity_check(sample=sampled)









