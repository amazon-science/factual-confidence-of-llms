"""
This script must be run to prepare the PopQA data
"""

import pandas as pd
import typer


def main(dataset_path: str = "data/popQA.tsv", save_path: str = "data/clean_popQA.csv"):
    """
    takes the popQA dataset as provided by the authors and cleans it up to have the same format as LAMA
    and be directly usable by facutal confidence scorers.

    Parameters:
    dataset_path: str: path to the popQA dataset (download from https://github.com/AlexTMallen/adaptive-retrieval/blob/main/data/popQA.tsv)
    save_path: str: path to save the cleaned dataset
    """
    df = pd.read_csv(dataset_path, sep="\t")
    df["text"] = df["question"]
    df.drop(columns=["question"], inplace=True)
    df["uuid"] = df["id"]
    df.drop(columns=["id"], inplace=True)
    df["is_factual"] = -1
    # deal with None (drop in any column)
    print(df.info())
    df = df.dropna()
    print(df.info())
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    typer.run(main)
