"""
data preparation script for generating and saving datasets for self-confidence tasks.
"""


import datasets
import pandas as pd
from datasets import load_dataset, load_from_disk

from self_knowledge.data_gen.sentence_extractors import FalseTrex, format_template_trex


def save_modified_lama(output_path, input_path=None, num_proc=10):
    """
    Downloads the LAMA dataset and saves it in the format for the self-confidence task.
    It generates both True facts and False facts by sampling false completions from the same category.
    Outputs a train and test set.

    Patameters
    ----------
    output_path: str
        path to save the dataset.
    input_path: str
        path to load the dataset from. If None, downloads the dataset.
    num_proc: int
        number of processes to use for multiprocessing.
    """
    if input_path is None:
        dataset = load_dataset("lama", "trex", split="train").shuffle(seed=42)
    else:
        dataset = load_from_disk(input_path).shuffle(seed=42)
    # split dataset into train/test
    # filter non_unique ["obj", "subj", "template"]
    dataset = pd.DataFrame(dataset)
    dataset = dataset.drop_duplicates(subset=["obj_label", "template", "sub_label"])
    dataset = datasets.Dataset.from_pandas(dataset)

    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset, test_dataset = dataset["train"], dataset["test"]

    false_data_maker = FalseTrex(test_dataset)
    fte_dataset = test_dataset.map(
        false_data_maker.format_false_trex,
        num_proc=num_proc,
        remove_columns=["uuid", "masked_sentence"],
    ).rename_column("uuid2", "uuid")
    tte_dataset = test_dataset.map(
        format_template_trex,
        num_proc=num_proc,
        remove_columns=["masked_sentence"],
    )

    ftr_dataset = train_dataset.map(
        false_data_maker.format_false_trex,
        num_proc=num_proc,
        remove_columns=["uuid", "masked_sentence"],
    ).rename_column("uuid2", "uuid")
    ttr_dataset = train_dataset.map(
        format_template_trex,
        num_proc=num_proc,
        remove_columns=["masked_sentence"],
    )
    # combine both
    te_dataset = datasets.concatenate_datasets([fte_dataset, tte_dataset])
    tr_dataset = datasets.concatenate_datasets([ftr_dataset, ttr_dataset])

    te_dataset.save_to_disk(output_path + "_test")
    tr_dataset.save_to_disk(output_path + "_train")


def save_generated_trex(
    input_path: str = "logs/slot_filling_2/slot_filling_2.csv",
    output_path: str = "data/sf2_trex",
):
    """
    save generated data as a hf dataset
    """
    # make into dataset
    dataset = load_dataset(
        "csv",
        data_files=input_path,
        delimiter=",",
    )
    # save as test dataset
    dataset["train"].save_to_disk(output_path + "_test")


if __name__ == "__main__":
    save_modified_lama("data/reduced_trex")
