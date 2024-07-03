# for a given model, sample 10 outputs with temperatures of 0.5 and 1 for slot filling on trex
# and save the results to a file
# can be used for both lama and popQA, with local models or by querying APIs. An example is implemented for OpenAI.
# python -m self_knowledge.self_knowledge.data_gen.sampling
# --model_name_or_path facebook/bart-base
# --model_type bart
# --output_dir self_knowledge/data/sampling
# --save_freq 10
# --batch_size 10
# --temperature 0.5
# --temperature 1
# --num_beams 10
# --num_return_sequences 10
# --num_samples 10

import logging
import os
import time

import pandas as pd
import torch
import typer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from self_knowledge.data_gen.openai_sampler import gpt

logging.basicConfig(level=logging.INFO)
app = typer.Typer()


def get_lama(rank, num_ranks):
    # initialize dataset
    logging.info("initializing dataset")
    dataset = load_dataset("csv", data_files="data/reduced_mix_pop_paraph/clean.csv")[
        "train"
    ]
    # drop duplicates - for consistency we don't care if a fact is true or false, only if the model "knows" it, which we evaluate by checking for the answer in the output in the slot filling script
    row_idx = (
        pd.DataFrame(dataset)
        .drop_duplicates(subset=["template", "obj_label", "sub_label"])
        .index
    )
    dataset = dataset.select(row_idx[rank::num_ranks])
    # drop duplicates for sub_label template obj_label
    dataset = dataset.map(
        lambda x: {
            "sf": x["template"].replace("[X]", x["obj_label"]).split("[Y]")[0],
            "expected": x["template"].replace("[X]", x["obj_label"]).split("[Y]")[0]
            + x["sub_label"]
            + x["template"].replace("[X]", x["obj_label"]).split("[Y]")[1],
        }
    )
    return dataset


def get_popQA(rank, num_ranks):
    dataset = load_dataset("csv", data_files="data/reduced_mix_pop_paraph/clean.csv")[
        "train"
    ].shuffle(seed=42)
    dataset = dataset.map(
        lambda x: {
            "sf": x["paraphrase"],
            "expected": x["uuid"],
        }
    )
    return dataset


@app.command()
def main(
    model_name_or_path: str = "gpt-3.5",
    output_dir: str = "data/gpt35_sampling",
    batch_size: int = 1,
    temperature: float = 1.0,
    num_beams: int = 1,
    num_samples: int = 10,
    save_freq: int = 100,
    rank: int = 0,
    num_ranks: int = 1,
    dataset_name: str = "lama",
):
    # initialize output dir
    logging.info("initializing output dir")
    os.makedirs(output_dir, exist_ok=True)
    # initialize model
    logging.info("initializing model")
    if model_name_or_path != "gpt-3.5":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map="balanced", quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, padding_side="left"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        pad_token_id = tokenizer.pad_token_id
    else:
        model = gpt()
        pad_token_id = None
        if batch_size > 1:
            raise ValueError(
                "batch_size must be 1 for gpt-3.5, other values not supported"
            )
    if dataset_name == "lama":
        dataset = get_lama(rank, num_ranks)
    elif dataset_name == "popQA":
        dataset = get_popQA(rank, num_ranks)
    else:
        raise ValueError("dataset must be lama or popQA")
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # sample and score
    logging.info("sampling and scoring")
    outputs = pd.DataFrame()
    for i, batch in enumerate(tqdm(dataloader)):
        # get num_samples samples
        if model_name_or_path != "gpt-3.5":
            sf = tokenizer(batch["sf"], return_tensors="pt", padding=True)
        else:
            sf = batch["sf"]
        max_new_tokens = 50  # exp["input_ids"].shape[1] - sf["input_ids"].shape[1] + 10
        for _ in range(num_samples):
            output = model.generate(
                **(
                    sf.to(f"cuda:{rank}")
                    if model_name_or_path != "gpt-3.5"
                    else {"input_text": sf[0]}
                ),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_beams=num_beams,
                do_sample=True,
                pad_token_id=pad_token_id,
            )
            df = pd.DataFrame(batch)
            df["output"] = (
                tokenizer.batch_decode(output, skip_special_tokens=True)
                if model_name_or_path != "gpt-3.5"
                else [output]
            )
            outputs = pd.concat([outputs, df])
        if (i + 1) % save_freq == 0 or i == len(dataloader) - 1:
            logging.info("saving")
            outputs.to_csv(os.path.join(output_dir, f"outputs_{rank}_{i}.csv"))
            outputs = pd.DataFrame()
        if model_name_or_path == "gpt-3.5":
            time.sleep(10)


@app.command()
def cleanup_sampling(
    sample_dir: str = "data/gpt35_sampling",
    output_dir: str = "data/gpt35_sampling",
    popQA_repair: bool = False,
    force_n_columns: int = None,
):
    """
    Once all data has been sampled, transform the log file into a single file with all outputs.
    Once formatted this file can be used for consistency evaluation.
    """
    # cleanup sampling dir
    logging.info("cleaning up sampling dir")
    # collect all in one file
    outputs = pd.DataFrame()
    ordered_ranks = {}
    dirs = list(os.listdir(sample_dir))
    for i, file in enumerate(dirs):
        if ".csv" not in file:
            dirs.pop(i)
    if "outputs.tsv" in dirs:
        dirs.pop(dirs.index("outputs.tsv"))
    print(dirs)

    for file in sorted(dirs, key=lambda x: int(x.split("_")[2].split(".")[0])):
        rank = file.split("_")[1]
        if rank not in ordered_ranks:
            ordered_ranks[rank] = pd.DataFrame()
        # find all lines that have the same "expected" and group "output" by them
        df = pd.read_csv(os.path.join(sample_dir, file))
        grouper = "expected" if not popQA_repair else ["uuid"]
        df = df.groupby(grouper, sort=False)["output"].apply(list).reset_index()
        if force_n_columns is not None:
            df["output"] = df["output"].apply(lambda x: x[:force_n_columns])
        # make each element in list a new column
        df = df.join(pd.DataFrame(df["output"].tolist()))
        df = df.drop(columns=["output"])
        # check that all outputs are the same length
        ordered_ranks[rank] = pd.concat([ordered_ranks[rank], df])
    # deal with empty files
    if len(ordered_ranks) == 0:
        logging.info("no files to process")
        return
    # order by rank
    for i in range(len(ordered_ranks[list(ordered_ranks.keys())[0]])):
        for rank in sorted(ordered_ranks, key=lambda x: int(x)):
            if i < len(ordered_ranks[rank]):
                outputs = pd.concat(
                    [outputs, ordered_ranks[rank].iloc[[i]]], ignore_index=True
                )
    outputs.to_csv(os.path.join(output_dir, "outputs.tsv"), sep="\t", index=False)
    logging.info("done")


if __name__ == "__main__":
    app()
