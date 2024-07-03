# extract hidden layers from model for each sentence in a training dataset
# this hidden layers are then used to train a classifier

import json
import logging
import os
from pathlib import Path

import torch
import typer
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import trange

from self_knowledge.arch import get_model

accelerator = Accelerator()


def save_hidden_layer(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
    dataset_path: str = "data/neg_beam/mist_hard_negs.csv",
    continue_from: str = None,
    log_path: str = "logs",
    hidden_layer_index: int = 24,
    save_file: str = "models/mist_hard_hidden24.pt",
    batch_size: int = 1,
    save_freq: int = 1000,
):
    Path(save_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=f"{log_path}/log_hidden.txt", level=logging.DEBUG)
    model, tokenizer = get_model(model_name, load_in_4bit=True, device_map="balanced")
    model, tokenizer = accelerator.prepare(model, tokenizer)
    if ".csv" in dataset_path:
        dataset = load_dataset("csv", data_files=dataset_path)["train"]
        tr_dataset = dataset.train_test_split(test_size=0.2, seed=42)["train"]
    else:
        tr_dataset = load_from_disk(dataset_path)

    # as accelerate is going back and forth between GPU and CPU, we separate them into two lists
    hidden_layer = []  # on GPU or CPU, smaller
    out = []  # only CPU, all data
    starting_idx = 0
    if continue_from is not None:
        out = torch.load(continue_from)
        starting_idx = len(out) // save_freq
    for i in trange(starting_idx, len(tr_dataset) // save_freq):
        restricted_dataset = tr_dataset.select(
            range(i * save_freq, (i + 1) * save_freq, 1)
        )
        tr_dataloader = DataLoader(restricted_dataset, batch_size=batch_size)
        for batch in tr_dataloader:
            _tok_batch = tokenizer(batch["text"], padding=True, return_tensors="pt").to(
                accelerator.device
            )
            o = model(**_tok_batch, return_dict=True, output_hidden_states=True)
            for i in range(o.hidden_states[hidden_layer_index].shape[0]):
                hidden_layer.append(
                    {
                        "hidden": o.hidden_states[hidden_layer_index][i, -1:, :]
                        .view(-1)
                        .cpu(),
                        "uuid": batch["uuid"][i],
                        "is_factual": batch[
                            (
                                "generation_correct"
                                if "generation_correct" in batch.keys()
                                else "is_factual"
                            )
                        ][i],
                    }
                )

        # gather objects from all processes
        hidden_layer = accelerator.gather_for_metrics(hidden_layer)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            torch.save(
                out + hidden_layer,
                save_file,
            )
        out = out + hidden_layer
        hidden_layer = []


def cleanup_dataset(
    hidden_path: str = "./models/hidden_layers/",
    out_name: str = "fal_7b_hidden.pt",
):
    hidden = []
    j = 0
    for file in Path(hidden_path).glob("*.pt"):
        if out_name not in str(file):
            hidden_layer = torch.load(file)
            uuid = json.loads(
                file.parent.joinpath(
                    file.name.split("hidden")[0] + "uuids.json"
                ).read_text()
            )
            for i in range(len(hidden_layer)):
                hidden.append(
                    {
                        "uuid": uuid["uuids"][i],
                        "hidden": hidden_layer[i],
                        "is_factual": uuid["factuality"][i],
                    }
                )
        j += 1
        if j % 10 == 0 or (j == len(os.listdir(hidden_path)) - 1):
            torch.save(hidden, Path(hidden_path) / out_name)


if __name__ == "__main__":
    typer.run(save_hidden_layer)
