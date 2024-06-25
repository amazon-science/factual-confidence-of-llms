import logging
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import tqdm
import typer
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedTokenizerFast,
)

from self_knowledge.arch import get_model
from self_knowledge.data_gen.paraphrase.nli import NLI

accelerator = Accelerator()


def slot_fill(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
    sentences: Dict[str, str],
    accelerator: Accelerator,
    is_popqa: bool = False,
) -> Tuple[bool, List[str]]:
    """get a batch of sentences from Trex or PopQA
    generate an amount of words greedily that corresponds to the possible answer
    check and label if answer is correct
    Parameters
    ----------
    model : torch.nn.Module
    tokenizer : transformers.PreTrainedTokenizerFast
    sentences : Dict{text: str, obj_label: str}
    accelerator : accelerate.Accelerator
    is_popqa : bool
        if True, we use the PopQA dataset, else we use Trex
    Returns
    -------
    success : list of bool
    """
    if is_popqa:
        ans = "obj"
        question = "text"
        nli = None
    else:
        ans = "obj_label"
        question = "prompt"
        nli = NLI()

    max_new_tokens = 20
    input_tok = tokenizer(
        sentences[question], padding=True, truncation=True, return_tensors="pt"
    ).to(accelerator.device)
    gen_o = model.generate(
        **input_tok, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id
    )
    logging.debug(f"input: {sentences[question]}")
    generated = tokenizer.batch_decode(gen_o, skip_special_tokens=True)
    logging.debug(f"generated: {generated}")
    logging.debug(f"expected: {sentences[ans]}")

    # in the popqa case, we check all proposed options
    if is_popqa:
        is_factual = [sentences[ans][i] in _gen for i, _gen in enumerate(generated)]
    else:
        _decoded = tokenizer.batch_decode(gen_o)
        # concat prompt and answer
        _truth = [
            f"{sentences[question][i]} {sentences[ans][i]}"
            for i in range(len(sentences[ans]))
        ]
        success = nli.check_equivalence(_truth, _decoded)
        failure = []
        for i, s in enumerate(generated):
            failure.append(False)
            if not success[i]:
                for word in s.split(" "):
                    if word in sentences["verified_false_facts"][i]:
                        failure[i] = True
                        break
        # three table into one table with 3 classes
        is_factual = [
            "factual" if s else "non-factual" if f else "unknown"
            for s, f in zip(success, failure)
        ]
    logging.debug(f"is_factual: {is_factual}")
    return is_factual, generated


def evaluate_all_slot_filling(
    model, tokenizer, dataloader, accelerator, is_popqa=False
) -> None:
    """evaluate all slot filling tasks"""
    out = {
        key: []
        for key in ["generated", "generation_correct", *dataloader.dataset.column_names]
    }
    for batch in tqdm.tqdm(dataloader):
        sc, gen = slot_fill(model, tokenizer, batch, accelerator, is_popqa=is_popqa)
        out["generated"].extend(gen)
        out["generation_correct"].extend(sc)
        for key in dataloader.dataset.column_names:
            out[key].extend(batch[key])
            # if needed, to cpu
            if isinstance(out[key][0], torch.Tensor):
                out[key][-1] = out[key][-1].cpu()
    return out


def log_to_pandas_dataframe(logpath: str = "logs/slot_filling_2/", save: bool = True):
    data = {"input": [], "text": [], "expected": [], "is_factual": []}
    for file in glob.glob(f"{logpath}*.log"):
        with open(file, "r") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                if "input" in lines[i]:
                    data["input"].extend(
                        [
                            text.replace("'", "")
                            .replace('"', "")
                            .replace("[", "")
                            .replace("]", "")
                            .replace("\n", "")
                            for text in lines[i].split("input: ")[1].split("',")
                        ]
                    )
                elif "generated" in lines[i]:
                    data["text"].extend(
                        [
                            text.replace("'", "")
                            .replace('"', "")
                            .replace("[", "")
                            .replace("]", "")
                            .replace("<|endoftext|>", "")
                            .replace("\n", "")
                            for text in lines[i].split("generated: ")[1].split("',")
                        ]
                    )
                elif "expected" in lines[i]:
                    data["expected"].extend(
                        [
                            text.replace("'", "")
                            .replace('"', "")
                            .replace("[", "")
                            .replace("]", "")
                            .replace("\n", "")
                            for text in lines[i].split("expected: ")[1].split(",")
                        ]
                    )
                elif "success" in lines[i]:
                    data["is_factual"].extend(
                        [
                            text.replace("[", "").replace("]", "").replace("\n", "")
                            == "True"
                            for text in lines[i].split("success: ")[1].split(",")
                        ]
                    )
    if save:
        pd.DataFrame(data).to_csv(f"{logpath}/slot_filling.csv")
    return data


def lama_gen(
    outpath="data/lama_sf.csv",
    log_path="logs/improved_sf",
    dataset_path="data/wikidata_trex",
    model_name="tiiuae/falcon-7b-instruct",
    batch_size=2,
    accelerator=None,
):
    Path(outpath).parent.mkdir(exist_ok=True, parents=True)
    Path(log_path).mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        level=logging.DEBUG,
        filename=f"{log_path}/gpu_{str(accelerator.device).split(':')[-1]}.log",
    )
    model, tokenizer = get_model(model_name=model_name)
    # load all jsonl in dataset_path
    tr_dataset = pd.DataFrame()
    for file in glob.glob(f"{dataset_path}/*.jsonl"):
        tr_dataset = pd.concat([tr_dataset, pd.read_json(file, lines=True)])
    tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size)
    model, tokenizer, tr_dataloader = accelerator.prepare(
        model, tokenizer, tr_dataloader
    )
    sf = torch.stack(
        evaluate_all_slot_filling(
            model, tokenizer, tr_dataloader, is_popqa=False, accelerator=accelerator
        )
    )
    sf.to_csv(f"{outpath}/lama_sf_{accelerator.device.index}.csv")


def popqa_gen(
    log_path="logs/pop_slot_filling",
    data_path="data/clean_popQA.csv",
    output_path="results_kn/",
    model_name="tiiuae/falcon-40b-instruct",
    no_accelerator=True,
    batch_size: int = 50,
    train_split=True,
):
    Path(log_path).mkdir(exist_ok=True, parents=True)
    Path(output_path).mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        level=logging.DEBUG,
        filename=f"{log_path}/gpu_{str(accelerator.device).split(':')[-1]}.log",
    )
    dataset = load_dataset("csv", data_files=data_path, delimiter=",")["train"].shuffle(
        seed=42
    )

    tr_dataloader = DataLoader(dataset, batch_size=batch_size)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model, tokenizer = get_model(
        model_name=model_name,
        quantization_config=bnb_config,
        device_map="balanced" if no_accelerator else None,
    )
    if not no_accelerator:
        model, tokenizer, tr_dataloader = accelerator.prepare(
            model, tokenizer, tr_dataloader
        )
    sf = evaluate_all_slot_filling(
        model, tokenizer, tr_dataloader, is_popqa=True, accelerator=accelerator
    )
    pd.DataFrame(sf).to_csv(f"{output_path}/popqa_sf_{accelerator.device.index}.csv")


if __name__ == "__main__":
    typer.run(popqa_gen)
