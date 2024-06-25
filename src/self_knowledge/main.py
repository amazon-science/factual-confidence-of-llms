import logging
from pathlib import Path
from typing import List

import torch
import typer
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BitsAndBytesConfig

from self_knowledge.arch import get_model
from self_knowledge.evaluation.score_functions import run_scoring
from self_knowledge.evaluation.train_scorer import MLP2

accelerator = Accelerator()


def main(
    model_name: str,
    dataset_path: str = "data/reduced_trex_test",
    methods: List[str] = [
        "sequence_log_score",
        "surrogate_logit_score",
        "verbalized",
        "hidden",
    ],
    hidden_scorer_path: str = None,
    save_path: str = "results/",
    hidden_layer_idx: int = 24,
    batch_size: int = 10,
    paraphrase: bool = False,
    log_path: str = "logs/log.log",
    dry_run: bool = False,
    save_freq: int = 50,
    no_accelerator: bool = True,
):
    """
    main experiment runner, runs all scorers on the dataset (except consistency, generated separately)
    trained methods must first be trained.

    Parameters
    ----------
    model_name : str
        Model name
    dataset_path : str
        Dataset path. If set to None, will attempt to load it from default position "data/", if not available, will generate it there
    methods : List[str]
        List of methods to run. Available methods are: 'sequence_log_score', 'surrogate_logit_score', 'verbalized', 'hidden'
        methods are run sequentially on each datapoint. If possible, it is faster to launch multiple runs in parallel, each with a single method.
    hidden_scorer_path : str
        Path to the hidden scorer model. Required if 'hidden' is in methods
    save_path : str
        Path to save the results. results are saved regularly in {save_path}/{method}_{accelerator.device.index}_{batch_idx}.pt
        if no accelerator is provided, None is used instead of accelerator.device.index
    hidden_layer_idx : int
        Index of the hidden layer to use for hidden scoring. Default is 24.
    batch_size : int
        Batch size
    paraphrase : bool
        If set to True, will use the paraphrase column of the dataset instead of the text column. will also save the paraphrases in the save_path.
    log_path : str
        Path to save the logs. Logs are saved in {log_path}/{accelerator.device.index}.log
    save_freq : int
        Frequency of saving the results. Results are saved every save_freq batches.
    no_accelerator : bool
        If set to True, will not use the accelerator, and set device_map to "balanced" when loading the model.
    dry_run: bool
        if set to True will run the pipeline on a small batch of data. Useful for debugging.
    Returns
    -------
    None
    """
    Path(log_path).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG, filename=Path(log_path) / f"{accelerator.device.index}.log"
    )
    Path(save_path).mkdir(parents=True, exist_ok=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model, tokenizer = get_model(
        model_name=model_name,
        quantization_config=bnb_config if no_accelerator else None,
        device_map="balanced" if no_accelerator else None,
    )
    if ".csv" in dataset_path:
        trex_dataset = load_dataset("csv", data_files=dataset_path, delimiter=",")[
            "train"
        ].shuffle(seed=42)

    else:
        trex_dataset = load_from_disk(dataset_path)

    if dry_run:
        trex_dataset = trex_dataset.select(range(batch_size * 2))

    if "hidden" in methods or "value" in methods:
        hidden_scorer = MLP2(hidden_size=model.config.hidden_size).to(
            accelerator.device
        )
        if hidden_scorer_path is not None:
            hidden_scorer = (
                torch.load(hidden_scorer_path).to(accelerator.device).to(torch.bfloat16)
            )
        elif "hidden" in methods or "value" in methods:
            logging.error(
                "no scorer_load_path was provided, methods {'hidden', 'value'} should not be used."
            )
    else:
        hidden_scorer = None

    if "consistency" in methods:
        logging.error(
            "consistency is not yet implemented from main scorer. They have to be generated independently"
        )

    dataloader = DataLoader(trex_dataset, batch_size=batch_size, drop_last=True)
    if not no_accelerator:
        model, tokenizer, dataloader = accelerator.prepare(model, tokenizer, dataloader)

    scores = {}
    labels = torch.tensor([], device="cpu")
    uuids = []
    paraphs = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            new_scores = run_scoring(
                model,
                tokenizer,
                batch["text"] if not paraphrase else batch["paraphrase"],
                uuids=batch["uuid"],
                device="cuda:0" if no_accelerator else accelerator.device,
                hidden_scorer=hidden_scorer,
                hidden_layer_idx=hidden_layer_idx,
                methods=methods,
            )
            labels = torch.concat(
                [
                    labels,
                    batch[
                        (
                            "generation_correct"
                            if "generation_correct" in batch.keys()
                            else "is_factual"
                        )
                    ]
                    .detach()
                    .cpu(),
                ]
            )
            uuids.extend(batch["uuid"])
            if paraphrase:
                paraphs.extend(batch["paraphrase"])
            for method in new_scores.keys():
                if method not in scores.keys():
                    scores[method] = new_scores[method].cpu()
                # pad new scores with 0s to match the length of the previous scores
                # vice versa
                else:
                    if len(new_scores[method].shape) > 1:
                        max_length = new_scores[method].shape[
                            1
                        ]  # max(new_scores[method].shape[1], scores[method].shape[1])
                        new_scores[method] = torch.nn.functional.pad(
                            new_scores[method],
                            (0, 0, 0, max_length - new_scores[method].shape[1]),
                        )
                        scores[method] = torch.nn.functional.pad(
                            scores[method],
                            (0, 0, 0, max_length - scores[method].shape[1]),
                        )
                    scores[method] = torch.cat(
                        [scores[method], new_scores[method].cpu()]
                    )
                if (
                    labels.shape != scores[method].shape
                ):  # FIXME should this be an assert? (can I use logging with asserts?)
                    logging.warning(
                        f"{method} dimension missmatch labels: {labels.shape} method: {scores[method].shape}"
                    )
                if (batch_idx + 1) % save_freq == 0 or batch_idx == len(dataloader) - 1:
                    logging.info(f"Saving results for batch {batch_idx}")
                    torch.save(
                        scores[method],
                        f"{save_path}/{method}_{accelerator.device.index}_{batch_idx}.pt",
                    )
                    scores.pop(method)
            if (batch_idx + 1) % save_freq == 0 or batch_idx == len(dataloader) - 1:
                torch.save(
                    labels,
                    f"{save_path}/{accelerator.device.index}_labels_{batch_idx}.pt",
                )
                torch.save(
                    uuids,
                    f"{save_path}/{accelerator.device.index}_uuids_{batch_idx}.pt",
                )
                labels = torch.tensor([], device="cpu")
                uuids = []
                if paraphrase:
                    torch.save(
                        paraphs,
                        f"{save_path}/{accelerator.device.index}_paraph_{batch_idx}.pt",
                    )
                    paraphs = []


if __name__ == "__main__":
    typer.run(main)
