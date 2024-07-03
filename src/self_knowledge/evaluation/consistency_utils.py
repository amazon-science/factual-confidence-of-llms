# This script takes as input sampled completions as generated in data_gen/sampling.py.
# The cleanup_sampling function must be used to ensure proper format.

import logging
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typer import Typer

app = Typer()


class SimilarityFuns(Enum):
    NLI = 1


# CONSISTENCY SCORING


def get_nli_scoring_fun(device):
    logging.info(f"Loading docnli on {device}...")
    model, tokenizer = _load_nli_model(device)
    return partial(_compute_entailment, entailment_model=model, tokenizer=tokenizer, device=device)


def _load_nli_model(device):
    model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer


def _compute_entailment(left_sequences, right_sequences, entailment_model, tokenizer, device):
    input = tokenizer(
        left_sequences,
        right_sequences,
        truncation=True,
        return_tensors="pt",
        padding=True,
    )
    for k, v in input.items():
        try:
            input[k] = v.to(device)
        except RuntimeError:
            pass

    output = entailment_model(**input)
    predictions = torch.softmax(output["logits"], -1).cpu().detach().numpy()[:, 0]
    return predictions


def _adjust_name(base_name: str, bidirectional: bool) -> str:
    """
    Adjust the name of the similarity metric based on the passed parameters.
    """
    if bidirectional:
        base_name += "_bidirectional"
    return base_name


def _base_consistency_function(
    sampled_responses: List[str],
    get_scores_fun: Callable[[List[str], List[str]], List[float]],
    bidirectional: bool = False,
) -> Tuple[float, str]:
    """
    Wrapper function for consistency scoring. It handles the logic for computing the similarity scores across
    all given samples (with the provided get_scores_fun) and combining them into a single value.

    It returns the final score and the best answer associated with that score.

    Parameters:
        sampled_responses: List of responses from the model.
        get_scores_fun: Function that takes two lists of strings and returns a list of similarity scores.
        bidirectional: Whether to enforce the similarity function to be biderectional (by taking the mean of the score
            for both possible orderings). Note: this argument is only relevant for functions that are not bidirectional
            by definition.
    Returns:
        Tuple of (score, best_answer).
    """
    num_samples = len(sampled_responses)
    sim_array = np.zeros((num_samples, num_samples))
    for s_idx, passage in enumerate(sampled_responses):
        sim_array[s_idx, :] = get_scores_fun([passage] * num_samples, list(sampled_responses))

    # Enforce self-similarity to be 1
    np.fill_diagonal(sim_array, 1)

    # Enforce similarity to be symmetric; if mean score is used bidirectionality doesn't affect the score.
    if bidirectional:
        sim_array = (sim_array + sim_array.T) / 2

    # Mean of all scores
    mean_score = sim_array.mean()

    # As the final score we take the avs score for the answer most similar to other answers.
    final_score, arg_mvote = sim_array.mean(1).max(), sim_array.mean(1).argmax()
    answer = sampled_responses[arg_mvote]
    return mean_score, final_score, answer


def compute_on_device(input: Tuple) -> Tuple[List[str], List[float]]:
    idx, sim_function, response_lists, bidirectional = input

    answers, scores, best_answer_scores = [], [], []
    with torch.inference_mode():
        answers, scores = [], []
        for responses in tqdm(
            response_lists,
            total=len(response_lists),
            position=idx,
            desc=f"Process {idx}",
        ):
            if not any(x != "" for x in responses):
                answers.append("")
                best_answer_scores.append(0)
                scores.append(0)
                continue

            mean_score, best_answer_score, answer = _base_consistency_function(
                responses, sim_function, bidirectional
            )
            answers.append(answer)
            best_answer_scores.append(best_answer_score)
            scores.append(mean_score)
    return idx, answers, scores, best_answer_scores


def compute_all_scores(
    response_lists: List[List[str]],
    bidirectional: bool = False,
    num_gpus: int = 1,
    methods=[SimilarityFuns.NLI],
    uuids: List[str] = None,
) -> Tuple[dict, str]:
    """
    Compute the self consistency scores for each list in the response_list.
    """
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    out_dict = {}
    out_file_extension = []
    if len(response_lists) < 50:
        num_gpus = 1

    devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
    device_count = len(devices)

    # Split the workload across the devices.
    response_lists_split = np.array_split(response_lists, device_count)
    logging.info(f"The data will be processed in parallel on {device_count} devices.")

    def _compute_scores(device2fun: Dict[str, Callable], method_key: str):
        method_key = _adjust_name(method_key, bidirectional)

        answers, scores, best_answer_scores = [], [], []
        input_to_pool = [
            (
                i,
                device2fun[device_name],
                response_lists_split[i],
                bidirectional,
            )
            for i, device_name in enumerate(devices)
        ]

        try:
            pool = mp.Pool(device_count)
            results = pool.map(compute_on_device, input_to_pool)
        finally:
            pool.close()
            pool.join()

        for result in sorted(results, key=lambda x: x[0]):
            idx, device_answers, device_scores, device_best_answer_scores = result
            scores.extend(device_scores)
            answers.extend(device_answers)
            best_answer_scores.extend(device_best_answer_scores)

        (
            out_dict[f"{method_key}_mean_scores"],
            out_dict[f"{method_key}_answers"],
            out_dict["uuids"],
            out_dict[f"{method_key}_best_answer_scores"],
        ) = (scores, answers, uuids, best_answer_scores)
        out_file_extension.append(method_key)

    # NOTE: one can easily add scores from different scoring methods using the following as an example:
    if SimilarityFuns.NLI in methods:
        logging.info("Computing NLI scores...")
        device2fun = {device: get_nli_scoring_fun(device) for device in devices}
        _compute_scores(device2fun, "nli")

    return out_dict, "_".join(out_file_extension)


@app.command()
def main(
    data_path: str,
    results_path: str,
    bidirectional: bool = False,
    save_suffix: str = "consistency",
):
    for path in Path(data_path).glob("*.tsv"):
        df = pd.read_csv(path, sep="\t")
        # columns 0 to 9 are the responses
        example_response_lists = df.iloc[:, 1:11].values.tolist()
        uuids = df["uuid"].values.tolist()
        print("examples", example_response_lists[:3][:2])

        out_dict, file_suffix = compute_all_scores(
            example_response_lists, bidirectional=bidirectional, uuids=uuids
        )
        print(f"Bidirectional: {bidirectional}")
        print(f"Scores:\n {out_dict}")
        # save scores as torch .pt to fit existing pipeline
        Path(results_path).mkdir(parents=True, exist_ok=True)
        torch.save(out_dict, Path(results_path) / f"consistency_{file_suffix}_{save_suffix}.pt")


if __name__ == "__main__":
    app()
