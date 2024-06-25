import logging

import yaml

from self_knowledge.evaluation.logit_scorer_utils import (
    sequence_log_score,
    surrogate_logit_score,
)
from self_knowledge.evaluation.verbalized_utils import confidence_in_words


def run_scoring(
    model,
    tokenizer,
    batch,
    device,
    uuids=None,
    hidden_scorer=None,
    value_scorer=None,
    hidden_layer_idx=-1,
    methods=[
        "sequence_log_score",
        "surrogate_logit_score",
        "verbalized",
        "hidden",
        "value",
    ],
    prompt_path: str = "./src/self_knowledge/prompts.yaml",
):
    # model = scorer_model.model
    prompts = yaml.safe_load(open(prompt_path))
    surrogate_targets = tokenizer(
        [
            "Yes",
            " Yes",
            "yes",
            " yes",
            "No",
            " No",
            "no",
            " no",
            "Maybe",
            " Maybe",
            "maybe",
            " maybe",
        ],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )["input_ids"].to(device)

    function_mappings = {
        "verbalized": {
            "scorer": confidence_in_words,
            "scorer_kwargs": {"tokenizer": tokenizer, "device": device},
            "prefix": (
                prompts["verbalized"]["prefix"]
                if "verbalized" in prompts.keys()
                else ""
            ),
            "suffix": (
                prompts["verbalized"]["suffix"]
                if "verbalized" in prompts.keys()
                else ""
            ),
            "model_call": model.generate,
            "model_kwargs": {
                "max_new_tokens": 20,
                "return_dict_in_generate": True,
                "pad_token_id": tokenizer.eos_token_id,
            },
        },
        "sequence_log_score": {
            "scorer": sequence_log_score,
            "scorer_kwargs": {
                "pad_token_id": tokenizer.pad_token_id,
            },
            "prefix": (
                prompts["sequence_log_score"]["prefix"]
                if "sequence_log_score" in prompts.keys()
                else ""
            ),
            "suffix": (
                prompts["sequence_log_score"]["suffix"]
                if "sequence_log_score" in prompts.keys()
                else ""
            ),
            "model_call": model,
            "model_kwargs": {},
        },
        "surrogate_logit_score": {
            "scorer": surrogate_logit_score,
            "scorer_kwargs": {"targets": surrogate_targets},
            "prefix": (
                prompts["surrogate_logit_score"]["prefix"]
                if "surrogate_logit_score" in prompts.keys()
                else ""
            ),
            "suffix": (
                prompts["surrogate_logit_score"]["suffix"]
                if "surrogate_logit_score" in prompts.keys()
                else ""
            ),
            "model_call": model.generate,
            "model_kwargs": {
                "max_new_tokens": surrogate_targets.shape[1],
                "pad_token_id": tokenizer.eos_token_id,
                "return_dict_in_generate": True,
                "output_scores": True,
            },
        },
        "hidden": {
            "scorer": hidden_scorer.score if hidden_scorer is not None else None,
            "scorer_kwargs": {"hidden_layer_idx": hidden_layer_idx},
            "prefix": "",
            "suffix": "",
            "model_call": model,
            "model_kwargs": {"output_hidden_states": True},
        },
        "value": {
            "scorer": value_scorer.score if value_scorer is not None else None,
            "scorer_kwargs": {},
            "prefix": "",
            "suffix": "",
            "model_call": model,
            "model_kwargs": {"output_hidden_states": True},
        },
        "consistency": {
            "scorer": None,
            "scorer_kwargs": {},
            "prefix": "",
            "suffix": "",
            "model_call": model,
            "model_kwargs": {"output_hidden_states": True},
        },
    }
    assert (
        len([m for m in methods if m not in function_mappings.keys()]) == 0
    ), f"{[m for m in methods if m not in function_mappings.keys()]} not implemented, only available are: {function_mappings.keys()}"
    scores = {}
    for method in methods:
        # when needed add prefix and suffix before tokenizing
        if (
            function_mappings[method]["prefix"] != ""
            or function_mappings[method]["suffix"] != ""
        ):
            to_tokenize = [
                function_mappings[method]["prefix"]
                + _t
                + function_mappings[method]["suffix"]
                for _t in batch
            ]
        else:
            to_tokenize = batch
        # tokenize and move to device
        tokenized_input = tokenizer(
            to_tokenize, padding=True, truncation=True, return_tensors="pt"
        )
        for _k in tokenized_input.keys():
            tokenized_input[_k] = tokenized_input[_k].to(device)
        # run and save
        o = function_mappings[method]["model_call"](
            **tokenized_input, **function_mappings[method]["model_kwargs"]
        )
        if method == "sequence_log_score":
            function_mappings[method]["scorer_kwargs"]["input_toks"] = tokenized_input[
                "input_ids"
            ]
        scores[method] = function_mappings[method]["scorer"](
            o, **function_mappings[method]["scorer_kwargs"]
        )
        logging.info(
            f"uuids:{uuids}, method:{method}, score:{scores[method]}"
            + f", input:{to_tokenize}"
            + (
                f", generated:{tokenizer.batch_decode(o.sequences, skip_special_tokens=True)}"
                if method == "verbalized" or method == "surrogate_logit_score"
                else ""
            )
        )
    return scores
