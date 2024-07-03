# dataset preparation utils

import json
import logging
import random
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np


def get_lpaqa_paraphrase(
    example,
    lpaqa_folder="./lpaqa_prompts",
    paraphrase_type="paraphrase",
):
    assert paraphrase_type in ["manual_paraphrase", "mine", "paraphrase"]
    lpaqa_file = (
        Path(lpaqa_folder) / paraphrase_type / (example["predicate_id"] + ".jsonl")
    )
    assert lpaqa_file.exists(), f"No such file {lpaqa_file}"
    # load json on random line in file
    with open(lpaqa_file, "r") as f:
        if lpaqa_file.suffix == ".jsonl":
            file = f.readlines()
            template = json.loads(file[np.random.randint(1, len(file))])["template"]
            if template != example["template"]:
                return template
        logging.warning("No paraphrase was found that was not in the exclusion list.")


def lpaqa_paraphrases(example):
    """provide lpaqa paraphrase of trex. simple version. Effet de bord, template is changed to paraphrased version"""
    paraphrase_template = get_lpaqa_paraphrase(example)
    return {
        **format_template_trex(
            {
                "template": paraphrase_template,
                "sub_label": example["sub_label"],
                "obj_surface": example["obj_surface"],
            }
        ),
        "paraphrase_template": paraphrase_template,
        "is_factual": True,
    }


def format_context_trex(example):
    """
    as before but using the contextualised version of the template
    takes a contextualised sentence like "Dirty Hands (French: Les Mains sales) is a play by [MASK]" and replace X and Y to get "Dirty Hands (French: Les Mains sales) is a play by Jean-Paul Sartre"
    """
    out_text = example["masked_sentence"].replace("[MASK]", example["obj_surface"])
    return {"text": out_text}


def format_template_trex(example):
    """
    Basic Lama format
    """
    fact = (
        example["template"]
        .replace("[X]", example["sub_label"])
        .replace("[Y]", example["obj_surface"])
    )
    return {"text": fact, "is_factual": True}


def format_slot_filling_trex(example):
    """
    Format to query the model for confidence
    """
    fact = (
        example["template"]
        .replace("[X]", example["sub_label"])
        .replace("[Y]", "")
        .replace(" .", '"')
    )
    return {"text": fact}


class FalseTrex:
    def __init__(self, dataset, paraphrase_type=None):
        """
        Main class to format the dataset using both factual from lama trex and false examples.

        Parameters
        ----------
        dataset : datasets.Dataset
            the dataset to use
        paraphrase_type : str
            the type of paraphrase to use. Can be "manual_paraphrase", "mine" or "paraphrase"
        """
        self.dataset = dataset
        self.paraphrase_type = paraphrase_type
        # as dict for faster lookup and replace in formvat_false_trex
        self.dict_dataset = defaultdict(list)
        for idx, example in enumerate(self.dataset):
            if example["template"] in self.dict_dataset.keys():
                self.dict_dataset[example["template"]].append(idx)
            else:
                self.dict_dataset[example["template"]] = [idx]

    def format_false_trex(self, example):
        """
        select a random obj_surface that is not the same as the obj_surface in the example but within the same category
        """
        random_obj_surface = example["obj_surface"]
        while True:
            random_obj_surface = self.dataset[
                np.random.choice(self.dict_dataset[example["template"]], 1)
            ]["obj_surface"][0]
            if random_obj_surface != example["obj_surface"]:
                break
        out_text = (
            example["template"]
            .replace("[Y]", random_obj_surface)
            .replace("[X]", example["sub_label"])
        )
        # generate a new uuid for the new example
        uuid_ = str(uuid.uuid5(uuid.NAMESPACE_DNS, out_text))
        return {"text": out_text, "uuid2": uuid_, "is_factual": False}

    def format_balanced_trex(self, example):
        """
        select a random obj_surface that is not the same as the obj_surface in the example but within the same category
        """
        is_factual = random.choice([True, False])
        if self.paraphrase_type is not None:
            template = get_lpaqa_paraphrase(
                example, paraphrase_type=self.paraphrase_type
            )
        else:
            template = example["template"]
        if is_factual:
            out_dict = format_template_trex(
                {
                    "template": template,
                    "sub_label": example["sub_label"],
                    "obj_surface": example["obj_surface"],
                }
            )
            return out_dict

        out_dict = self.format_false_trex(example)
        out_dict["is_factual"] = is_factual
        return out_dict
