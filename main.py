import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_config,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.utils.data import DataLoader

# accelerator
accelerator = Accelerator()

### replace by a model class
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model(model_name="EleutherAI/pythia-70m", activate_peft=True):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # if not hasattr(tokenizer, "pad_token"):
    tokenizer.pad_token = tokenizer.eos_token

    if activate_peft:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    return model, tokenizer


### end model class


### start evaluation class
def eval1(model, tokenizer, dataloader=None):
    # We iterate through the dataset, looking at logit score given to whole statements
    scores = torch.zeros(len(dataloader) * dataloader.batch_size)  # maybe too huge to be realistic?
    for i, batch in enumerate(dataloader):
        # get seq_length
        batch = batch["text"]  # temporary hack, to be corrected in data pipeline

        _tok_batch = tokenizer(batch, return_tensors="pt", padding=True)
        o = model(**_tok_batch)
        # get the logit score for the whole statement
        logits = o.logits[:, :-1, :]
        probs = logits.softmax(-1)
        tokens_probs = torch.gather(probs, 2, _tok_batch["input_ids"][:, 1:, None]).squeeze(-1)
        # mask to replace padding tokens by identity multiplications
        mask = _tok_batch["input_ids"] == tokenizer.pad_token_id
        masked_token_probs = tokens_probs.masked_fill(mask, 1.0)
        # compute the probability of the whole statement
        minibatch_probs = masked_token_probs.prod(
            -1
        )  # If this gets too small, then stay in log space and use sum
        scores[i * dataloader.batch_size : (i + 1) * dataloader.batch_size] = minibatch_probs.cpu()
    return scores


def eval2(model, tokenizer, dataloader=None):
    # Value Head method
    # take masked input, output new tokens, train the value-head to point out true when true, false when false
    # !! Beware, there is a train/test split to be made
    raise NotImplementedError()
    return scores


def eval2(model, tokenizer, dataloader=None):
    # Value Head method
    # take masked input, output new tokens, train the value-head to point out true when true, false when false
    # !! Beware, there is a train/test split to be made
    raise NotImplementedError()
    return scores


def eval3(model, tokenizer, dataloader=None, prompted=True):
    # surrogate logits - add True/False at the end, then check the logit probability of either being output
    scores = torch.zeros(len(dataloader) * dataloader.batch_size)
    for i, batch in enumerate(dataloader):
        # get seq_length
        batch = batch["text"]  # temporary hack, to be corrected in data pipeline
        _tok_batch = tokenizer(batch, return_tensors="pt", padding=True)
        o = model(**_tok_batch)
        # get the logit score for the last word
        logits = o.logits[:, -1:, :]
        probs = logits.softmax(-1)
        tokens_probs = torch.gather(probs, 2, _tok_batch["input_ids"][:, :, None]).squeeze(-1)
        # mask to replace padding tokens by identity multiplications
        mask = _tok_batch["input_ids"] == tokenizer.pad_token_id
        masked_token_probs = tokens_probs.masked_fill(mask, 1.0)
        # compute the probability of the whole statement
        minibatch_probs = masked_token_probs.prod(
            -1
        )  # If this gets too small, then stay in log space and use sum
        scores[i * dataloader.batch_size : (i + 1) * dataloader.batch_size] = minibatch_probs.cpu()
    return scores


### end evaluation class


### start data class
def question_to_statement():
    # takes as input a question + answer format and outputs a concatenation of both
    raise NotImplementedError()
    pass


def format_true_trex(example):
    out_text = example["masked_sentence"].replace("[MASK]", example["obj_surface"])
    return {"text": out_text}


def format_template_trex(example):
    out_text = (
        example["template"]
        .replace("[X]", example["sub_label"])
        .replace("[Y]", example["obj_surface"])
    )
    return {"text": out_text}


class FalseTrex:
    def __init__(self, dataset):
        self.dataset = dataset

    def format_false_trex(self, example):
        # select a random obj_surface that is not the same as the obj_surface in the example, but does have the same template
        random_obj_surface = example["obj_surface"]
        while random_obj_surface == example["obj_surface"]:
            random_obj_surface = self.dataset[np.random.randint(0, len(self.dataset))][
                "obj_surface"
            ]
        out_text = example["masked_sentence"].replace("[MASK]", random_obj_surface)
        return {"text": out_text}


### end data class


if __name__ == "__main__":
    dataset = load_dataset("lama", "trex", split="train")  # , streaming=True)
    # with t-rex we can easily do three things.
    # 1) use the masked sentence as a true statement to be evaluated
    # 2) use the negated template as a false statement to be evaluated
    # 3) use the template as a true statement to be evaluated
    # additionaly a more complex false statement can be built by replacing the expected object by a random one in the same category

    # look into this remove column thing - can it be done from within the formatting function?
    template_dataset = dataset.map(
        format_template_trex
    )  # this is sub-optimal, let's go with dataloaders
    dataloader = DataLoader(template_dataset, batch_size=10)
    model, tokenizer = get_model(activate_peft=False)
    for p in model.parameters():
        p.requires_grad = False
    sc = eval1(model, tokenizer, dataloader)
    print(sc.mean())
