from typing import Any, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer


def get_tokenizer(model_name: str, **kwargs) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_model(
    model_name: str, no_grad: bool = True, **kwargs: Any
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer = get_tokenizer(model_name=model_name)

    if no_grad:
        for p in model.parameters():
            p.requires_grad = False
    return model, tokenizer
