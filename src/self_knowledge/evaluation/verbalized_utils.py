import logging
import re

import torch
from transformers import PreTrainedTokenizerFast
from word2number import w2n


def number_parser(text):
    """
    use re to check for digits
    use w2n to convert to numbers
    return the number
    Parameters
    ----------
    text: str
        the text to convert to a number
    Returns
    -------
    float
        the number converted from the text
    """
    _re = re.findall(r"[+-]?(\d*\.\d+|\d+)", text)
    if len(_re) != 0:
        if len(_re) > 1:
            logging.info(
                f"More than one number found in {text}. Overriding error and selecting the first."
            )
        return float(_re[0])

    try:
        return w2n.word_to_num(text)
    except ValueError:
        logging.info("ValueError")
        logging.info(
            f"Error override. Could not convert to number, setting to -1. text: {text}"
        )
    return -1


def confidence_in_words(
    o, tokenizer: PreTrainedTokenizerFast = None, device: str = "cpu"
):
    if tokenizer is not None:
        out_text = tokenizer.batch_decode(o.sequences)
    else:
        out_text = o
    return torch.tensor([number_parser(t.split("A:")[-1]) for t in out_text]).to(device)
