# once paraphrase are generated, this script selects only generations that are not in the original dataset, and which preserve meaning
# scoring methods are also available to measure syntactic or lexical distance from the original sentence

from pathlib import Path

import pandas as pd
import spacy
from datasets import load_dataset
from tqdm import tqdm

from self_knowledge.data_gen.paraphrase.nli import NLI


def filter_raw_paraphrase(raw_paraph, text, nli=None):
    # remove last line if no full stop
    if "." not in raw_paraph.split("\n")[-1] or "?" not in raw_paraph.split("\n")[-1]:
        raw_paraph = raw_paraph.rsplit("\n", 1)[0]
    raw_paraph = raw_paraph.rsplit("New sentences:", 1)[1]
    clean_paraphs = raw_paraph.replace("\n", "").split("- ")[1:]
    if nli is not None and len(clean_paraphs) > 1:
        nli_mask = nli.check_equivalence(clean_paraphs, [text] * len(clean_paraphs))
        clean_paraphs = list(set([p for p, m in zip(clean_paraphs, nli_mask) if m and p != text]))
    return clean_paraphs


def syntactic_variation(paraph, text):
    # use spacy em_core_web_md
    spacy_model = spacy.load("en_core_web_md")
    # get syntactic variation
    p = spacy_model(paraph)
    t = spacy_model(text)
    # get syntactic variation
    p_s = [token.pos_ for token in p]
    t_s = [token.pos_ for token in t]
    # get syntactic variation
    synt_var = [p_s[i] != t_s[i] for i in range(len(p_s))] / len(p_s)
    return synt_var


def lexical_variation(paraph, text, tokenizer):
    """this one expects exact matching"""
    # get lexical variation
    p = tokenizer.tokenize(paraph)
    t = tokenizer.tokenize(text)
    # get lexical variation
    lexical_var = [p[i] != t[i] for i in range(len(p))] / len(p)
    return lexical_var


def lexical_variation_2(paraph, text):
    """this one expects bag of words matching"""
    t = text.split()
    score = 0
    # get lexical variation
    for p in paraph.split():
        if p in t:
            t.remove(p)
            score += 1
    lexical_var = score / len(paraph.split())
    return lexical_var


def main():
    path = "data/reduced_mix_lama_paraph"
    save_path = "data/reduced_mix_lama_paraph/"
    Path(save_path).mkdir(parents=True, exist_ok=True)

    paths = Path(path).glob("*_0*.csv")
    nli = NLI()
    concatenated_data = []
    for path in tqdm(paths):
        dataset = pd.DataFrame(load_dataset("csv", data_files=str(path), delimiter=",")["train"])
        # unique
        dataset = dataset.drop_duplicates(subset=["text"])
        dataset["paraphrase"] = dataset.apply(
            lambda x: filter_raw_paraphrase(x["paraphrase"], x["text"], nli=nli), axis=1
        )
        # remove empty paraphrase
        dataset = dataset[dataset["paraphrase"].map(len) > 0]
        dataset = dataset.explode("paraphrase")

        concatenated_data.append(dataset)
        print(dataset.head(5))
        # concatenate
        pd.concat(concatenated_data).to_csv(save_path + "clean.csv")


if __name__ == "__main__":
    main()
