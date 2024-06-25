# generates paraphrases from a a given dataset
from pathlib import Path

import pandas as pd
import torch
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from self_knowledge.arch import get_model

accelerator = Accelerator()


class paraphraser(torch.nn.Module):
    # we use generative models to paraphrase
    def __init__(self, model, tokenizer) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, prompt, batch, max_new_tokens):
        batch = [prompt.replace("$sentence", t) for t in batch["text"]]
        encoded_source_sentence = self.tokenizer(batch, padding=True, return_tensors="pt")
        for k, v in encoded_source_sentence.items():
            encoded_source_sentence[k] = v.to(accelerator.device)
        generated_target_tokens = self.model.generate(
            **encoded_source_sentence,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        target_sentence = self.tokenizer.batch_decode(
            generated_target_tokens, skip_special_tokens=True
        )
        return target_sentence


def paraphrase(model_name, dataset_path, prompt, batch_size=100, save_frq=100, save_path=None):
    Path(save_path).mkdir(exist_ok=True, parents=True)
    model, tokenizer = get_model(model_name, load_in_4bit=True, device_map="balanced")
    # check if deepspeed is setup on accelerator
    paraphraser_model = paraphraser(model, tokenizer)
    if ".csv" not in dataset_path:
        dataset = load_from_disk(dataset_path)
    else:
        dataset = load_dataset("csv", data_files=dataset_path, split="train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    model, tokenizer, dataloader = accelerator.prepare(model, tokenizer, dataloader)

    # index through df, add paraphrase to new column
    batches, paraphs = [], []
    for i, batch in enumerate(tqdm(dataloader)):
        paraph = paraphraser_model(prompt, batch, 100)
        # batch to cpu to df
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cpu()
        batch = pd.DataFrame(batch)
        batches.append(batch)
        paraphs.append(pd.DataFrame(paraph, columns=["paraphrase"]))
        if len(batches) % save_frq == 0 or len(batches) == len(dataloader):
            df_batches = pd.concat(batches)
            df_batches["paraphrase"] = pd.concat(paraphs)
            df_batches.to_csv(
                Path(save_path) / f"{accelerator.device.index}_paraph_{i//save_frq:03d}.csv"
            )
            batches, paraphs = [], []
    df_batches["paraphrase"] = paraphs
    return df_batches


if __name__ == "__main__":
    prompt = "Given a sentence, generate paraphrases of it as follows:\n\t- You can change and/or add words, and/or change the syntactic structure of the sentence;\n\t- Make sure the new sentence does not add additional details with respect to the original.\n\t- Make sure the new sentence does not omit any details with respect to the original.\n\t- Make sure the new sentence is natural and plausible, in spite of the changes.\n\t- Do not generate the original sentence or previously generated ones.\nList your paraphrases as bulletpoint.\nSentence: $sentence\nNew sentences:"
    paraphrases = paraphrase(
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "data/reduced_trex_test",
        prompt,
        save_path="data/reduced_mix_lama_paraph",
    )
