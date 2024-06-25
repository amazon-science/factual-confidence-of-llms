# Description: This script uses the OpenAI API to generate verbalized scores for the T-REx dataset.
import time

import datasets
import pandas as pd
import yaml
from openai import OpenAI
from tqdm import tqdm

from self_knowledge.evaluation.verbalized_utils import number_parser


def main():
    # Set up your OpenAI API credentials
    client = OpenAI()

    # load the dataset
    dataset_path = "data/reduced_trex_test"
    dataset = datasets.load_from_disk(dataset_path)
    dataset = dataset.select(range(0, 10000))
    prefix_file = "src/self_knowledge/prompts.yaml"
    affix = yaml.safe_load(open(prefix_file))["verbalized"]
    for i, data in enumerate(tqdm(dataset)):
        messages = [
            {
                "role": "system",
                "content": "provide a short answer to the following question:",
            },
            {
                "role": "user",
                "content": affix["prefix"] + data["text"] + affix["suffix"],
            },
        ]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=5,
            n=1,
            stop=None,
            temperature=0.7,
        )

        generated_text = response.choices[0].message.content.strip()
        score = number_parser(generated_text)

        df = pd.DataFrame(
            {
                **dataset[i],
                **{
                    "generated_text": generated_text,
                    "verbalized_score": score,
                },
            },
            index=[i * 10 + j for j in range(10)],
        )

        df.to_csv("gpt35_verbalized_output.csv", mode="a", header=False)
        time.sleep(1)


if __name__ == "__main__":
    main()
