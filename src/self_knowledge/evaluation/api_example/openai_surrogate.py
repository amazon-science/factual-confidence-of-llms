# uses the openai API to implement the surrogate score method.
import time

import datasets
import pandas as pd
import yaml
from openai import OpenAI
from tqdm import tqdm


def main():
    # Set up your OpenAI API credentials
    client = OpenAI()

    # load the dataset
    dataset_path = "data/reduced_trex_test"
    dataset = datasets.load_from_disk(dataset_path)
    dataset = dataset.select(range(0, 10000))
    prefix_file = "src/self_knowledge/prompts.yaml"
    affix = yaml.safe_load(open(prefix_file))["surrogate_logit_score"]
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
            n=10,
            stop=None,
            temperature=0.7,
        )

        generated_text = [response.choices[i].message.content.strip() for i in range(10)]
        # count number of yes, no and maybe
        # yes_score, no_score, maybe_
        yes_score = 0
        no_score = 0
        maybe_score = 0
        ood_score = 0
        for text in generated_text:
            if "yes" in text.lower():
                yes_score += 1 / 10
            elif "no" in text.lower():
                no_score += 1 / 10
            elif "maybe" in text.lower():
                maybe_score += 1 / 10
            else:
                ood_score += 1 / 10
        print("yes_score", yes_score)
        print("no_score", no_score)
        print("maybe_score", maybe_score)
        print("ood_score", ood_score)
        # to df save as csv
        df = pd.DataFrame(
            {
                **dataset[i],
                **{
                    "generated_text": generated_text,
                    "yes_score": yes_score,
                    "no_score": no_score,
                    "maybe_score": maybe_score,
                    "ood_score": ood_score,
                },
            },
            index=[i * 10 + j for j in range(10)],
        )
        df.to_csv("gpt35_verbalized_output.csv", mode="a", header=False)
        time.sleep(1)


if __name__ == "__main__":
    main()
