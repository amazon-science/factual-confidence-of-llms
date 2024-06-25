# uses AWS Translate to translate the dataset from English to Polish
# you must login to the API before running this script

import boto3
import datasets
import pandas as pd
import tqdm


def main():
    translate = boto3.client(
        service_name="translate", region_name="us-east-1", use_ssl=True
    )
    # load the dataset
    dataset_path = "data/reduced_trex_test"
    dataset = datasets.load_from_disk(dataset_path)
    dataset = dataset
    for i, data in tqdm.tqdm(enumerate(dataset)):
        result = translate.translate_text(
            Text=data["text"], SourceLanguageCode="en", TargetLanguageCode="pl"
        )
        _dic = {
            **data,
            "translated_text": result.get("TranslatedText"),
        }
        df = pd.DataFrame(_dic, index=[i])
        df.to_csv("data/polish_lama.csv", mode="a", header=(i == 0))


if __name__ == "__main__":
    main()
