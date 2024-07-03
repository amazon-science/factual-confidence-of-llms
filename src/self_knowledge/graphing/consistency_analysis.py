from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from torcheval.metrics.functional import binary_auprc


def main():
    # match uuids in test_d with uuids in popQA_d to get indices
    results_path = Path("data/reduced_sampling_pop")
    for f in results_path.glob("*/*/*.pt"):
        all_consistency = pd.DataFrame(torch.load(f))
        all_consistency = pd.DataFrame(all_consistency)
        model_family = f.parent.parent.name
        model_path = model_family + "/" + f.parent.name
        sf_results = "results_kn_popqa/" + model_path + "/popqa_sf_None_paraph.csv"
        sf_results = load_dataset("csv", data_files=sf_results)["train"].shuffle(
            seed=42
        )

        sf_results = pd.DataFrame(sf_results)
        # !! use of eval is unsafe. option to repair encoding issue
        if "tensor" in sf_results["uuid"][0]:
            sf_results["uuid"] = sf_results["uuid"].apply(lambda x: eval(x).item())

        all_consistency = all_consistency[
            all_consistency["uuids"].isin(sf_results["uuid"])
        ].sort_values("uuids")
        sf_results = sf_results[
            sf_results["uuid"].isin(all_consistency["uuids"])
        ].sort_values("uuid")
        all_consistency["generation_correct"] = [None] * len(all_consistency)
        for i, row in all_consistency.iterrows():
            for _, paraph in sf_results[sf_results["uuid"] == row["uuids"]].iterrows():
                if paraph["paraphrase"] in row["nli_answers"]:
                    if all_consistency["generation_correct"][i] is None:
                        all_consistency.loc[i, "generation_correct"] = [
                            paraph["generation_correct"]
                        ]
                    else:
                        all_consistency.loc[i, "generation_correct"].append(
                            paraph["generation_correct"]
                        )

        # drop None
        all_consistency = all_consistency.dropna(subset=["generation_correct"])
        # count how many paraphrases are correct
        all_consistency["count"] = all_consistency["generation_correct"].apply(
            lambda x: len(x)
        )
        print("n repetitions", len(all_consistency[all_consistency["count"] > 1]))
        # print average number in uuid groups
        print(
            "n paraphrases",
            all_consistency["uuids"].groupby(all_consistency["uuids"]).count().mean(),
        )
        all_consistency["generation_correct"] = all_consistency[
            "generation_correct"
        ].apply(lambda x: any(x))
        auprc = binary_auprc(
            torch.tensor(all_consistency["nli_mean_scores"].to_list()),
            torch.tensor(all_consistency["generation_correct"].to_list()),
        )
        print(model_path, auprc)


if __name__ == "__main__":
    main()
