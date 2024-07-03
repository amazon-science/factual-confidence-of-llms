# select a random member of every uuid group
# compute auprc for each method
# compute variation of auprc when sampling different paraphrases

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torcheval.metrics.functional import binary_auprc

from self_knowledge.graphing.graph_utils import collect_distributed


def main():
    p = Path("results_pt_paraph")
    auprcs = pd.DataFrame(columns=["AUPRC", "model", "method"])

    for f in p.glob("*/*"):
        s, l, uuids, _ = collect_distributed(
            f, correct_logproduct="Mix" in f.stem or "Mis" in f.stem
        )
        if s is None:
            print(f"Skipping {f}")
            continue
        if "yes" in s.keys():
            s["Surrogate"] = s["yes"] - s["no"]
            # drop yes, no and maybe
            s = {key: s[key] for key in s.keys() if key not in ["yes", "no", "maybe"]}
        for _ in range(10):
            df = pd.DataFrame()
            for key in s:
                if key not in ["paraph", "None_paraph"]:
                    df[key] = s[key].tolist()
            df["uuid"] = uuids
            df["label"] = l.tolist()
            # select a random member of every uuid group
            df = (
                df.groupby("uuid")
                .apply(lambda x: x.sample(1), include_groups=False)
                .reset_index(drop=True)
            )
            for method in df.keys():
                if method not in ["paraph", "None_paraph", "uuid", "label"]:
                    auprc = binary_auprc(
                        torch.tensor(df[method]), torch.tensor(df["label"])
                    ).item()
                    auprcs = pd.concat(
                        [
                            auprcs,
                            pd.DataFrame(
                                {
                                    "AUPRC": [auprc],
                                    "model": [f.stem],
                                    "method": [method],
                                }
                            ),
                        ]
                    )

    # plot all models and all methods in same plot
    sns.set_theme(style="whitegrid")
    sns.set_palette("colorblind")
    sns.set_context("paper")
    # remove None from method keys
    auprcs["method"] = auprcs["method"].apply(lambda x: x.replace("_None", ""))
    # replace verbalized by Verbalised
    auprcs["method"] = auprcs["method"].apply(
        lambda x: x.replace("verbalized", "Verbalised")
    )
    auprcs["method"] = auprcs["method"].apply(
        lambda x: x.replace("hidden", "Trained probe")
    )
    auprcs["method"] = auprcs["method"].apply(
        lambda x: x.replace("sequence_log_score", "Avg. Seq. Prob.")
    )

    # reorder models
    auprcs["model"] = pd.Categorical(auprcs["model"], auprcs["model"].unique())
    # model subplots
    n = len(auprcs["method"].unique())
    fig, ax = plt.subplots(1, n, figsize=(3 * n, 5))

    for i, model in enumerate(auprcs["method"].unique()):
        # add a subplot for each model
        ax = plt.subplot(1, n, i + 1)
        ax.set_title(model)
        # plot all methods for this model
        sns.boxplot(
            data=auprcs[auprcs["method"] == model],
            x="model",
            y="AUPRC",
            ax=ax,
            hue="model",
        )
        # 45 degree legend
        plt.xticks(rotation=45, ha="right")
        # all y axis are the same
        plt.ylim(0.2, 1)
        if i != 0:
            plt.ylabel("")
    plt.tight_layout()
    plt.savefig("graphs/auprc_paraph_pt.png")


if __name__ == "__main__":
    main()
