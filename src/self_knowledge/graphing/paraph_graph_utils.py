from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torcheval.metrics.functional import binary_auprc

from self_knowledge.graphing.graph_utils import (
    collect_distributed,
    remove_none_colnames,
)


def get_popqa_pop(uuids, popqa_path):
    if "tensor" in uuids[0]:
        uuids = [eval(u).item() for u in uuids]
    popqa = pd.read_csv(popqa_path)
    popqa = popqa[popqa["uuid"].isin(uuids)]
    # reorder popqa to match uuids
    popqa = popqa.set_index("uuid").loc[uuids].reset_index()
    return popqa["o_pop"]


def mean_improvement(results_path):
    """
    micro average over same uuids vs macro average over all
    """
    files = Path(results_path).glob("*/*")
    dump = []
    # prepare one subplot for each file
    files = list(files)
    n = len(files)
    fig, ax = plt.subplots(2, n // 2, figsize=(12, 6))
    all_trained_df = pd.DataFrame()
    for file in files:
        dump.append(file.stem)
        s, l, u, pop = collect_distributed(
            file, correct_logproduct="Mix" in file.stem or "Mis" in file.stem
        )

        if s is None:
            continue
        s = {k: v.to(torch.float32) for k, v in s.items() if "paraph" not in k}
        if "yes" in s.keys():
            s["surrogate"] = s["yes"] - s["no"]
            # drop yes, no and maybe
            s = {key: s[key] for key in s.keys() if key not in ["yes", "no", "maybe"]}
        df_s = pd.DataFrame(s)
        df_s = remove_none_colnames(df_s)
        df_s["uuid"] = u
        df_s["popularity"] = pop
        df_s["label"] = l
        auprcs = {}
        # macro
        # rename columns verbalized Verbalised and hidden Trained probe
        df_s.rename(
            columns={
                "verbalized": "Verbalised",
                "hidden": "Trained probe",
                "sequence_log_score": "Avg. Seq. Prob.",
                "surrogate": "Surrogate",
            },
            inplace=True,
        )
        for method in df_s.columns:
            if method in ["label", "uuid"]:
                continue
            auprc = binary_auprc(
                torch.tensor(df_s[method]), torch.tensor(df_s["label"])
            )
            auprcs[method] = auprc.item()
        dump.append(auprcs)
        auprcs2 = {}

        # micro
        for method in df_s.columns:
            if method in ["label", "uuid"]:
                continue
            group = df_s[[method, "label", "uuid"]].groupby("uuid").max()
            auprc = binary_auprc(
                torch.tensor(group[method]), torch.tensor(group["label"])
            )
            auprcs2[method] = auprc.item()
        dump.append(auprcs2)
        variation = {}
        sns.set(style="whitegrid")
        sns.set_palette("colorblind")

        for method in df_s.columns:
            # variation[method] = {}
            if method in ["label", "uuid", "popularity"]:
                continue

            # min max normalization
            df_s[method] = (df_s[method] - df_s[method].min()) / (
                df_s[method].max() - df_s[method].min()
            )

            group = df_s[[method, "uuid"]].groupby("uuid").std()

            # per method std
            variation[method] = group[method].mean()

        # merge all but uuid and label under one column called method
        df_s = df_s.melt(
            id_vars=["uuid", "label", "popularity"],
            var_name="method",
            value_name="Score",
        )
        # show method name for each hue
        ax = plt.subplot(2, n // 2, files.index(file) + 1)

        ax.set_title(file.stem)
        ax.set_ylim(0, 0.6)
        df_s["model"] = [file.stem] * len(df_s)
        all_trained_df = pd.concat(
            [all_trained_df, df_s[df_s["method"] == "Trained probe"]]
        )

        #
        # remove all but the outermost y axis
        if files.index(file) % (n // 2) != 0:
            ax.set_ylabel("")
            ax.yaxis.set_visible(False)
        dump.append(variation)
        group_l = df_s[["label", "uuid"]].groupby("uuid").std()
        dump.append(group_l["label"].mean())
    plt.tight_layout()
    plt.savefig("graphs/std_dist.png")
    plt.close()

    # plot all models of trained probe in same plot
    data = (
        all_trained_df[["model", "uuid", "Score", "label"]]
        .groupby(["uuid", "model"])
        .std()
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    nbins = 10

    # get index of each bin for each model
    # get the average popularity of each bin
    mean_pops = (
        all_trained_df[["model", "uuid", "Score", "label"]]
        .groupby(["uuid", "model"])
        .max()
    )
    ys = []

    for model in all_trained_df["model"].unique():
        sns.kdeplot(
            x=data.xs(model, level="model")["Score"],
            y=mean_pops.xs(model, level="model")["Score"],
            ax=ax,
            label=model,
            fill=True,
        )

        std_index = np.digitize(
            data.xs(model, level="model")["Score"], bins=np.linspace(0, 0.6, nbins)
        )
        _mod = mean_pops.xs(model, level="model")

        y = []
        x = []
        for i in range(0, nbins):
            labs = _mod[std_index == i]["label"]
            if len(labs) >= 15:
                y.append(sum(labs) / len(labs))

            else:
                y.append(None)
            x.append(data.xs(model, level="model")[std_index == i]["Score"].mean())

        ys.append(y)

    df_ys = pd.DataFrame(
        ys,
        index=all_trained_df["model"].unique(),
        columns=np.linspace(0, 0.6, nbins) - 0.6 / nbins,
    ).T
    df_ys = df_ys.melt(var_name="model", value_name="popularity", ignore_index=False)
    # index is now x
    df_ys["x"] = df_ys.index
    df_ys = df_ys.reset_index(drop=True)
    # plot mean popularity for each bin
    plt.title("Trained probe")
    plt.tight_layout()
    # remove grid
    # rename x axis: Standard Deviation
    ax.set_xlabel("Standard Deviation")
    ax.grid(False)

    plt.savefig("graphs/trained_probe_std_dist_pik.png")
    plt.close()


if __name__ == "__main__":
    results_path = "results_pik_paraph2/"
    mean_improvement(results_path)
