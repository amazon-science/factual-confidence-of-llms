# the main important function to build any graph is collect_distributed, which comes from graph_utils.py
# it takes as input a path to a folder containing the results as .pt files, with the following naming convention:
# {method}_{accelerator.device.index}_{batch_idx}.pt (method can also be "labels" or "uuids" or "paraph" if paraphrase=True --> in that case paraph will not be a method but a list of paraphrases, in order)
# it outputs the folowing:
# scores: {method: [scores]}
# labels: [labels]
# uuids: [uuids]
# pop: [popularity]

# per_model_scores will take as input a path to a folder with multiple result folders, themselves containing the results as .pt files, and will output the following:
# df_s: a dataframe with the following columns: model, method, score
# corrs: a dictionary with the following keys: model, value: a dataframe with the correlation between methods


import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_from_disk
from torch import tensor
from torcheval.metrics.functional import binary_auprc

from self_knowledge.graphing.graph_utils import (
    collect_distributed,
    kl_per_template,
    lama_models_cleanup,
    remove_none_colnames,
    sort_per_template,
    wasserstein_per_template,
)


def get_entity_popularity(uuids, dataset_path, popularity_path):
    # use uuid to get text from dataset, then extract object by checking what comes before the relation
    df_dataset = pd.DataFrame(load_from_disk(dataset_path))
    # load tsv
    popularities = []
    pop_data = pd.read_csv(popularity_path, sep="\t")
    for uuid in uuids:
        uuid_data = df_dataset[df_dataset["uuid"] == uuid][0]
        # get object
        rel = uuid_data["template"].split("[X]")[-1].split("[Y]")[0]
        obj = uuid_data["text"].split(rel)[0]
        # get popularity
        popularity = pop_data[pop_data["obj"] == obj]["o_pop"].values
        if len(popularity) == 0:
            popularities.append(None)
        else:
            popularities.append(popularity[0])
    return popularities


def get_categ_scoring():
    print("WARNING: all paths are hard-coded. TODO: make them arguments")

    scoresx, labelsx, klx, wassersteinx, auprcx = order_categories(
        results_path="results/mixinst",
        dataset_path="data/balanced_trex_test",
        fast=True,
    )
    scores, labels, kl, wasserstein, auprc = order_categories(
        results_path="results/mist_btt3",
        dataset_path="data/balanced_trex_test",
        fast=True,
    )

    auprcx = auprcx.rename(columns={"verbalized_None": "verbalized"})

    print(auprcx - auprc)
    input()

    # general scores
    all_sc = {name: [] for name in scores.keys()}
    all_lb = {name: [] for name in scores.keys()}
    allkl = {name: [] for name in scores.keys()}
    all_wasserstein = {name: [] for name in scores.keys()}
    all_auprc = {
        name: pd.DataFrame(index=kl.index, columns=["auprc"]) for name in scores.keys()
    }

    for name in scores.keys():
        for row in scores[name]:
            all_sc[name].extend(row)
        for row in labels[name]:
            all_lb[name].extend(row)

        pos_idx = np.argwhere(np.array(all_lb[name]))
        neg_idx = np.argwhere(not np.array(all_lb[name]))
        pos_scores = np.array(all_sc[name])[pos_idx]
        neg_scores = np.array(all_sc[name])[neg_idx]
        allkl[name] = kl_per_template(pos_scores, neg_scores)
        all_wasserstein[name] = wasserstein_per_template(pos_scores, neg_scores)
        all_auprc[name] = binary_auprc(
            torch.tensor(all_sc[name]), torch.tensor(all_lb[name])
        ).item()
    # macro averages
    print("average kl for all ", pd.DataFrame(allkl, index=["kl"]))
    print("average wasserstein for all ", pd.DataFrame(all_wasserstein, index=["w"]))
    print("average auprc for all ", pd.DataFrame(all_auprc, index=["auprc"]))

    # micro averages
    print("micro kl mean:\n", kl.mean(axis=0), " micro kl std:", kl.std(axis=0))
    print(
        "micro wasserstein mean:\n",
        wasserstein.mean(axis=0),
        " micro wasserstein std:\n",
        wasserstein.std(axis=0),
    )
    print(
        "micro auprc mean:\n",
        auprc.mean(axis=0),
        " micro auprc std:",
        auprc.std(axis=0),
    )

    # pairwise correlations
    print("kl correlations:")
    print(kl.corr())
    print("wasserstein correlations:")
    print(wasserstein.corr())
    print("auprc correlations:")
    print(auprc.corr())


def entity_popularity_acc(
    results_path="results/btuuid2",
    out_path="graphs/btuuid2/entity_popularity_acc.png",
):
    """plots the accuracy depending on the popularity of the entity"""
    os.makedirs(Path(out_path).parent, exist_ok=True)
    scores, labels, _, pop = collect_distributed(results_path)
    # filter where pop is -1
    mask = pop != -1
    scores = {name: score[mask] for name, score in scores.items()}
    labels = labels[mask]
    pop = pop[mask]
    # everyone in dataframe
    df = pd.DataFrame(
        {
            "pop": pop,
            "label": labels,
            **{name: score for name, score in scores.items()},
        }
    )
    print(df.corr())
    # sort by popularity
    pop, indices = torch.sort(pop)
    scores = {name: score[indices] for name, score in scores.items()}
    labels = labels[indices]

    # split into 10 bins
    bins = torch.linspace(pop.min(), pop.max(), 10)
    # get the bin index for each pop
    bin_indices = torch.bucketize(pop, bins)
    # get the accuracy for each bin
    accs = []
    for i in range(len(bins) - 1):
        accs.append(
            {
                name: (score[bin_indices == i] - labels[bin_indices == i])
                .float()
                .abs()
                .mean()
                for name, score in scores.items()
            }
        )
    # plot lines, colourblind friendly
    for name in scores.keys():
        plt.plot(
            torch.arange(len(accs)),
            [acc[name].item() for acc in accs],
            label=name,
        )
    plt.legend()
    plt.savefig(out_path)
    plt.close()


def entity_popularity_kl(
    results_path="results/btuuid2",
    out_path="graphs/btuuid2/entity_popularity_kl.png",
):
    """plots the KL divergence between the distributions of true and false fact scores depending on the popularity of the entity"""
    os.makedirs(Path(out_path).parent, exist_ok=True)
    scores, labels, _, pop = collect_distributed(results_path)
    # filter where pop is -1
    mask = pop != -1
    scores = {name: score[mask] for name, score in scores.items()}
    labels = labels[mask]
    pop = pop[mask]
    # sort by popularity
    pop, indices = torch.sort(pop)
    scores = {name: score[indices] for name, score in scores.items()}
    labels = labels[indices]

    # split into 10 bins
    bins = torch.linspace(pop.min(), pop.max(), 10)
    # get the bin index for each pop
    bin_indices = torch.bucketize(pop, bins)
    # get the kl div for each bin between label 1 and 0
    kl_divs = []
    for i in range(len(bins) - 1):
        true_scores = {
            name: score[bin_indices == i][labels[bin_indices == i] == 1]
            for name, score in scores.items()
        }
        false_scores = {
            name: score[bin_indices == i][labels[bin_indices == i] == 0]
            for name, score in scores.items()
        }
        dist_true = {
            name: torch.histc(true_scores[name], bins=10, min=0, max=1)
            for name in scores.keys()
        }
        dist_false = {
            name: torch.histc(false_scores[name], bins=10, min=0, max=1)
            for name in scores.keys()
        }

        kl_divs.append(
            {
                name: torch.nn.functional.kl_div(
                    torch.nn.functional.softmax(dist_true[name], dim=0).log(),
                    torch.nn.functional.softmax(dist_false[name], dim=0),
                    reduction="batchmean",
                )
                for name in scores.keys()
            }
        )
    # plot lines, colourblind friendly
    for name in scores.keys():
        plt.plot(
            torch.arange(len(kl_divs)),
            [kl_div[name].item() for kl_div in kl_divs],
            label=name,
        )
    plt.legend()
    plt.savefig(out_path)
    plt.close()


def order_categories(results_path, dataset_path, fast=False):
    scores, labels, uuids, _ = collect_distributed(results_path)
    # we only use 1000 random elements
    if fast:
        indices = torch.randperm(len(uuids))[:1000]
        uuids = [uuids[i] for i in indices]
        labels = labels[indices]
        scores = {name: score[indices] for name, score in scores.items()}
    template_scores, template_labels, _ = sort_per_template(
        scores, labels, uuids, dataset_path
    )
    # sort by average acc = abs(score - label)
    df_ts = pd.DataFrame(template_scores)
    df_tl = pd.DataFrame(template_labels)
    # transpose to get templates as rows
    df_ts = df_ts.transpose()
    df_tl = df_tl.transpose()
    # all lists to np arrays
    kl = pd.DataFrame(index=df_ts.index, columns=df_ts.columns)
    wasserstein = pd.DataFrame(index=df_ts.index, columns=df_ts.columns)
    auprc = pd.DataFrame(index=df_ts.index, columns=df_ts.columns)
    for col in df_ts.columns:
        for i, row in enumerate(df_ts.iterrows()):
            pos_idx = np.argwhere(np.array(df_tl[col].iloc[i]))
            neg_idx = np.argwhere(not np.array(df_tl[col].iloc[i]))
            pos_scores = np.array(row[1][col])[pos_idx]
            neg_scores = np.array(row[1][col])[neg_idx]
            kl[col].iloc[i] = kl_per_template(pos_scores, neg_scores)
            wasserstein[col].iloc[i] = wasserstein_per_template(pos_scores, neg_scores)
            auprc[col].iloc[i] = binary_auprc(
                torch.tensor(row[1][col]), torch.tensor(df_tl[col].iloc[i])
            ).item()

    return df_ts, df_tl, kl, wasserstein, auprc


def boxplots(df):
    # boxplot each model, cluster per method
    sns.set_theme(palette="colorblind")
    # large plot
    plt.figure(figsize=(10, 5))

    sns.barplot(x="method", y="score", hue="models", data=df)
    # increase fontsize
    fontsize = 15
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel("Method", fontsize=fontsize)
    plt.ylabel("AUPRC", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig("graphs/barplot_test.png")


def per_model_scores(path, lama_cleanup=False, force_check_labels_path=None):
    # get all folders in path
    folders = Path(path).glob("*/*")
    # model name is folder name
    auprcs = {}
    corrs = {}
    for folder in sorted(folders):
        correct_logproduct = False
        if any(
            [
                x in str(folder)
                for x in [
                    "Mix",
                    "Mis",
                ]
            ]
        ):
            correct_logproduct = True
        scores, labels, uuids, _ = collect_distributed(
            folder, correct_logproduct=correct_logproduct
        )

        if scores is None:
            print("WARNING: no results for ", folder.stem)
            continue
        if force_check_labels_path is not None:
            if "tensor" in uuids[0].__str__():
                uuids = [uuid.item() for uuid in uuids]
            gen_data = pd.read_csv(
                Path(force_check_labels_path)
                / folder.parent.name
                / folder.name
                / "popqa_sf_None.csv"
            )
            gen_data["uuid"] = [eval(uuid).item() for uuid in gen_data["uuid"]]
            gen_data = gen_data[gen_data["uuid"].isin(uuids)]
            gen_data = gen_data.set_index("uuid").loc[uuids].reset_index()
            print(len(gen_data["uuid"]) == len(uuids))
            labels = tensor([bool(t) for t in gen_data["generation_correct"].values])
            print(labels)

        # get indexes of unique uuids
        _, idx = np.unique(uuids, return_index=True)
        # # keep only unique scores
        scores = {
            key: scores[key][idx]
            for key in scores.keys()
            if key not in ["paraph", "None_paraph"]
        }
        labels = labels[idx]
        if "yes" in scores.keys():
            scores["surrogate"] = scores["yes"] - scores["no"]
            # drop yes, no and maybe
            scores = {
                key: scores[key]
                for key in scores.keys()
                if key not in ["yes", "no", "maybe"]
            }
        # get correlation between all methods averaged over all models
        print("DEBUG:", folder.stem)
        print(
            "DEBUG: len of each key", [(key, len(scores[key])) for key in scores.keys()]
        )
        if len(scores.keys()) == 0:
            continue
        corr = torch.corrcoef(torch.stack([scores[key] for key in scores.keys()]))
        clean_keys = [key.replace("_None", "") for key in scores.keys()]
        clean_keys = [key.replace("hidden", "Trained Probe") for key in clean_keys]
        clean_keys = [key.replace("surrogate", "Surrogate") for key in clean_keys]
        clean_keys = [key.replace("verbalized", "Verbalized") for key in clean_keys]
        clean_keys = [
            key.replace("sequence_log_score", "Avg. Seq. Prob.") for key in clean_keys
        ]
        scores = {clean_keys[i]: scores[key] for i, key in enumerate(scores.keys())}
        corrs[folder.stem] = pd.DataFrame(corr, columns=clean_keys, index=clean_keys)
        auprc = {}
        for method in scores.keys():
            if not torch.is_tensor(scores[method]):
                scores[method] = torch.tensor(scores[method])
            auprc[method] = binary_auprc(scores[method], labels).item()
        auprcs[folder.stem] = auprc
    df_s = pd.DataFrame(auprcs)
    if lama_cleanup:
        df_s = lama_models_cleanup(df_s)
    df_s = df_s.transpose()

    df_s = remove_none_colnames(df_s)
    df_s = df_s.reset_index().melt(
        id_vars=["index"], var_name="method", value_name="score"
    )
    # rename index
    df_s = df_s.rename(columns={"index": "models"})
    return df_s, corrs


def corr_graphs(corrs):
    Path("graphs").mkdir(parents=True, exist_ok=True)
    for key in corrs.keys():
        sns.heatmap(
            corrs[key],
            mask=1 - np.tril(np.ones(corrs[key].shape)),
            annot=True,
            vmin=0,
            vmax=1,
            linewidth=5,
            cmap="crest",
        )
        plt.tight_layout()
        plt.savefig("graphs/" + key + "_heatmap.png")
        plt.close()

    # plot seaborn heatmap showing average and std of correlations
    sns.set_theme(palette="colorblind")
    # large plot
    plt.figure(figsize=(10, 8))
    # keep column and row names
    # drop corrs with less columns
    print(corrs.keys())
    corrs = {key: corrs[key] for key in corrs.keys() if len(corrs[key].columns) > 3}
    print(corrs.keys())
    _corrs = {
        col: [corrs[model][col].values for model in corrs.keys()]
        for col in corrs[list(corrs.keys())[0]].columns
    }
    _k = list(corrs.keys())[0]
    mean_corrs = {}
    sd_corrs = {}
    for col in _corrs.keys():
        mean_corrs[col] = np.mean(_corrs[col], axis=0)
        sd_corrs[col] = np.std(_corrs[col], axis=0)
    mean_corrs = pd.DataFrame(mean_corrs, index=corrs[_k].columns)
    print(mean_corrs)
    print(pd.DataFrame(sd_corrs))
    # no grid

    sns.set_style("whitegrid", {"axes.grid": False})
    ax = sns.heatmap(
        mean_corrs,
        mask=1 - np.tril(np.ones(mean_corrs.shape)),
        annot=True,
        vmin=0,
        vmax=1,
        linewidth=5,
        cmap="crest",
    )

    # get ordered std from triu
    sd_corrs = pd.DataFrame(sd_corrs, index=corrs[_k].columns)
    sd_corrs = sd_corrs.where(np.tril(np.ones(sd_corrs.shape)).astype(bool))
    sd_corrs = sd_corrs.stack().reset_index()
    print(sd_corrs)

    for i, t in enumerate(ax.texts):
        # 2 decimals
        t.set_text(t.get_text() + "±" + str(round(sd_corrs[0][i], 2)))
    # increase fontsize
    plt.tight_layout()
    plt.savefig("graphs/correlations.png")


if __name__ == "__main__":
    scores, corrs = per_model_scores(
        "results_pik2/", force_check_labels_path="results_kn_popqa/"
    )
    print(scores)
    print("correlations :")
    # set plot size
    sns.set(rc={"figure.figsize": (8, 6)})
    corr_graphs(corrs)
    boxplots(scores)
