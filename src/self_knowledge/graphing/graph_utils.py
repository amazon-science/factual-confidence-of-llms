import glob
import json
import os
import re
from pathlib import Path

import pandas as pd
import torch
from datasets import load_from_disk
from scipy import stats


def sample_save_as_json(score_path: str = "", output_path: str = None):
    os.makedirs(Path(output_path).parent, exist_ok=True)
    selected_scores = {}
    random_indices = None
    for file in glob.glob(f"{score_path}/*_labels.pt"):
        score = torch.load(file[:-10] + ".pt")
        method_name = Path(file[:-10]).stem
        if random_indices is None:
            # select 1000 random samples while maintaining order
            random_indices = torch.randperm(len(score))[:1000].sort().values.tolist()
            selected_scores["labels"] = torch.load(file)[random_indices].tolist()
        score = score[random_indices]
        selected_scores[method_name] = score.tolist()
        if "verbalized" in file:
            score[score > 100] = 0
        if (
            "log" in file
            or "yes" in file
            or "no" in file
            or "maybe" in file
            or "verbalized" in file
        ):
            # normalize to 0-1
            score = (score - score.min()) / (score.max() - score.min())
        if "no" in file:
            score = 1 - score

        if "hidden" in file:
            score = torch.nn.functional.sigmoid(score)
        selected_scores[method_name + "_normalized"] = score.tolist()
        score[selected_scores["labels"] == 0] = (
            1 - score[selected_scores["labels"] == 0]
        )
        selected_scores[method_name + "_acc"] = score.tolist()

    with open(output_path, "w") as f:
        json.dump(selected_scores, f)


def post_process_scores(files, percentage_verbalized=False):
    scores = []
    labels = []
    names = []
    for file in files:
        if Path(file).is_file():
            label = torch.load(file)
            score = torch.load(file[:-10] + ".pt")
            if "verbalized" in file and percentage_verbalized:
                score[score > 100] = 0
            if (
                "log" in file
                or "yes" in file
                or "no" in file
                or "maybe" in file
                or "verbalized" in file
            ):
                # normalize to 0-1
                score = (score - score.min()) / (score.max() - score.min())
            if "no" in file:
                score = 1 - score

            if "hidden" in file:
                score = torch.nn.functional.sigmoid(score)

            scores.append(score)
            names.append(Path(file[:-10]).stem)
            labels.append(label)
    return scores, labels, names


def collect_distributed(
    results_path,
    identifier=".pt",
    correct_logproduct=False,
):
    """
    results_path: path to folder containing results
    identifier: string found in all files to be collected
    """
    # glob inside the folder
    files = Path(results_path).glob(f"*{identifier}*")
    scores = {}
    labels = {}
    for file in sorted(files):
        stem = str(file.stem)
        # find the gpu number in the file name
        gpu_id = re.findall(r"\d", stem)
        if len(gpu_id) == 0:
            print(f"Could not find gpu id in {stem}")
            continue
        gpu_id = gpu_id[0]
        if "labels" in stem or "uuids" in stem or "pop" in stem:
            if gpu_id not in labels.keys():
                labels[gpu_id] = {}

            if "labels" in stem:
                if "labels" not in labels[gpu_id].keys():
                    labels[gpu_id]["labels"] = torch.load(file, map_location="cpu")
                else:
                    labels[gpu_id]["labels"] = torch.cat(
                        [labels[gpu_id]["labels"], torch.load(file, map_location="cpu")]
                    )
            if "uuids" in stem:
                if "uuids" not in labels[gpu_id].keys():
                    labels[gpu_id]["uuids"] = torch.load(file, map_location="cpu")
                else:
                    labels[gpu_id]["uuids"].extend(torch.load(file, map_location="cpu"))
            if "s_pop" in stem:
                if "pop" not in labels[gpu_id].keys():
                    labels[gpu_id]["pop"] = torch.load(file, map_location="cpu")
                else:
                    labels[gpu_id]["pop"] = torch.cat(
                        [labels[gpu_id]["pop"], torch.load(file, map_location="cpu")]
                    )
        else:
            score = torch.load(file, map_location="cpu")
            if "surrogate" in stem:
                # ['Yes', ' Yes', 'yes', ' yes', 'No', ' No', 'no', ' no', 'Maybe', ' Maybe', 'maybe', ' maybe']
                # take maximum of yes, no, maybe
                if correct_logproduct:
                    if score.shape[0] == 149:
                        # add one dummy to get to 150
                        score = torch.cat([score, torch.ones(1, score.shape[1])])
                    try:
                        score = score.reshape(-1, 3, 50)
                        score = score.prod(1)
                    except Exception as e:
                        print(e)
                        score = score.reshape(-1, 3, score.shape[0] // 32)
                        score = score.prod(1)
                        print(score.shape)
                        # continue
                try:
                    score = score.reshape(-1, 3, 4).max(-1).values
                except Exception as e:
                    print(e)
                    print(score.shape)
                    continue

                if "yes" not in scores.keys():
                    scores["yes"] = {}
                    scores["no"] = {}
                    scores["maybe"] = {}
                if gpu_id not in scores["yes"].keys():
                    scores["yes"][gpu_id] = score[:, 0]
                    scores["no"][gpu_id] = score[:, 1]
                    scores["maybe"][gpu_id] = score[:, 2]
                else:
                    scores["yes"][gpu_id] = torch.cat(
                        [scores["yes"][gpu_id], score[:, 0]]
                    )
                    scores["no"][gpu_id] = torch.cat(
                        [scores["no"][gpu_id], score[:, 1]]
                    )
                    scores["maybe"][gpu_id] = torch.cat(
                        [scores["maybe"][gpu_id], score[:, 2]]
                    )
            else:
                name = Path(file).stem.split(gpu_id)[0][:-1]
                if name not in scores.keys():
                    scores[name] = {}
                if gpu_id not in scores[name].keys():
                    scores[name][gpu_id] = score
                else:
                    # check type
                    if isinstance(scores[name][gpu_id], torch.Tensor):
                        scores[name][gpu_id] = torch.cat([scores[name][gpu_id], score])

    # merge all gpu results
    if len(scores.keys()) == 0:
        print("No scores found")
        return None, None, None, None
    for name in scores.keys():
        if isinstance(scores[name][list(scores[name].keys())[0]], torch.Tensor):
            scores[name] = torch.cat(
                [scores[name][gpu_id] for gpu_id in scores[name].keys()]
            )
    # merge all labels
    merged_labels = None
    for gpu_id in labels.keys():
        if merged_labels is None:
            merged_labels = labels[gpu_id]["labels"]
            merged_uuids = labels[gpu_id]["uuids"]
            if "pop" in labels[gpu_id].keys():
                merged_pops = torch.tensor(labels[gpu_id]["pop"])
        else:
            merged_labels = torch.cat([merged_labels, labels[gpu_id]["labels"]])
            if "pop" in labels[gpu_id].keys():
                merged_pops = torch.cat([merged_pops, labels[gpu_id]["pop"]])
            merged_uuids.extend(labels[gpu_id]["uuids"])
        if "pop" not in labels[gpu_id].keys():
            merged_pops = torch.tensor([-1] * len(merged_labels))

    return scores, merged_labels, merged_uuids, merged_pops


def sort_per_template(scores, labels, uuids, dataset_path, chosen_relations=[]):
    # sort by template. we get template by checking uuid against dataset
    df_dataset = pd.DataFrame(load_from_disk(dataset_path))
    template_scores = {}
    template_labels = {}
    template_names = {}
    print(len(uuids), [len(scores[name]) for name in scores.keys()])
    for index, uuid in enumerate(uuids):
        template = df_dataset[df_dataset["uuid"] == uuid]["template"].values
        if len(template) == 0:
            continue
        template = template[0]
        if chosen_relations != [] and template not in chosen_relations:
            continue
        if template not in template_scores.keys():
            template_scores[template] = {name: [] for name in scores.keys()}
            template_labels[template] = {name: [] for name in scores.keys()}
            template_names[template] = [name for name in scores.keys()]
        for name in scores.keys():
            template_scores[template][name].append(scores[name][index].item())
            template_labels[template][name].append(labels[index].item())
    return template_scores, template_labels, template_names


def lama_models_cleanup(df_s):
    df_s["mixinst"] = df_s["mixinst"].fillna(df_s["mixinst_l24"])
    df_s = df_s.drop(columns=["mixinst_l24"])
    df_s["mixnoinst"] = df_s["mixnoinst"].fillna(df_s["mixnoinst_l24"])
    df_s = df_s.drop(columns=["mixnoinst_l24"])
    df_s = df_s.drop(
        columns=["mistinst1_l1", "mistnoinst1_l1", "falinst_l1", "mistinst2_l1"]
    )
    return df_s


def remove_none_colnames(df_s, clean_index=False):
    for key in df_s.keys():
        if "_None" in key:
            if key.replace("_None", "") in df_s.keys():
                df_s[key.replace("_None", "")] = df_s[key.replace("_None", "")].fillna(
                    df_s[key]
                )
            else:
                df_s[key.replace("_None", "")] = df_s[key]
            df_s = df_s.drop(columns=[key])
    if clean_index:
        df_s = remove_none_colnames(df_s.transpose()).transpose()
    return df_s


def kl_per_template(pos_scores, neg_scores):
    # get the kl div for each bin between label 1 and 0
    dist_true = torch.histc(torch.tensor(pos_scores), bins=100, min=0, max=1)
    dist_false = torch.histc(torch.tensor(neg_scores), bins=100, min=0, max=1)
    kl_div = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(dist_true, dim=0),
        torch.nn.functional.log_softmax(dist_false, dim=0),
        log_target=True,
        reduction="batchmean",
    ).item()
    return kl_div


def wasserstein_per_template(pos_scores, neg_scores):
    # get the wasserstein distance for each bin between label 1 and 0
    dist_true = torch.histc(torch.tensor(pos_scores), bins=100, min=0, max=1)
    dist_false = torch.histc(torch.tensor(neg_scores), bins=100, min=0, max=1)
    wasserstein = stats.wasserstein_distance(
        dist_true.detach().cpu().numpy(),
        dist_false.detach().cpu().numpy(),
    )
    return wasserstein
