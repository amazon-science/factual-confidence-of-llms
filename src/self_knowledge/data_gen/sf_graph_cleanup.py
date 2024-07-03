import pandas as pd
import torch
from datasets import load_from_disk


def check_relation(graph, property_id, obj_id, subj_id):
    """
    Check if a relation exists in the graph.
    """
    print(obj_id)
    selected_obj_graph = pd.DataFrame(
        graph[graph["id"] == obj_id]["relations"].values[0]
    )
    print(selected_obj_graph)
    expected_subj = selected_obj_graph[
        selected_obj_graph["property_id"] == property_id
    ]["ids"].values[0]
    print(expected_subj)
    print(subj_id)
    return subj_id in expected_subj


def extract_ids(text, df_dataset):
    # check amongst templates which ones match the text
    for template in df_dataset["template"].unique():
        text_rel = template.split("[X]")[-1].split("[Y]")[0]
        if text_rel in text:
            p_id = df_dataset[df_dataset["template"] == template][
                "predicate_id"
            ].values[0]
            # get X and Y
            obj = text.split(text_rel)[0]
            subj = text.split(text_rel)[-1]
            # get ids
            obj_id = df_dataset[df_dataset["obj_label"] == obj]["obj_uri"].values
            if len(obj_id) != 0:
                obj_id = obj_id[0]
            subj_id = df_dataset[df_dataset["sub_label"] == subj]["sub_uri"].values
            if len(subj_id) != 0:
                subj_id = subj_id[0]
            return obj_id, subj_id, p_id


def verify_existing_sf(sf_path, jsonl_path, df_dataset, write_path):
    # load jsonl
    graph = pd.read_json(jsonl_path, lines=True)
    # load sf
    sf = pd.read_csv(sf_path)
    # check if sf is in graph
    write_file = open(write_path, "w")
    for i, row in sf.iterrows():
        obj_id, subj_id, property_id = extract_ids(row["text"], df_dataset)
        if obj_id != [] and subj_id is not None:
            row["is_factual"] = check_relation(graph, property_id, obj_id, subj_id)
            # write as csv line to file using pandas
            row.to_csv(write_file, header=i == 0, index=False)


def pop_preparation(df_dataset, pop_dataset):
    pop_dataset["o_uri"] = pop_dataset["o_uri"].apply(lambda x: x.split("/")[-1])
    pop_dataset["s_uri"] = pop_dataset["s_uri"].apply(lambda x: x.split("/")[-1])
    # check percentage of df in pop
    in_pop = 0
    both = 0
    for _, row in df_dataset.iterrows():
        if (
            row["obj_uri"] in pop_dataset["o_uri"].values
            or row["sub_uri"] in pop_dataset["s_uri"].values
        ):
            in_pop += 1
        if (
            row["obj_uri"] in pop_dataset["o_uri"].values
            and row["sub_uri"] in pop_dataset["s_uri"].values
        ):
            both += 1
    print("% of o or s in pop", in_pop / len(df_dataset))
    print("% of o and s in pop", both / len(df_dataset))
    # check percentage of pop in df
    in_df = 0
    both = 0
    for _, row in pop_dataset.iterrows():
        if (
            row["o_uri"] in df_dataset["obj_uri"].values
            or row["s_uri"] in df_dataset["sub_uri"].values
        ):
            in_df += 1
        if (
            row["o_uri"] in df_dataset["obj_uri"].values
            and row["s_uri"] in df_dataset["sub_uri"].values
        ):
            both += 1
    print("% of o or s in df", in_df / len(pop_dataset))
    print("% of o and s in df", both / len(pop_dataset))


def o_pop_matching(res_uuids, df_dataset, pop_dataset):
    pop_dataset["o_uri"] = pop_dataset["o_uri"].apply(lambda x: x.split("/")[-1])
    o_pops = []
    for uuid in res_uuids:
        obj = df_dataset[df_dataset["uuid"] == uuid]["obj_uri"].unique()
        if len(obj) != 1:
            print(f"warning, multiple obj for {uuid}: {obj}")
        if len(obj) != 0 and obj in pop_dataset["o_uri"].values:
            o_pop = pop_dataset[pop_dataset["o_uri"] == obj[0]]["o_pop"].values[0]
        else:
            o_pop = -1
        o_pops.append(o_pop)
    return o_pops


def s_pop_matching(res_uuids, df_dataset, pop_dataset):
    pop_dataset["s_uri"] = pop_dataset["s_uri"].apply(lambda x: x.split("/")[-1])
    s_pops = []
    for uuid in res_uuids:
        sub = df_dataset[df_dataset["uuid"] == uuid]["sub_uri"].unique()
        if len(sub) != 1:
            print(f"warning, multiple sub for {uuid}: {sub}")
        if len(sub) != 0 and sub in pop_dataset["s_uri"].values:
            s_pop = pop_dataset[pop_dataset["s_uri"] == sub[0]]["s_pop"].values[0]
        else:
            s_pop = -1
        s_pops.append(s_pop)
    return s_pops


if __name__ == "__main__":
    # example sentence
    df_dataset = pd.DataFrame(load_from_disk("data/balanced_trex_train"))

    # o pop
    pop_dataset = pd.read_csv("data/popQA.tsv", sep="\t")
    res_uuids = torch.load("results/btuuid2/0_uuids.pt")
    o_pops = o_pop_matching(res_uuids, df_dataset, pop_dataset)
    torch.save(o_pops, "results/btuuid2/0_o_pops.pt")
    # some stats - check num of -1
    print(f"missing values: {len([x for x in o_pops if x == -1])}/{len(o_pops)}")
    # range of values
    print(f"min: {min(o_pops)}, max: {max(o_pops)}")

    # s pop
    s_pops = s_pop_matching(res_uuids, df_dataset, pop_dataset)
    torch.save(s_pops, "results/btuuid2/0_s_pops.pt")
    # some stats - check num of -1
    print(f"missing values: {len([x for x in s_pops if x == -1])}/{len(s_pops)}")
    # range of values
    print(f"min: {min(s_pops)}, max: {max(s_pops)}")
