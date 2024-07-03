# Factual Confidence of LLMs: on Reliability and Robustness of Current Estimators

This repository contains the code used for experiments from: [Factual Confidence of LLMs: on Reliability and Robustness of Current Estimators](https://arxiv.org/abs/2406.13415).

![Descriptive diagram of a sentence being processed by multiple testing methods](Fact_Conf.png)

This repository regroups 5 types of Methods used to estimate factual confidence in LLMs, which can then be used to reproduce experiments and test them on question answering datasets: 
- Verbalised (prompt based)
- Trained probe (requires training)
- Surrogate token probability (prompt based)
- Average sequence probability
- Model consistency

We additionally set up a paraphrasing pipeline, using strong filtering to ensure semantic preservation. This allows to test models for a fact across different phrasings and translations.
## Getting Started

### Installation
The project uses `poetry` for dependency management and packaging. The latest version and instructions can be
found on [https://python-poetry.org](https://python-poetry.org/docs/).
official installer:
```shell
curl -sSL https://install.python-poetry.org | python3 -
```

```shell
poetry install
```

>Using poetry takes care of all dependencies, and therefore removes the need for requirements.txt. Should you still need that file for any reason, it can be generated using:
```shell
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

#### Accelerate
This project uses huggingface's accelerate for GPU management. 
Feel free to launch accelerate config to get the most out of it.

## Usage
### data generation pipeline:
Data has at least the following columns: ["text","uuid","is_factual"]. If the paraphrasing option is used, a ["paraphrase"] column will be used.

To prepare the True/False Lama TRex dataset use dataset_prep.py, which will create a test and train set in a data folder at root.
To experiment with the PopQA dataset :
 - Download csv file from the following [link](https://github.com/AlexTMallen/adaptive-retrieval/blob/main/data/popQA.tsv) (tested on 25/06/2024)
 - run slot_filling.py to get a specific model's ability to correctly answer each question, and generate the ["is_factual"] column
### to run experiments:
1. run training pipeline ("hidden") method
2. run main.py (all results are saved except for consistency)
3. run consistency pipeline
example scripts: scripts/main.sh, scripts/main_pop.sh, scripts/main_translated.sh, scripts/main_pik_lama.sh
for openai results, they are computed by running either evaluation/openai_surrogate.py, evaluation/openai_verbalized.py or data_gen/openai_sampler.py followed by the consistency pipeline.
Don't forget to set the variable in your environment before running. OPENAI_KEY=$mysecretkey 

### training pipeline - run, in order:
example script: scripts/extract_hidden.sh 
1. evaluation/extract_hidden_layers.py (runs a given model on a given dataset, and saves the hidden dimensions + labels for training)
2. train_scorer_2 (takes as input the hidden dimensions from previous script, runs gradient descent, saves the resulting model)

### consistency pipeline - run, in order:
1. slot_filling.py (checks, either for popqa or for lama, whether a model outputs the expected answer to a given prompt - serves as labels. If those were generated for previous experiments, skip)
1. (b) for the lama dataset, an alternative is to run comparative_knowledge.py which tests which of the true fact or the hardest false fact the model is most likely to output. This requires wikidata graphs.
2. data_gen/sampling.py (generates n completions. saves them as csv (raw) and tsv (processed by cleanup_sampling function))
3. evaluation/consistency_utils.py (takes as input the .tsv file, returns a .pt file matching uuids with consistency scores)

example scripts: scripts/sf.sh, scripts/sampling.sh

### paraphrasing pipeline:
- data_gen/paraphrases/gen_paraphrasing.py (saves a .csv version of the dataset with an additional "paraphrase" column)
- run main.py, with the paraphrase flag set to True

### to draw graphs from data see:
* graphing/draw_graphs.py (bar plots and method correlation plot - further directions commented @ start of doc)
* graphing/consistency_analysis.py (get auprc numbers from sampling pipeline, then needs to be manualy added to barplot)
* graphing/paraph_graph_utils.py (computes micro-average across paraphrases, macro-average, and normalized standard deviation)

## References

Please cite as [[1]](https://arxiv.org/abs/2406.13415).


[1] M. Mahaut, L. Aina, P. Czarnowska, M. Hardalov, T. Müller, L. Màrquez ["*Factual Confidence of LLMs:
on Reliability and Robustness of Current Estimators"*]() Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL). 2024.


```
@inproceedings{mahaut-etal-2024-factual,
    title = "Factual Confidence of LLMs: on Reliability and Robustness of Current Estimators",
    author="Mahaut, Mat{\'e}o and
                  Aina, Laura and 
                  Czarnowska, Paula and 
                  Hardalov, Momchil and 
                  M{\"u}ller, Thomas and 
                  M{\`a}rquez, Llu{\'\i}s",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)",
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2406.13415",
}
```

## License
- This project is licensed under the Apache-2.0 License.
