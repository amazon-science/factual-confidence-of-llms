
# runs in two steps - 1 extracts hidden layers from the model corresponding to the last token of a question / slot filling task. 
# 2 trains a scorer on the hidden layers to determine factuality

# here list models to extract hidden layers from
# model_names=("mistralai/Mistral-7B-Instruct-v0.1" "tiiuae/falcon-7b-instruct" "tiiuae/falcon-7b" "mistralai/Mistral-7B-Instruct-v0.2" "mistralai/Mixtral-8x7B-Instruct-v0.1" "mistralai/Mixtral-8x7B-v0.1" "tiiuae/falcon-40b-instruct" "tiiuae/falcon-40b")
model_names=("mistralai/Mistral-7B-Instruct-v0.1")
for model_name in "${model_names[@]}"; do
    # here chose which layers to extract
    for i in 24 -1; do
        echo $model_name
        echo $i
        python -m self_knowledge.evaluation.extract_hidden_layers --model-name=$model_name --hidden-layer-index=$i --save-file=../models/$model_name/lama3_hlayer_$i.pt --dataset-path=../data/reduced_trex_train
        python -m self_knowledge.evaluation.train_scorer_2 --hidden-path=../models/$model_name/lama3_hlayer_$i.pt --out-name=../models/lama_hidden_scorer/$model_name/lama3_hscorer_$i
    done
done