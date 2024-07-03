# Description: Script to run self-knowledge experiments on the popQA dataset

# models=("tiiuae/falcon-40b-instruct" "tiiuae/falcon-40b" "mistralai/Mixtral-8x7B-Instruct-v0.1" "mistralai/Mixtral-8x7B-v0.1" "tiiuae/falcon-7b" "mistralai/Mistral-7B-Instruct-v0.2"  "mistralai/Mistral-7B-v0.1" "mistralai/Mistral-7B-Instruct-v0.1" "tiiuae/falcon-7b-instruct")
models=("mistralai/Mixtral-8x7B-Instruct-v0.1" "mistralai/Mixtral-8x7B-v0.1" "tiiuae/falcon-40b-instruct" "tiiuae/falcon-40b")
#iterate through models
for model_name_or_path in "${models[@]}"; do
  echo $model_name_or_path
  # for the biggest models, single rank
  python ./src/self_knowledge/main.py --model-name=$model_name_or_path --dataset-path=../results_kn_popqa/$model_name_or_path/popqa_sf_None_paraph.csv --hidden-scorer-path=../models/pikpopqa_hidden_scorer/$model_name_or_path/lama_hscorer_-1 --save-path=../results_pik/$model_name_or_path/ --batch-size=50 --log-path=../logs2/$model_name_or_path/
  wait
done
