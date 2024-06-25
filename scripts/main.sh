# run P(IK) version of self-confidence evaluation on popQA dataset

# models=("mistralai/Mixtral-8x7B-Instruct-v0.1" "tiiuae/falcon-40b" "tiiuae/falcon-40b-instruct")
models=("mistralai/Mixtral-8x7B-v0.1")
#iterate through models
for model_name_or_path in "${models[@]}"; do
  echo $model_name_or_path
  # for the biggest models, single rank
  python ./src/self_knowledge/main.py --model-name=$model_name_or_path --dataset-path=../data/clean_popQA.csv --hidden-scorer-path=../models/pikpopqa_hidden_scorer/$model_name_or_path/lama_hscorer_24 --save-path=../results_pik2/$model_name_or_path/ --batch-size=32 --log-path=../logs/pt/$model_name_or_path/
  wait
done
