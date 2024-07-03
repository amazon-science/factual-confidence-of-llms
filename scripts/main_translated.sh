# runs self-confidence evaluation on translated datasets

models=("mistralai/Mixtral-8x7B-Instruct-v0.1" "mistralai/Mixtral-8x7B-v0.1" "tiiuae/falcon-7b" "mistralai/Mistral-7B-Instruct-v0.2"  "mistralai/Mistral-7B-v0.1" "mistralai/Mistral-7B-Instruct-v0.1" "tiiuae/falcon-7b-instruct" "tiiuae/falcon-40b" "tiiuae/falcon-40b-instruct")
#iterate through models
for model_name_or_path in "${models[@]}"; do
  echo $model_name_or_path
  # for the biggest models, single rank
  if [ $model_name_or_path == "tiiuae/falcon-40b-instruct" ] || [ $model_name_or_path == "tiiuae/falcon-40b" ]; then
    python ./src/self_knowledge/main.py --model-name=$model_name_or_path --dataset-path=../data/french_lama.csv --hidden-scorer-path=../models/hidden_scorer/$model_name_or_path/hscorer_24.pt --save-path=../results_french/$model_name_or_path/ --batch-size=16 --log-path=../logs/$model_name_or_path/ --no-accelerator
    wait
    python ./src/self_knowledge/main.py --model-name=$model_name_or_path --dataset-path=../data/polish_lama.csv --hidden-scorer-path=../models/hidden_scorer/$model_name_or_path/hscorer_24.pt --save-path=../results_polish/$model_name_or_path/ --batch-size=16 --log-path=../logs/$model_name_or_path/ --no-accelerator
  else
    python ./src/self_knowledge/main.py --model-name=$model_name_or_path --dataset-path=../data/french_lama.csv --hidden-scorer-path=../models/hidden_scorer/$model_name_or_path/hscorer_24.pt --save-path=../results_french/$model_name_or_path/ --batch-size=100 --log-path=../logs/$model_name_or_path/ 
    wait
    python ./src/self_knowledge/main.py --model-name=$model_name_or_path --dataset-path=../data/polish_lama.csv --hidden-scorer-path=../models/hidden_scorer/$model_name_or_path/hscorer_24.pt --save-path=../results_polish/$model_name_or_path/ --batch-size=100 --log-path=../logs/$model_name_or_path/ 

  fi
  wait
done
