# Description: Script to run self-confidence evaluation on LAMA dataset. 
# Uses P(IK) definition of self-confidence --> prompts are incomplete and scorer must evaluate if it could correctly complete the prompt.

# here list models to evaluate
# models=("tiiuae/falcon-40b-instruct" "tiiuae/falcon-40b" "mistralai/Mixtral-8x7B-Instruct-v0.1" "mistralai/Mixtral-8x7B-v0.1" "tiiuae/falcon-7b" "mistralai/Mistral-7B-Instruct-v0.2"  "mistralai/Mistral-7B-v0.1" "mistralai/Mistral-7B-Instruct-v0.1" "tiiuae/falcon-7b-instruct")
models=("mistralai/Mistral-7B-Instruct-v0.1") #  ("mistralai/Mistral-7B-v0.1" "mistralai/Mistral-7B-Instruct-v0.1" "tiiuae/falcon-7b-instruct" "tiiuae/falcon-7b")

#iterate through models
for model_name_or_path in "${models[@]}"; do
  echo $model_name_or_path
  # for the biggest models, single rank
  if [ $model_name_or_path == "mistralai/Mixtral-8x7B-Instruct-v0.1" ] || [ $model_name_or_path == "mistralai/Mixtral-8x7B-v0.1" ] || [ $model_name_or_path == "tiiuae/falcon-40b-instruct" ] || [ $model_name_or_path == "tiiuae/falcon-40b" ]; then
    python /home/ec2-user/self-knowledge/src/self_knowledge/main.py --model-name=$model_name_or_path --dataset-path=../results_kn_lama/$model_name/lama_sf_None_test.csv --hidden-scorer-path=../models/hidden_scorer/$model_name_or_path/hscorer_24.pt --save-path=../results_3/$model_name_or_path/ --batch-size=50 --log-path=../logs/$model_name_or_path/ --no-accelerator
  else
    accelerate launch /home/ec2-user/self-knowledge/src/self_knowledge/main.py --model-name=$model_name_or_path --dataset-path=../results_kn_lama/$model_name/lama_sf_None_test.csv --hidden-scorer-path=../models/hidden_scorer/$model_name_or_path/hscorer_24.pt --save-path=../results_3/$model_name_or_path/ --batch-size=50 --log-path=../logs/$model_name_or_path/ 
  fi
  wait
done
