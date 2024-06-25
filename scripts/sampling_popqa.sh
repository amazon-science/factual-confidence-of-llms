# sample answers for PopQA questions
# useful for constructing the consistency metric. 
# can be distributed on multiple (here 8) GPUs using rank

# models=("tiiuae/falcon-7b" "mistralai/Mixtral-8x7B-Instruct-v0.1" "mistralai/Mixtral-8x7B-v0.1" "tiiuae/falcon-40b-instruct" "tiiuae/falcon-40b" "mistralai/Mistral-7B-Instruct-v0.2"  "mistralai/Mistral-7B-v0.1" "mistralai/Mistral-7B-Instruct-v0.1" "tiiuae/falcon-7b-instruct")
# models=("mistralai/Mistral-7B-Instruct-v0.2"  "mistralai/Mistral-7B-v0.1" "tiiuae/falcon-7b" "tiiuae/falcon-7b-instruct" )
models=("tiiuae/falcon-7b-instruct")
# 
#iterate through models
for model_name_or_path in "${models[@]}"; do
  echo $model_name_or_path
  for temp in 1; do
    for rank in 0; do
        python ./src/self_knowledge/data_gen/sampling.py --rank=$rank --temperature=$temp --num-ranks=1 --output-dir=../data/reduced_sampling_pop/$model_name_or_path --model-name-or-path=$model_name_or_path --batch-size=1 --dataset-name=popQA &
    done
    wait
  done
done