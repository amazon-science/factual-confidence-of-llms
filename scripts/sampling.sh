models=("tiiuae/falcon-7b" "mistralai/Mixtral-8x7B-Instruct-v0.1" "mistralai/Mixtral-8x7B-v0.1" "tiiuae/falcon-40b-instruct" "tiiuae/falcon-40b" "mistralai/Mistral-7B-Instruct-v0.2"  "mistralai/Mistral-7B-v0.1" "mistralai/Mistral-7B-Instruct-v0.1" "tiiuae/falcon-7b-instruct")
# models=("tiiuae/falcon-40b-instruct" "tiiuae/falcon-40b" )
#iterate through models
for model_name_or_path in "${models[@]}"; do
  echo $model_name_or_path
  save_name=$(echo $model_name_or_path | sed 's/\//-/g')
  echo $save_name
  for temp in {0.5,1}; do
    for rank in {0..7}; do
        python ./src/self_knowledge/data_gen/sampling.py --rank=$rank --temperature=$temp --num-ranks=8 --output-dir="../data/reduced_sampling_pop/($save_name)_t_1" --model-name-or-path=$model_name_or_path --batch-size=50 --dataset-name=lama &
    done
    wait
done
