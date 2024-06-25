# Description: Run slot filling on chosen models
# This gives us the gold labels for whether the model can fill in the slot correctly, used for training P(IK) probes and generally evaluating P(IK)

# models=("mistralai/Mistral-7B-v0.1" "tiiuae/falcon-40b-instruct" "tiiuae/falcon-40b" "mistralai/Mixtral-8x7B-v0.1" "mistralai/Mixtral-8x7B-Instruct-v0.1" "mistralai/Mistral-7B-Instruct-v0.2" "mistralai/Mistral-7B-Instruct-v0.1" "tiiuae/falcon-7b" "tiiuae/falcon-7b-instruct")
models=("mistralai/Mixtral-8x7B-v0.1")
for model in "${models[@]}"
do
    batch_size=32
    # if the model is a 7B, increase the batch size
    if [[ $model == *"7B"* ]]; then
        batch_size=100
    fi
    # uncomment for multi-gpu
    # 8 gpus, 8 ranks
    # for i in 0 1 2 3 4 5 6 7; do
    # echo "Running slot filling on $model, rank $i"
    # python ./src/self_knowledge/slot_filling.py --model-name=$model --output-path=../results_kn_lama/$model/ --data-path=../data/reduced_trex_test --batch-size=$batch_size --rank=$i --no-accelerator --num-ranks=8 --log-path=../logs/$model/ &
    # done

    #comment for multi-gpu
    python ./src/self_knowledge/slot_filling.py --model-name=$model --output-path=../results_kn_popqa/$model/ --data-path=../data/clean_popQA.csv --batch-size=$batch_size --no-accelerator --log-path=../logs/$model/ &

    wait
done

# uncomment to chain experiments
# bash ./src/self_knowledge/extract_layers.sh
# bash ./src/self_knowledge/main_pop.sh