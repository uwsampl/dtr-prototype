#!/bin/bash

baseline_commands=(
    'python experiments/experiment_budget_sweep.py --model-name "VGG16" -b 256 --platform p32xlarge'
    'python experiments/experiment_budget_sweep.py --model-name "MobileNet" -b 512 --platform p32xlarge'
    'python experiments/experiment_budget_sweep.py --model-name "vgg_unet" -b 32 --platform p32xlarge'
)

simrd_commands=(
    'python experiments/experiment_budget_sweep_simrd.py --model-name "VGG16" -b 256 --platform p32xlarge'
    'python experiments/experiment_budget_sweep_simrd.py --model-name "MobileNet" -b 512 --platform p32xlarge'
    'python experiments/experiment_budget_sweep_simrd.py --model-name "vgg_unet" -b 32 --platform p32xlarge'
)

if [[ $1 == "baseline" ]]; then
    rm -rf baseline_stdout_err
    mkdir baseline_stdout_err
    rm baseline_results.txt
    index=1
    for i in "${baseline_commands[@]}"; do
        echo $i
        start=$(date +%s%N | cut -b1-13)
        if [[ $2 == "skip-ilp" ]]; then
            echo "skipping ilp..."
            eval "$i --skip-ilp" &>baseline_stdout_err/$index.txt
        else
            eval "$i" &>baseline_stdout_err/$index.txt
        fi
        end=$(date +%s%N | cut -b1-13)
        runtime=$((end - start))
        echo $i >>baseline_results.txt
        echo "$runtime ms" >>baseline_results.txt
        ((index = index + 1))
    done
elif [[ $1 == "simrd" ]]; then
    rm -rf simrd_stdout_err
    mkdir simrd_stdout_err
    rm simrd_results.txt
    index=1
    for i in "${simrd_commands[@]}"; do
        echo $i
        start=$(date +%s%N | cut -b1-13)
        eval "$i" &>simrd_stdout_err/$index.txt
        end=$(date +%s%N | cut -b1-13)
        runtime=$((end - start))
        echo $i >>simrd_results.txt
        echo "$runtime ms" >>simrd_results.txt
        ((index = index + 1))
    done
elif [[ $1 == "plot" ]]; then
    python experiments/plot_simrd_comparison.py
else
    echo 'unknown option $1'
    exit 1
fi
