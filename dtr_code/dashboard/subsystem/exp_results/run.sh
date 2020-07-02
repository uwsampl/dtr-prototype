#!/bin/bash
config_dir=$1
home_dir=$2
output_dir=$3

source $BENCHMARK_DEPS/bash/common.sh

include_shared_python_deps
add_to_pythonpath $(pwd)

python3 run.py --config-dir "$config_dir" --home-dir "$home_dir" --output-dir "$output_dir"
