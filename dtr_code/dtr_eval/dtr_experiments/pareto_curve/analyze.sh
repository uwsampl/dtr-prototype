#!/bin/bash
config_dir=$1
data_dir=$2
dest_dir=$3

source $BENCHMARK_DEPS/bash/common.sh

include_shared_python_deps
add_to_pythonpath $(pwd)
add_to_pythonpath $(realpath "$(pwd)/../../shared")

python3 analyze.py --config-dir $config_dir --data-dir $data_dir --output-dir $dest_dir
