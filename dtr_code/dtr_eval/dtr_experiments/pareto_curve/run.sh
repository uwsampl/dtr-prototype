#!/bin/bash
config_dir=$1
data_dir=$2

source $BENCHMARK_DEPS/bash/common.sh

include_shared_python_deps
add_to_pythonpath $(pwd)
add_to_pythonpath $(realpath "$(pwd)/../../shared")

python3 "run.py" "--config-dir" "$config_dir" "--output-dir" "$data_dir"
if [ $? -ne 0 ]; then
    exit 1;
fi
