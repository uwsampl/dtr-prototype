#!/bin/bash
config_dir=$1
data_dir=$2

source $BENCHMARK_DEPS/bash/common.sh

include_shared_python_deps
add_to_pythonpath $(pwd)
add_to_pythonpath $(realpath "$(pwd)/../../shared")

# set up treelstm data if it isn't already
# (hack around the lack of global dependency management)
# shared_dir=$(realpath "$(pwd)/../../shared")
# $shared_dir/setup_treelstm.sh "$config_dir"

python3 "run.py" "--config-dir" "$config_dir" "--output-dir" "$data_dir"
if [ $? -ne 0 ]; then
    exit 1;
fi
