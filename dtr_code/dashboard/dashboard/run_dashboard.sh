#!/bin/bash
#
# Sets up environment to run dashboard infrastructure
#
# Arguments (must be in this order):
# dashboard home (mandatory)
# experiment dir (optional)
# subsystem dir (optional)
dashboard_home="$(realpath $1)"
if [ "$#" -ge 2 ]; then
    save_arg_1="$(realpath $2)"
fi
if [ "$#" -ge 3 ]; then
    save_arg_2="$(realpath $3)"
fi

# store path to this script
cd "$(dirname "$0")"
script_dir=$(pwd)
experiments_dir=$script_dir/../experiments
subsystem_dir=$script_dir/../subsystem
if [ "$#" -ge 2 ]; then
    experiments_dir=$save_arg_1
fi
if [ "$#" -ge 3 ]; then
    subsystem_dir=$save_arg_2
fi

# need /usr/local/bin in PATH
export PATH="/usr/local/bin${PATH:+:${PATH}}"

# ensure CUDA will be present in the path
export PATH="/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/NsightCompute-2019.1${PATH:+:${PATH}}"

# need CUDA LD libraries too
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

cd $script_dir/..

# export because benchmarks may need it
export BENCHMARK_DEPS=$(pwd)/shared
# allow using the shared Python libraries for the dashboard infra
source $BENCHMARK_DEPS/bash/common.sh
include_shared_python_deps

cd $script_dir
python3 dashboard.py --home-dir "$dashboard_home" --experiments-dir "$experiments_dir" --subsystem-dir "$subsystem_dir"
