#!/bin/bash
# Takes an experiment's config dir as an argument (dumb hack).
# Checks if the TreeLSTM data dependencies are properly in place.
# If they are, does nothing.
# If not, it moves them into place from the setup directory
# if they are there or otherwise runs the download script.
source $BENCHMARK_DEPS/bash/common.sh
include_shared_python_deps

config_dir=$1

shared_dir=$(dirname "$0")
treelstm_dir="$shared_dir/torch_models/treelstm"

# working around the lack of global dependency management
setup_dir=$(python3 "$shared_dir/get_global_config.py" --home-dir "$config_dir/../../..")/treelstm_files
mkdir -p "$setup_dir"

# Adapted from relay-bench/experiments/treelstm/run.sh
# Takes a base directory as an argument and checks for
# all the datafiles treelstm should have
function check_data_exists {
    declare -a dirs=("data"
                     "lib"
                     "data/sick/dev"
                     "data/sick/test"
                     "data/sick/train")

    declare -a files=("data/glove/glove.840B.300d.txt"
                      "data/sick/SICK_test_annotated.txt"
                      "data/sick/SICK_train.txt"
                      "data/sick/SICK_trial.txt"
                      "data/sick/vocab-cased.txt"
                      "data/sick/vocab.txt"
                      "lib/DependencyParse.class"
                      "lib/CollapseUnaryTransformer.class"
                      "lib/ConstituencyParse.class")

    declare -a data_files=("a.cparents"
                           "a.parents"
                           "a.rels"
                           "a.toks"
                           "a.txt"
                           "b.cparents"
                           "b.parents"
                           "b.rels"
                           "b.toks"
                           "b.txt"
                           "id.txt"
                           "sim.txt")

    for dir in "${dirs[@]}"
    do
        if ! [ -d "$1/$dir" ]; then
            data_exists=false
            return
        fi
    done

    for file in "${files[@]}"
    do
        if ! [ -f "$1/$file" ]; then
            data_exists=false
            return
        fi
    done

    for data_file in "${data_files[@]}"
    do
        if ! [ -f "$1/data/sick/dev/$data_file" ] || ! [ -f "$1/data/sick/test/$data_file" ] || ! [ -f "$1/data/sick/train/$data_file" ]; then
            data_exists=false
            return
        fi
    done

    data_exists=true
    return
}

# already configured -> nothing to do
check_data_exists "$treelstm_dir"
if $data_exists; then
    exit 0
fi

# if setup dir doesn't have the data files, run the download script and copy them for later (the download script must be done in-place)
check_data_exists "$setup_dir"
if ! $data_exists; then
    # run the script, move the datafiles
    cd "$treelstm_dir"
    ./fetch_and_preprocess.sh

    # Remove the subdirectories because cp -r
    # behaves differently when pointed to existing vs non-existing destinations:
    # 1. non-existing destination: creates the destination directory
    #    and copies contents directly into destination
    # 2. existing destination: creates a new subdirectory
    #    inside the destination directory and copies to subdirectory
    # (we want the first behavior)
    rm -rf "$setup_dir/data" "$setup_dir/lib"
    /bin/cp -rf "data" "$setup_dir/data"
    /bin/cp -rf "lib" "$setup_dir/lib"
    exit 0
fi

# if the data files are already there, just copy them
# (see above note on cp -r)
rm -rf "$treelstm_dir/lib" "$treelstm_dir/data"
/bin/cp -rf "$setup_dir/lib" "$treelstm_dir/lib"
/bin/cp -rf "$setup_dir/data" "$treelstm_dir/data"
