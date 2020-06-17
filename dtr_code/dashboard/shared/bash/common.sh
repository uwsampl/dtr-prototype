#!/bin/bash
# Bash utilities for dashboard experiments

# First arg should be the name of the calling script (i.e., $0 from the script)
# Sets dir_val to the directory of the script
function script_dir {
    dir=$(pwd)
    cd "$(dirname "$1")"
    ret=$(pwd)
    cd $dir
    dir_val=$ret
}
export -f script_dir

# Adds the argument to the PYTHONPATH var
function add_to_pythonpath {
    export PYTHONPATH=$1:${PYTHONPATH}
}
export -f add_to_pythonpath

# Adds the shared python deps to the PYTHONPATH var
function include_shared_python_deps {
    add_to_pythonpath "$BENCHMARK_DEPS/python"
}
export -f include_shared_python_deps

# Produces a status JSON file in the target directory
# First arg: Is the status successful?
# Second arg: Status message
# Third arg: Target directory
function emit_status_file {
    flag=$1
    msg=$2
    dest=$3

    flag_str="true"
    if [ $flag = false ] || [ $flag = 0 ]; then
        flag_str="false"
    fi

    # Have to filter out newlines from the message and escape quotes
    # or it won't be valid JSON.
    # Also uses iconv to trim non-ASCII characters to prevent JSON parsing errors later.
    altered_msg=$(echo "$msg" | sed 's/$/\\n/' | tr -d '\n' | sed 's/"/\\"/g' | iconv -c -t ascii)
    content="{\"success\": $flag_str, \"message\": \"$altered_msg\"}"
    echo "$content" > "$dest/status.json"
}
export -f emit_status_file

# Runs a Python script with the passed arguments and exits if its exit
# code is nonzero
function check_python_exit_code {
    python3 "$@"
    if [ $? -ne 0 ]; then
        exit 1;
    fi
}
export -f check_python_exit_code

# Takes at least three arguments (a Python script meant to run a trial,
# a directory meant to be the config dir, and a directory meant to be
# the output dir) and runs the following:
# python3 script.py --config-dir config_dir --output-dir output_dir [all remaining args]
# Exits if the exit code is nonzero
function python_run_trial {
    check_python_exit_code "$1" "--config-dir" "$2" "--output-dir" "$3" "${@:4}"
}
export -f python_run_trial

# Runs a given script with arguments and captures stderr in a status.json
# file if the script exits with a nonzero code (then this function calls exit)
# Input format: wrap_script_status "dest_for_status" [script with args]
#
# Warning: Do not use with a script that handles its own failures or sets a
# status because this may overwrite the status that the script sets
#
# Note that wrap_script_status will not work with a non-script command and
# wrap_command_status will not work with a script
function wrap_script_status {
    dest="$1"
    out=$(mktemp)
    bash "${@:2}" &>"$out"
    success=$?
    cp "$out" "$dest/out.log"
    msg=$(tail -n 50 "$out")
    rm "$out"
    if [ $success -ne 0 ]; then
        emit_status_file false "$msg" "$dest"
        exit 1;
    fi
}
export -f wrap_script_status

# Runs a given command with arguments and captures stderr in a status.json
# file if the script exits with a nonzero code (then this function calls exit).
# Note that if an argument to the command needs to be quoted in bash for
# the command to work, the argument to *this function* should contain
# escaped quotes (e.g., python3 -c "print('lol')" should be passed to this
# function as wrap_command_status $dir python3 -c "\"print('lol')\"")
#
# Input format: wrap_command_status "dest_for_status" [command with args]
#
# Warning: Do not use with a command that handles its own failures or sets a
# status because this may overwrite the status that the command sets
#
# Note that wrap_script_status will not work with a non-script command and
# wrap_command_status will not work with a script
function wrap_command_status {
    dest="$1"
    out=$(mktemp)
    bash -c "${*:2}" &>"$out"
    success=$?
    cp "$out" "$dest/out.log"
    msg=$(tail -n 50 "$out")
    rm "$out"
    if [ $success -ne 0 ]; then
        emit_status_file false "$msg" "$dest"
        exit 1;
    fi
}
export -f wrap_command_status
