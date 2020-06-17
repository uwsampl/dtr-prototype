# Dynamic Tensor Rematerialization (DTR) Prototype

DTR Authors: Marisa Kirisame, Steven Lyubomirsky, Altan Haan, Jennifer Brennan, Mike He, Jared Roesch, Tianqi Chen, Zachary Tatlock

## Repository Contents

This repo contains the following:
* `data_files`: Data produced from runs in the prototype evaluation figures in the paper
* `dtr_configs`: Configuration files for running the prototype DTR implementation
* `dtr_code`: DTR prototype implementation and infrastructure for running the experimental evaluation
* `simrd`: Simulator implementation and logs used to generate figures in the simulated evaluation

This corresponds to the versions of the simulated and PyTorch implementation in the present preprint;
we may make our development repositories public once they become more stable.

Note: We ran the simulated and prototype evaluation using Python 3.7.4 on an Ubuntu 18.04 machine with an NVidia Titan-V GPU (12 GB memory), using CUDA 10.1 and CuDNN 7.5.4. We documented every dependency we were aware of, but it is possible that we were unaware of OS-level dependenices.

## Running the Simulator

The simulator is named "simrd": Simulated (tensor) rematerialization, dynamic

### Quickstart
Assuming you have Python 3 and `venv` (for Python virtual environments), and
have a Unix-like system with `bash`, simply run `bash setup.sh` from the `simrd`
root and then `bash run.sh`.

This will output the paper figures (and data) to the `data/` directory.

### Full Setup
#### Python Dependencies
First, make sure `pip` is up to date by running
`python3 -m pip install --upgrade pip`.
Otherwise, the following step may fail with an error message about the `wheel`
package.

To install the Python 3 dependencies for `simrd`, run 
`python3 -m pip install requirements.txt`
from the `simrd` root. We highly recommend using a new `venv` for this so that
package versions do not conflict (and make sure to update `pip` in the `venv`).

**NOTE:** we have only tested the simulator on Linux and macOS systems with the
`bash` shell, and cannot guarantee it will work on Windows.

#### Unzip PyTorch Logs
Unzip the `models.zip` file in the `simrd` root a directory named `models/` (also
in the `simrd` root).

### Run Simulated Evaluations
To generate data and plots for all the figures in the paper (full pareto curve
evaluation, ablation study, banishing, tensor accesses), run
`env PYTHONPATH=.:$PYTHONPATH python3 experiments/eval/pareto/main.py`
from the `simrd` root.

The resulting figures will be saved in `data/` (and raw data in
`data/eval/pareto`).

**NOTE**: on some systems, Python will complain about the module path, so we
need to export `PYTHONPATH` to include the `simrd` root.

## Running the PyTorch Prototype Implementation

The prototype implementation is needed both for generating logs for the simulator as well as for directly measuring the prototype's performance.

### Executive Summary

All one-time setup is collected in `setup.sh`. Once all the setup is complete, a configuration of the prototype can be run by calling `./dashboard/dashboard/run_dashboard.sh ./dtr_home ./dtr_eval/dtr_experiments` (this is a stripped-down version of the Relay dashboard in https://github.com/uwsampl/relay-bench).

You can change the configuration of the prototype by substituting `dtr_home/config/experiments/pareto_curve/config.json` with one of the configuration files in `dtr_configs`. See below for how to post a summary to Slack (most convenient). Any visualizations produced will be found in `dtr_home/results/experiments/graph` and data files (like those in `data_files`) will be in `dtr_home/results/experiments/data`. If logging is enabled in the configuration (`"save_logs"`), logs will be deposited in `~/dtr_logs` (configurable under `"log_dest"`).

### Commands and Reading Results

These experiments use a dashboard infrastructure provided in the `dashboard` directory. They rely on configurations that are given in `dtr_home` (namely in `dtr_home/config/experiments/pareto_curve/config.json`). Results will be posted in `dtr_home/results/experiments/data` (processed data in a JSON format), `dtr_home/results/experiments/summary` (a more human-readable text summary), and `dtr_home/results/experiments/graph` (graphs).

It is recommended, though not necessary, that you use the dashboard's Slack integration to post results to Slack (requires a webhook URL, which can be created by following the steps here: https://api.slack.com/messaging/webhooks). This functionality can be configured in `dtr_home/config/subsystem/exp_results/config.json` (filling in the webhook URL field).

We provide three configs in `dtr_configs` that can be used for running the same experiments as in the paper:

* `config_get_log.json` for obtaining logs for the simulator (we provide the logs used in the paper's figures in `dtr_logs`)
* `config_run_graphs.json` for generating the prototype performance graphs in the paper (we provide the data files generated in `dtr_data`)
* `config_large_inputs.json` for measuring the prototype's performance on large inputs for ResNet-1202 and TreeLSTM (data files also provided in `dtr_data`)
* `config_encoders.json` to run the encoder models on which DTR and the simulator could not save memory (`treelstm_old` will time out, but you can increase the `timeout` setting to have it complete, though it may take a long time).

Simply substitute one of these files for the config in `dtr_home/config/experiments/pareto_curve/config.json` (please ensure it will still be named `config.json`) in order to run with their settings.

Once configured, the dashboard can be invoked as follows: `./dashboard/dashboard/run_dashboard.sh ./dtr_home ./dtr_eval/dtr_experiments`

### Our Modified PyTorch Code

We build our DTR prototype by modifying [PyTorch](https://github.com/pytorch/pytorch), 
starting from commit `1546d2afeb98fcd7bc5b58261d6e31ad7794e833`. 
To avoid compilications with git submodules, we simply include the entire source code, 
including submodules, in the `dtr_pytorch` folder (forgive us).

Most of our modifications to standard PyTorch are in `aten/src/ATen/native/native_functions.yaml`.
However, the bulk of our changes are in new files we added ourselves:
* `aten/src/ATen/CheckpointTensorImpl.h` and ``aten/src/ATen/CheckpointTensorImpl.cc` (checkpointing logic)
* `aten/src/ATen/Logger.h` (logging)
* `aten/src/ATen/native/Checkpoint.cc` (operator overloads)

The model setup in `dtr_code/dtr_eval/shared/model_util.py` and execution in `dtr_code/dtr_eval/shared/run_torch_trial.py`
demonstrate how to invoke DTR's checkpointing.
In general, a tensor should be converted to a DTR-managed one using the `checkpoint()` method,
which will ensure that any computation involving that tensor will go through DTR's overload layer.
Once any computation involving checkpointed tensors is finished, those tensors should be converted
into an ordinary PyTorch tensor using `decheckpoint()`.
A memory budget for DTR can be specified using `torch.set_memory_budget(budget)`.

### Global Dependencies

The version of PyTorch used for developing DTR depends on having CUDA 10.1 and a version of CuDNN that is compatible with it. Building PyTorch also requires a C++ compiler that can support at least C++14. (The README for PyTorch lists various other dependencies, but we found that we were able to successfully build PyTorch using `setup.py develop` without them, oddly.)

The `run_dashboard.sh` script in `dashboard/dashboard` ensures that references to CUDA and CuDNN are found on the user's `PATH`, which is necessary for PyTorch (including the standard distribution) to work.

### Python Dependencies

Requires the dependencies in `requirements.txt`. Please install these files in whatever Python environments you wish to use by running `pip3 install -r requirements.txt`. This should also be done for the `venv` for installing the DTR-modified PyTorch (`dtr_venv/bin/pip3 install -r requirements.txt`).

The Unrolled GAN implementation in `dtr_eval/shared/torch_models/unroll_gan` also requires the library [Higher](https://github.com/facebookresearch/higher), which could not be included in `requirements.txt`, so if you want to execute that model for logging, you must install Higher as follows:

```
git clone git@github.com:facebookresearch/higher.git
cd higher
pip3 install .
~/dtr_venv/bin/pip3 install .
```

### Data setup

The original [TreeLSTM version](https://github.com/dasguptar/treelstm.pytorch) (`treelstm_old`) we used requires pulling in external data to run its trials, if you want to use it.

To do this, run `fetch_and_preprocess.sh` in `dtr_eval/shared/torch_models/treelstm` in order to pull in the data (about 2GB). The script depends on relative directories, so you must be in that directory before calling it. It also requires having the Java JDK installed (we used Javac 11, but it is likely that an earlier version would work).

### DTR Setup

In order to compare DTR-modified PyTorch with the baseline directly, these experiments assume that you have installed the DTR-modified PyTorch to a Python `venv`, whose location you can specify in the relevant experiment's `dtr_torch_cmd` config field (expects a path to a `python` binary in a `venv` where the DTR Pytorch is installed as `torch`).

You can create and initialize a suitable virtual environment by doing the following:

```
python3 -m venv ~/dtr_venv
~/dtr_venv/bin/pip3 install -r dtr_code/requirements.txt
cd dtr_code/dtr_pytorch
# warning: the first build may take a long time, perhaps over an hour
~/dtr_venv/bin/python3 setup.py develop
```

Once these steps are finished, `~/dtr_venv/bin/python3` will point to a Python executable with all appropriate depdencies installed and with DTR's PyTorch present.

### Supported Models

See `dtr_eval/shared/validate_config.py` for a list of all supported models.

In particular, the configs include `resnet32`, `resnet1202`, `densenet100`, `unet`, and `unroll_gan` (supported by the simulator but not the prototype unless the budget is set to infinite for logging), adapted from various public implementations:
* https://github.com/akamaster/pytorch_resnet_cifar10
* https://github.com/bamos/densenet.pytorch
* https://github.com/milesial/Pytorch-UNet
* https://github.com/mk-minchul/unroll_gan

We also include several dynamic models taken from public implementations, namely LSTM and GRU encoders from [PyTorch's official word language model examples](https://github.com/pytorch/examples/tree/master/word_language_model) and [TreeLSTM](https://github.com/dasguptar/treelstm.pytorch). We include these as `lstm_encoder`, `gru_encoder`, and `treelstm_old`. We also provide LSTM RNN and TreeLSTM implementations we wrote ourselves to avoid the memory bottlenecks we encountered on the publicly available versions, called simply `lstm` and `treelstm`.

### Saving DTR Logs

The config provides the options `save_logs` (false by default) and `log_dest` (only used if `save_logs` is true) to copy over the DTR logs produced in an experiment, if any are produced. If `save_logs` is set to true, the experiment will copy over the last log produced when running a command to the specified destination directory. This allows for producing logs (which contain timings) to benefit from the warm-up runs in the dashboard.

The final run in the log will be marked with a "`START`" annotation.

### Experiment Commands

The experiment commands are defined in `dtr_settings`.
To add commands for the experiment of a `model`, add a `JSONObject` that has the following structure:
```json
{
  "<model_name>" : [<command>]
}
```
Where `<command>` has the following structure:
```json
{
  "type" : ["baseline" | "dtr"],
  "kind" : ["fixed" | "ratio"],
  "memory_budget" : [number+],
  "ratio" : [real+]
}
```
For a `ratio` command, the infrastructure will first run a baseline trial and record the maximum memory allocated and then calculate the `memory_budget` based on the `ratio` given in the command config.

For a `fixed` command, the infrastructure will run the model using `dtr` with the `memory_budget` provided in the command config. 

If there are fields other than `type` and `kind` defined as a list, the command will be unfolded using the Cartesian Product of those list fields and run each setting. This can blow up quickly so try to avoid doing this too often.

Note that `kind` should be defined iff the `type` of a command is `dtr`. For a `fixed` command, at least one `memory_budget` should be provided. For a `ratio` command, at least one `ratio` number should be provided. 
