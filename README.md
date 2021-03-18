# Dynamic Tensor Rematerialization (DTR) Prototype

DTR Authors: Marisa Kirisame, Steven Lyubomirsky, Altan Haan, Jennifer Brennan, Mike He, Jared Roesch, Tianqi Chen, Zachary Tatlock

## Archive Contents

This archive contains the following:
* `data_files`: Data produced from runs in the prototype evaluation figures
* `dtr_configs`: Configuration files for running the prototype DTR implementation
* `dtr_code`: DTR prototype implementation and infrastructure for running the experimental evaluation
* `simrd`: Simulator implementation and logs used to generate figures in the simulated evaluation
* `checkmate_comp`: Modified version of Jain et al's MLSys 2020 reproducibility artifact that includes comparisons against `simrd` as a solver

Note: We ran the simulated and prototype evaluation using Python 3.7.4 on an Ubuntu 18.04 machine with an NVidia Titan-V GPU (12 GB memory), using CUDA 10.1 and CuDNN 7.5.4. We documented every dependency we were aware of, but it is possible that we were unaware of OS-level dependenices.

## Running the DTR Simulator

The simulator is named "simrd": Simulated (tensor) rematerialization, dynamic

### Setup
#### Step 1. Install Anaconda and set up environment
Install the Anaconda Python environment manager from
https://www.anaconda.com/distribution/.
Next, create and activate the `dtr-iclr21` environment:
```
conda create -n dtr-iclr21 python=3.7
conda activate dtr-iclr21
```

Now, export the `PYTHONPATH` to include this directory, so that the experiments
can locate the necessary files:
```
export PYTHONPATH=.:$PYTHONPATH
```

#### Step 2. Install dependencies and unzip logs
Install the dependencies for the simulator by running (in the new environment):
```
python -m pip install -r requirements.txt
```

The simulator comes with logs so that they do not need to be gathered, although
we provide instructions for gathering logs in the prototype folder. These logs
have been zipped to `logs.zip`. Extract them to the `logs` folder:
```
unzip logs.zip
```


### Run Simulations and Plotting
Finally, run the simulated evaluation by running
```
python simrd_experiments/eval/pareto/main.py
```

The resulting data and figures can be found under the `data` directory.
Note that this can take a few hours due to the ablation study.

## DTR vs Checkmate Baselines

### Installation
#### Step 0: Set up DTR simulator (simrd)
First, follow the setup instructions for the DTR simulator (simrd), which should
be bundled with this in the parent directory.

#### Step 1: Install Anaconda
Make sure that you have created the DTR Anaconda environment and have it
activated, as per the simulator instructions. Activate the environment by:
```
conda activate dtr-iclr21
```

#### Step 2: Install the Checkmate `remat` package and dependencies
From this directory,
```
$ conda install -c conda-forge python-graphviz
$ pip install -e .
```
Next, install tensorflow with
```
$ pip install tensorflow==2.0.1
```

#### Step 3: Install Gurobi
Checkmate uses the Gurobi optimziation library to solve an integer linear program that chooses a recomputation schedule for a given neural network architecture. This requires a license to Gurobi, which is free for academic use. The `grbgetkey` command used below must be run on a computer connected to a university network directly or via a VPN.

1. Please follow these instructions to install Gurobi on your system: https://www.gurobi.com/documentation/quickstart.html. For example, on Linux, follow `https://www.gurobi.com/documentation/9.0/quickstart_linux/software_installation_guid.html`.
2. Make an academic account with Gurobi at: https://pages.gurobi.com/registration
3. Request an acadmic license at: https://www.gurobi.com/downloads/end-user-license-agreement-academic/
4. Install the license by running the `grbgetkey` command at the end of the page. Ensure you are on an academic network like Airbears2 for UC Berkeley. If you save the license to a non-default location (outside your home directory), you will need to export the `GRB_LICENSE_FILE` variable with the path to the licence.
5. Set up the gurobipy Anaconda channel by running `conda config --add channels http://conda.anaconda.org/gurobi`
6. Install gurobipy by running: `conda install gurobi`

### Reproducing Figure 2: Computational overhead versus memory budget
We have provided a reproduceability script adapted from Checkmate's,
as `reproduce_simrd.sh`.

First, run the Checkmate baselines:
```
bash reproduce_simrd.sh baselines
```
Note that the Checkmate baselines include Checkmate's ILP solver, which took up
to 24 hours on our machine to complete in total for the models.

Then, run the DTR simulator (simrd):
```
bash reproduce_simrd.sh simrd
```

Lastly, plot the results:
```
bash reproduce_simrd.sh plot
```

The results will be saved under `data/budget_sweep`.

## Running the Prototype Implementation

### Executive Summary

All one-time setup is collected in `setup.sh`. Once all the setup is complete, a configuration of the prototype can be run by calling `./dashboard/dashboard/run_dashboard.sh ./dtr_home ./dtr_eval/dtr_experiments`

*Important*: Please ensure the configuration variable `sync_gpu` is set to `true` to put PyTorch into blocking mode before running timing trials. This is required to ensure the correctness of DTR's profiling timings.

You can change the configuration of the prototype by substituting `dtr_home/config/experiments/pareto_curve/config.json` with one of the configuration files in `dtr_configs`. See below for how to post a summary to Slack (most convenient). Any visualizations produced will be found in `dtr_home/results/experiments/graph` and data files (like those in `data_files`) will be in `dtr_home/results/experiments/data`. If logging is enabled in the configuration (`"save_logs"`), logs will be deposited in `~/dtr_logs` (configurable under `"log_dest"`).

To reproduce the graph in Figure 5 without having to rerun the eval, you can run `./dtr_code/graphing_util/visualize_pareto_curve.py dtr_configs/full-run-config.json data_files/data-full.json`.  

### Commands and Reading Results

These experiments use a dashboard infrastructure provided in the `dashboard` directory. They rely on configurations that are given in `dtr_home` (namely in `dtr_home/config/experiments/pareto_curve/config.json`). Results will be posted in `dtr_home/results/experiments/data` (processed data in a JSON format), `dtr_home/results/experiments/summary` (a more human-readable text summary), and `dtr_home/results/experiments/graph` (graphs).

It is recommended, though not necessary, that you use the dashboard's Slack integration to post results to Slack (requires a webhook URL, which can be created by following the steps here: https://api.slack.com/messaging/webhooks). This functionality can be configured in `dtr_home/config/subsystem/exp_results/config.json` (filling in the webhook URL field).

We provide two configs in `dtr_configs` that can be used for running the same experiments as in the paper:

* `full-run-config.json` for generating the prototype data used in the profiling comparison in Figure 4
* `table-data-run-config.json` for generating the prototype performance data reported in Table 1

Simply substitute one of these files for the config in `dtr_home/config/experiments/pareto_curve/config.json` (please ensure it will still be named `config.json`) in order to run with their settings.

Once configured, the dashboard can be invoked as follows: 
```
./dashboard/dashboard/run_dashboard.sh ./dtr_home ./dtr_eval/dtr_experiments
```

### Creating the PyTorch Code

Due to PyTorch's use of submodules and our own additional dependencies, we do need to provide a properly configured git repository with a history. We do this by providing a git patch that, if applied to the correct commit of PyTorch, will restore our code. The file is provded as `dtr-implementation.patch`.

The following steps will restore the PyTorch code:
```
git clone --recursive https://github.com/pytorch/pytorch dtr_pytorch
cd dtr_pytorch
# the commit we started from
git checkout d15b9d980c0cd504ce6e82db4e88f66cee7e0289
git submodule sync
git submodule update --init --recursive

# patch modifies submodules too so sync again after applying
git am --signoff < ../dtr-implementation.patch
git submodule sync
git submodule update --init --recursive
```

### Global Dependencies

The version of PyTorch used for developing DTR depends on having CUDA 10.1 and a version of CuDNN that is compatible with it. Building PyTorch also requires a C++ compiler that can support at least C++14. (The README for PyTorch lists various other dependencies, but we found that we were able to successfully build PyTorch using `setup.py develop` without them, oddly.)

The `run_dashboard.sh` script in `dashboard/dashboard` ensures that references to CUDA and CuDNN are found on the user's `PATH`, which is necessary for PyTorch (including the standard distribution) to work.

### Python Dependencies

Requires the dependencies in `requirements.txt`. Please install these files in whatever Python environments you wish to use by running `pip3 install -r requirements.txt`. This should also be done for the `venv` for installing the DTR-modified PyTorch (`dtr_venv/bin/pip3 install -r requirements.txt`).

The Unrolled GAN implementation in `dtr_eval/shared/torch_models/unroll_gan` also requires the library Higher, which could not be included in `requirements.txt`, so if you want to execute that model for logging, you must install Higher as follows:

```
git clone git@github.com:facebookresearch/higher.git
cd higher
pip3 install .
~/dtr_venv/bin/pip3 install .
```

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

See `dtr_eval/shared/validate_config.py` for a list of all included models, taken from various public implementations (noted in their definitions in `dtr_eval/shared/torch_models`). 

### Saving DTR Logs

The config provides the options `save_logs` (false by default) and `log_dest` (only used if `save_logs` is true) to copy over the DTR logs produced in an experiment, if any are produced. If `save_logs` is set to true, the experiment will copy over the last log produced when running a command to the specified destination directory. This allows for producing logs (which contain timings) to benefit from the warm-up runs in the dashboard.

The final run in the log will be marked with a "`START`" annotation.

### Experiment Commands

See `dtr_configs/README.md` for a description of configuration settings.

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
