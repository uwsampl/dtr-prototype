# Dynamic Tensor Rematerialization vs Checkmate Baselines

This document contains instructions for reproducing plots in the ICLR 2021 paper "Dynamic Tensor Rematerialization".

**NOTE: this README has been heavily adapted from Checkmate's README by the
authors of Dynamic Tensor Rematerialization to work with their ICLR
supplementary materials submission.**

## Installation
### Step 0: Set up DTR simulator (simrd)
First, follow the setup instructions for the DTR simulator (simrd), which should
be bundled with this in the parent directory.

### Step 1: Install Anaconda
Make sure that you have created the DTR Anaconda environment and have it
activated, as per the simulator instructions. Activate the environment by:
```
conda activate dtr-iclr21
```

### Step 2: Install the Checkmate `remat` package and dependencies
From this directory,
```
$ conda install -c conda-forge python-graphviz
$ pip install -e .
```
Next, install tensorflow with
```
$ pip install tensorflow==2.0.1
```

### Step 3: Install Gurobi
Checkmate uses the Gurobi optimziation library to solve an integer linear program that chooses a recomputation schedule for a given neural network architecture. This requires a license to Gurobi, which is free for academic use. The `grbgetkey` command used below must be run on a computer connected to a university network directly or via a VPN.

1. Please follow these instructions to install Gurobi on your system: https://www.gurobi.com/documentation/quickstart.html. For example, on Linux, follow `https://www.gurobi.com/documentation/9.0/quickstart_linux/software_installation_guid.html`.
2. Make an academic account with Gurobi at: https://pages.gurobi.com/registration
3. Request an acadmic license at: https://www.gurobi.com/downloads/end-user-license-agreement-academic/
4. Install the license by running the `grbgetkey` command at the end of the page. Ensure you are on an academic network like Airbears2 for UC Berkeley. If you save the license to a non-default location (outside your home directory), you will need to export the `GRB_LICENSE_FILE` variable with the path to the licence.
5. Set up the gurobipy Anaconda channel by running `conda config --add channels http://conda.anaconda.org/gurobi`
6. Install gurobipy by running: `conda install gurobi`


## Reproducing Figure 2: Computational overhead versus memory budget
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
