# simrd
Simulated (tensor) rematerialization, dynamic


## Setup
### Step 1. Install Anaconda and set up environment
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

### Step 2. Install dependencies and unzip logs
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


## Run Simulations and Plotting
Finally, run the simulated evaluation by running
```
python simrd_experiments/eval/pareto/main.py
```

The resulting data and figures can be found under the `data` directory.
Note that this can take a few hours due to the ablation study.
