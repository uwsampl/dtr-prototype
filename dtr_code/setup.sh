#!/bin/bash
# All one-time setup from the readme. Should be run from dtr_code
# Will create a new python venv with DTR-modified pytorch installed
# Warning: building PyTorch can take a long time

# You can put the DTR venv wherever you'd like,
# but the provided configs assume it is in the home directory;
# you can change the "dtr_torch_cmd" field in
# dtr_home/config/experiments/pareto_curve/config.json
# to point to the right python executable
python3 -m venv ~/dtr_venv

pip3 install -r requirements.txt
~/dtr_venv/bin/pip3 install -r requirements.txt

cd dtr_pytorch
# The lengthy installation of PyTorch
~/dtr_venv/bin/python3 setup.py develop
cd ..

# Comment out if you do not want to run treelstm_old;
# this can take over 10 minutes to run
cd dtr_eval/share/torch_models/treelstm
./fetch_and_preprocess.sh
cd ../../../..

# Comment out if you do not want to run unroll_gan,
# which needs higher
git clone git@github.com:facebookresearch/higher.git
cd higher
pip3 install .
~/dtr_venv/bin/pip3 install .

# You can now run the experiments from the dtr_code directory with the following command:
# ./dashboard/dashboard/run_dashboard.sh ./dtr_home ./dtr_eval/dtr_experiments
