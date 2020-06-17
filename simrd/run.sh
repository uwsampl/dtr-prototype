#!/bin/bash
source ./venv/bin/activate
env PYTHONPATH=.:$PYTHONPATH python3 experiments/eval/pareto/main.py
