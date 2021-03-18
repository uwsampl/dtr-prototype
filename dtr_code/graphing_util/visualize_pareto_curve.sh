#!/bin/sh
rm ./*.png
export PYTHONPATH="${PYTHONPATH}:../dashboard/shared/python:../shared"
python3 run_visualize.py $1 $2
