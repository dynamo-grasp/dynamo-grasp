#!/bin/bash

# Check if a Conda environment is active
if [ -z "$CONDA_PREFIX" ]; then
    echo "No Conda environment is active."
    exit 1
elif [ "$CONDA_DEFAULT_ENV" == "base" ]; then
    echo "The base Conda environment is active."
    echo "Please activate a different Conda environment and rerun the script."
    echo "Example:"
    echo "  conda activate [YourEnvName]"
    exit 1
else
    echo "Active Conda environment path: $CONDA_PREFIX"
fi

# Construct and export LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib

# Check for the correct number of arguments and provide a helpful usage message if incorrect
if [ "$#" -lt 2 ]; then
    echo "Error: Incorrect number of arguments"
    echo "Usage: $0 --num-envs NUM_ENVS"
    echo
    echo "    NUM ENVS  : Integer representing the number of environments to run."
    echo "                Select this number based on your computational resources."
    exit 1
fi

python data_collection.py "$@"