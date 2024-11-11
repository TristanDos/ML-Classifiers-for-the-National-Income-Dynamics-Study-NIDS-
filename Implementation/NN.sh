#!/bin/bash

# Source the .bashrc file
source ~/.bashrc

# Activate the conda environment
conda activate Research

# Submit the job multiple times using sbatch
sbatch NN.batch

# Monitor the job queue
watch -n 1 squeue -u tdremendos