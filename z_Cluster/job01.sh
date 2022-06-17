#!/bin/bash
#SBATCH -J job01.sh
#SBATCH --partition=defq
#SBATCH --get-user-env
#SBATCH --nodes=1
#SBATCH --tasks-per-node=20
#SBATCH --gres=gpu:4

srun ./ML_for_DD/python 01_XGBoost_BayesSearchCV.py