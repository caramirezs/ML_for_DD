from lib.main_datasets_fun import uniprot_id_datasets
from lib.grid_XGBoost_fun import BayesSearchCV_XGBoost

# Parameters
uniprot_id = 'P56817'  # protein
fp_name = 'morgan2_c'  # Fingerprint
seed = 142857
frac_iter = 0.5  # No. jobs  50%
t_max = int(1000*60)  # Max time 60h
gpu_id = 2

# https://scikit-learn.org/stable/modules/model_evaluation.html

uniprot_id_datasets(uniprot_id, fp_name=fp_name, seed=seed)
BayesSearchCV_XGBoost(uniprot_id, fp_name=fp_name, seed=seed, t_max=t_max, frac_iter=frac_iter, gpu_id=gpu_id,
                      scoring='jaccard', resample_factor=4, resample_mode='under_sampling')
