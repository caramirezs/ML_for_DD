from lib.main_datasets_fun import uniprot_id_datasets
from lib.grid_XGBoost_fun import BayesSearchCV_XGBoost

#####################################
# proteina (uniprot_ID)
uniprot_id = 'P36544-1'
metric='precision'
resample_factor = 3
resample_mode = 'over_sampling'
gpu_id = 0

# Parametros
seed = 142854
fp_name = 'morgan2_c'
frac_iter = 0.5  # No. jobs  50%
t_max = int(24*4*60)  # Max time 60h

uniprot_id_datasets(uniprot_id, fp_name=fp_name, seed=seed)
BayesSearchCV_XGBoost(uniprot_id, fp_name=fp_name, seed=seed, t_max=t_max, frac_iter=frac_iter, gpu_id=gpu_id,
                      metric=metric, resample_factor=resample_factor, resample_mode=resample_mode)
# https://scikit-learn.org/stable/modules/model_evaluation.html
