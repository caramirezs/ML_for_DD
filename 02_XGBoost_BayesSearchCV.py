from lib.main_datasets_fun import uniprot_id_datasets
from lib.grid_XGBoost_fun import BayesSearchCV_XGBoost

#####################################
# proteina (uniprot_ID)
uniprot_id = 'P49841'
# Parametros
seed = 142854
fp_name = 'morgan2_c'


frac_iter = 0.5  # No. jobs  50%
t_max = int(24*4*60)  # Max time 60h
gpu_id = 1
scoring='accuracy'
resample_factor = 0
resample_mode = 'under_sampling'

# https://scikit-learn.org/stable/modules/model_evaluation.html

uniprot_id_datasets(uniprot_id, fp_name=fp_name, seed=seed)
BayesSearchCV_XGBoost(uniprot_id, fp_name=fp_name, seed=seed, t_max=t_max, frac_iter=frac_iter, gpu_id=gpu_id,
                      scoring=scoring, resample_factor=resample_factor, resample_mode=resample_mode)
