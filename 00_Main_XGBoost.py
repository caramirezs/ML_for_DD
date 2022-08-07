from lib.main_datasets_fun import uniprot_id_datasets
from lib.grid_XGBoost_fun import BayesSearchCV_XGBoost

from lib.main_func_p1 import timer, path
from lib.main_func_p4 import modelXGBoost_fit_scores
from lib.main_func_p4 import brier_score
from datetime import datetime
from collections import OrderedDict

import pandas as pd

# XGBoost library
import xgboost as xgb
import os

#####################################
# proteina (uniprot_ID)
uniprot_id = 'P49841'
metric = 'accuracy'
resample_factor = 0
resample_mode = 'under_sampling'
gpu_id = 1
#####################################

# Parametros
seed = 142854
fp_name = 'morgan2_c'
frac_iter = 0.5  # No. jobs  50%
t_max = 2  # Max time 60h
path_file = path(uniprot_id)
#####################################
# BayesSearchCV
uniprot_id_datasets(uniprot_id, fp_name=fp_name, seed=seed)
excel_name = BayesSearchCV_XGBoost(uniprot_id, fp_name=fp_name, seed=seed, t_max=t_max, frac_iter=frac_iter,
                                   gpu_id=gpu_id, metric=metric, resample_factor=resample_factor,
                                   resample_mode=resample_mode)
# https://scikit-learn.org/stable/modules/model_evaluation.html
os.system('clear')  # Limpiar pantalla
print('>> Proceso BayesSearchCV terminado')
print('--------------------------------------------------------')

#####################################
# Tuned model
name_grid_file = f'./grid_results/{excel_name}.xlsx'

# Cargar archivo / eliminar columnas innecesarias
df_ori = pd.read_excel(name_grid_file, sheet_name=0)
df_grid_results = df_ori[['params', 'mean_test_score', 'std_test_score', 'rank_test_score', 'mean_train_score',
                          'std_train_score', 'rank_train_score']]
df_grid_results = df_grid_results.drop_duplicates()

# Organizar dataframe rank_test_score
df_grid_results['delta_mean'] = abs(df_grid_results['mean_test_score'] - df_grid_results['mean_train_score'])
df_grid_results.sort_values(by=['delta_mean', 'rank_test_score'], ascending=True, inplace=True)

# df_grid_results.sort_values(by=['rank_test_score'], ascending=True, inplace=True)
df_grid_results.reset_index(drop=True, inplace=True)

print(f'Duplicados eliminados: {len(df_ori) - len(df_grid_results)}. Total datos: {len(df_grid_results)}')

# Load train and validation datasets
df_train = pd.read_pickle(f'{path_file}_dataset_train')
df_valid = pd.read_pickle(f'{path_file}_dataset_valid')

params_dict_len = min(int(0.8 * len(df_grid_results)), 500)

tick_main = timer()

head_names = ['model', 'params_dict', 'AUC_train', 'AUC_valid', 'accuracy_train', 'accuracy_valid', 'recall_train',
              'recall_valid',
              'specificity_train', 'specificity_valid', 'precision_train', 'precision_valid',
              'f1_score_train', 'f1_score_valid', 'conf_matrix_train', 'conf_matrix_valid']

new_row_list = list()
df_list = list()
plots_name_list = list()

print(f'Process starting, protein ID {uniprot_id}')

for i, params_dict in enumerate(df_grid_results['params'].iloc[:params_dict_len]):
    eval_metric = ['error', 'auc']
    tick = timer()
    params_dict = dict(eval(params_dict))
    default_params_xgb = {'booster': 'gbtree', 'tree_method': 'gpu_hist',
                          'objective': 'binary:logistic', 'grow_policy': 'depthwise',
                          'eval_metric': eval_metric, 'early_stopping_rounds': 10}
    params_dict.update(default_params_xgb)

    xgb_clf_ini = xgb.XGBClassifier(**params_dict)

    # Cross validation XGBoost
    # xgb_clf = model_cv(params_dict, df_train, fp_name)

    # Train model and evaluating scores (train / validation)
    xgb_clf, scores_train, scores_valid = modelXGBoost_fit_scores(xgb_clf_ini, fp_name, df_train, df_valid,
                                                                  resample_factor=resample_factor,
                                                                  resample_mode=resample_mode)

    new_row = [f'modelID_{i}', params_dict, scores_train[0], scores_valid[0], scores_train[1], scores_valid[1],
               scores_train[2], scores_valid[2], scores_train[3], scores_valid[3], scores_train[4], scores_valid[4],
               scores_train[5], scores_valid[5], scores_train[6], scores_valid[6]]
    new_row_list.append(new_row)

    # save pred and pred_prob of train set
    df = df_train[['activity', 'prediction', 'prediction_prob']].copy()
    df_list.append(df)

    plots_name_list.append(f'modelID_{i}')
    print(
        f'Model {i}/{params_dict_len}. AUC_socre=(train={scores_train[0]}, '
        f'valid={scores_valid[0]}). Time elapsed: {timer(tick)}')
top_scores = pd.DataFrame(new_row_list, columns=head_names)
print(f'Total time elapsed: {timer(tick_main)}')

# Calibration scores (sklearn.metrics.brier_score_loss)
clf_score_list = brier_score(df_list)
top_scores['calibration_score'] = clf_score_list
top_scores.sort_values(by=['calibration_score'], inplace=True)

# Save top_records
now = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
excel_name = f'./top_scores/{uniprot_id}_{now}_top_scores_XGBClassifier_{metric}.xlsx'
top_scores.to_excel(excel_name, sheet_name=uniprot_id, index=False)
print(f'file {excel_name} save')

#####################################
