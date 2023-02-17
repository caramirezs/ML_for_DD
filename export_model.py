import pandas as pd
import xgboost as xgb
from lib.main_func_p1 import path
from lib.main_func_p4 import modelXGBoost_fit_scores

#INPUT
uniprot_id = 'P00533'
best_model_excel = f'P00533_20220823081135_top_scores_XGBClassifier_precision_rf0.xlsx'

# Params
path_file = path(uniprot_id)
seed = 142854
fp_name = 'morgan2_c'
score_metric = '_'.join(str.split(best_model_excel, '_')[5:-1])
resample_factor = int(str.split(best_model_excel, '_')[-1][2:][:-5])
resample_mode = 'under_sampling'
model_name = f'XGBoost_Clf'
top_scores = pd.read_excel(f'./top_scores/{uniprot_id}/{best_model_excel}')

# Load train and validation datasets
df_train = pd.read_pickle(f'{path_file}_dataset_train')
df_valid = pd.read_pickle(f'{path_file}_dataset_valid')
X_valid, y_valid = df_valid[fp_name].tolist(), df_valid['activity'].tolist()

eval_metric = ['error', 'auc']
params_dict = dict(eval(top_scores.params_dict[0]))
default_params_xgb = {'booster': 'gbtree', 'tree_method': 'gpu_hist',
                      'objective': 'binary:logistic', 'grow_policy': 'depthwise',
                      'eval_metric': eval_metric, 'early_stopping_rounds': 10}
params_dict.update(default_params_xgb)

# train and save the 'best' model
xgbc_tuned = xgb.XGBClassifier(**params_dict)
# Train model and evaluating scores (train / validation)
xgbc_tuned, scores_train, scores_valid = modelXGBoost_fit_scores(xgbc_tuned, fp_name, df_train, df_valid,
                                                                 resample_factor=resample_factor,
                                                                 resample_mode=resample_mode, verbose=False)
xgbc_tuned.save_model(f'./models/{uniprot_id}_model.ubj')