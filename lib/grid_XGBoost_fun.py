from lib.main_func_p1 import path, timer
import pandas as pd
import numpy as np
from datetime import datetime

# XGBoost library
import xgboost as xgb

# Grid search and cross-validation
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV

# calllbacks funtion
from skopt.callbacks import DeadlineStopper, DeltaYStopper

####
# https://scikit-learn.org/stable/modules/model_evaluation.html
# from sklearn import metrics
# print(sorted(metrics.SCORERS.keys()))
####

def BayesSearchCV_XGBoost(uniprot_id, fp_name='morgan2_c', seed=142857, t_max=10, frac_iter=0.25, gpu_id=0,
                          scoring='f1_weighted', resample_factor=0, resample_mode='under_sampling'):
    from lib.main_func_p4 import resampling_set

    print('---------------------------------------------------------')
    if resample_factor != 0:
        print(F'CONFIGURACIONES: scoring: {scoring}, resample: {resample_factor} / {resample_mode}')
    else:
        print(F'CONFIGURACIONES: scoring: {scoring}')

    # Load datasets
    path_file = path(uniprot_id)
    df_set = pd.read_pickle(f'{path_file}_dataset_train')
    print(f'>>> LOAD: {uniprot_id}_dataset_train')

    # scale_pos_weigh parameter
    negative_class = max(list(df_set['activity'].value_counts()))
    positive_class = min(list(df_set['activity'].value_counts()))
    spw = round(negative_class / positive_class, 2)  # Mayority / minority
    if spw > 3:
        list_spw = [1.0, 1.5, 2.0, spw, round(0.85*spw, 2), round(0.88*spw, 2), round(0.91*spw, 2), round(0.94*spw),
                    round(0.97*spw, 2), round(1.03*spw, 2), round(1.06*spw, 2), round(1.09*spw),
                    round(1.12*spw, 2), round(1.15*spw, 2)]
        print('scale_pos_weight')
        print(list_spw)
    else:
        list_spw = [1.0]
    print('---------------------------------------------------------')

    # Balanced samplers
    if resample_factor != 0:
        ori_compounds_len = len(df_set)
        df_set_rsmp = resampling_set(df_set, mode=resample_mode, ratio=resample_factor)
        print(f'{resample_mode} - {resample_factor}: {ori_compounds_len} to {len(df_set_rsmp)}')
    else:
        df_set_rsmp = df_set.copy()

    # Separate attributes and tags
    X_set, y_set = df_set_rsmp[fp_name], df_set_rsmp['activity']

    # train test split with randomization performed (although randomization is not necessary)
    X_train, X_test, y_train, y_test = train_test_split(X_set, y_set, test_size=0.2, random_state=seed, stratify=y_set)

    X_train, X_test = X_train.to_list(), X_test.to_list()

    # setting grid
    param_grid = {
        # Tree boster parameter. L1 regularization term on weights.
        'alpha': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
        # Tree boster parameter. Minimum loss reduction required to make a further partition on a leaf node of the tree
        'gamma': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        # Tree boster parameter. L2 regularization term on weights.
        'lambda': [1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.5, 3, 4],
        # Tree boster parameter. Step size shrinkage used in update to prevents overfitting
        'learning_rate': [0.05, 0.1, 0.2, 0.25, 0.28, 0.3, 0.32, 0.35, 0.4, 0.45, 0.5],
        # Tree boster parameter. Maximum depth of a tree.
        'max_depth': [5, 6, 7, 8, 9, 10, 11, 12],
        # Tree boster parameter. Minimum sum of instance weight (hessian) needed in a child.
        'min_child_weight': [1, 2, 3],
        # Control the balance of positive and negative weights
        'scale_pos_weight': list_spw,
        # Tree boster parameter. Subsample ratio of the training instances.
        'subsample': [0.6, 0.7, 0.8, 0.9, 1]}

    n_combinations = np.cumproduct([len(x) for x in param_grid.values()])[-1]
    # No. of jobs
    n_iter = frac_iter * n_combinations

    print(f'No. total of posible combinations: {n_combinations}')

    # list values of default parameters
    default_params_xgb = {'booster': 'gbtree', 'tree_method': 'gpu_hist', 'gpu_id': gpu_id,
                          'objective': 'binary:logistic', 'eval_metric': 'auc',
                          'grow_policy': 'depthwise'}

    # xgbc model
    xgbc = xgb.XGBClassifier(**default_params_xgb)

    # start time
    tick = timer()
    # Executing BayesSearchCV
    clf = BayesSearchCV(estimator=xgbc, search_spaces=param_grid, cv=5, n_iter=n_iter, n_jobs=-1,
                        scoring=scoring, refit=True,
                        return_train_score=True, verbose=3, random_state=seed)

    # Callback
    overdone_control = DeltaYStopper(delta=1e-4)
    time_limit_control = DeadlineStopper(total_time=60 * t_max)

    def cb(result):
        status = True
        iters = result.x_iters
        score = round(-result['fun'], 3)
        print(f'---------------------------------------------------------')
        print(f'Protein_id: {uniprot_id} - gpu_id:{gpu_id}')
        if resample_factor !=0:
            print(f'resample_factor: {resample_factor} - {resample_mode}')

        print(f'>>>> Iteration {len(iters)} of {int(n_iter)} done. '
              f'scoring: {scoring} - Score: {score}. \n'
              f'Time elapsed: {timer(tick)}')
        try:
            df_temp = pd.read_csv('early_stops.csv', index_col=0)
            status = df_temp.loc[uniprot_id]['early_stop']
            print(f'>>>> Early stop status: {status}')
        except:
            print(f'>>>> No early stop implemented')
            pass
        if status:
            print('>>>>>>>> Entrenamiento terminado por usuario')
        print(f'---------------------------------------------------------')
        return status

    clf.fit(X_train, y_train, callback=[overdone_control, time_limit_control, cb])

    # elapsed time
    print(f'Process finished, Total time elapsed: {timer(tick)}')
    print(f'---------------------------------------------------------')
    # results dataframe
    df = pd.DataFrame(clf.cv_results_)
    now = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
    file_name = f'./grid_results/{uniprot_id}_{now}_BayesSearchCV_XGBoots_{scoring}_rf{resample_factor}.xlsx'
    df.to_excel(file_name, sheet_name=uniprot_id, index=False)
    print(f'Exported file: {file_name}')
    return df
