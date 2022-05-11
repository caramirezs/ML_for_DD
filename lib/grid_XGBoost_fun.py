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
# from sklearn import metrics
# print(sorted(metrics.SCORERS.keys()))
####

def BayesSearchCV_XGBoost(uniprot_id, fp_name='morgan2_c', seed=142857, t_max=10, frac_iter=0.25, gpu_id=0,
                          scoring='roc_auc'):
    path_file = path(uniprot_id)

    # Load datasets
    df_set = pd.read_pickle(f'{path_file}_dataset_train')
    print(f'>>> LOAD: {uniprot_id}_dataset_train')
    df_unseen = pd.read_pickle(f'{path_file}_dataset_test')
    print(f'>>> LOAD: {uniprot_id}_dataset_test')

    # Separate attributes and tags
    X_set, y_set = df_set[fp_name], df_set['activity']

    # train test split with randomization performed (although randomization is not necessary)
    X_train, X_test, y_train, y_test = train_test_split(X_set, y_set, test_size=0.2, random_state=seed, stratify=y_set)
    X_unseen, y_unseen = df_unseen[fp_name], df_unseen['activity']

    X_train, X_test, X_unseen = X_train.to_list(), X_test.to_list(), X_unseen.to_list()

    # setting grid
    param_grid = {
        # Tree boster parameter. Maximum depth of a tree.
        'max_depth': [3, 4, 5, 6, 7, 8, 9],
        # Tree boster parameter. Step size shrinkage used in update to prevents overfitting
        'learning_rate': [0.01, 0.02, 0.07, 0.1, 0.2, 0.3],
        # Tree boster parameter. Minimum loss reduction required to make a further partition on a leaf node of the tree
        'gamma': [0, 0.2, 0.5, 0.8, 1.5, 3, 7, 12.8, 25.6, 51.2, 102.4, 200],
        # Tree boster parameter. Minimum sum of instance weight (hessian) needed in a child.
        'min_child_weight': [1, 2, 5, 8, 12, 20],
        # Tree boster parameter. Subsample ratio of the training instances.
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
        # Tree boster parameter. L2 regularization term on weights.
        'lambda': [1, 1.5, 2, 4, 8, 16, 32],
        # Tree boster parameter. L1 regularization term on weights.
        'alpha': [0, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, ]}

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
        iters = result.x_iters
        print(f'----------------------------------------------------------------------------------------------')
        print(f'>>> Iteration {len(iters)} of {int(n_iter)} done')
        timer(tick)
        print(f'----------------------------------------------------------------------------------------------')

    clf.fit(X_train, y_train, callback=[overdone_control, time_limit_control, cb])

    # elapsed time
    print(f'Process finished, elapsed time:')
    timer(tick)
    print(f'----------------------------------------------------------------------------------------------')
    # results dataframe
    df = pd.DataFrame(clf.cv_results_)
    now = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
    file_name = f'{path_file}_BayesSearchCV_XGBoots_{scoring}_{now}.xlsx'
    df.to_excel(file_name, sheet_name=uniprot_id, index=False)
    print(f'Exported file: {file_name}')
    return df
