import pandas as pd
# sklearn:
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix


def path(uniprot_id):
    return f'./data/{uniprot_id}/{uniprot_id}'


def carga_datos(uniprot_id, resample=True):
    # Carga de datos
    path_file = path(uniprot_id)
    df_target = pd.read_pickle(f'{path_file}_dataset')

    if resample:
        fp_df = df_target.copy()
        fp_df_active = fp_df[fp_df.activity == 1]
        fp_df_inactive = fp_df[fp_df.activity == 0]
        if len(fp_df_active) > len(fp_df_inactive):
            fp_df_active = fp_df_active.sample(len(fp_df_inactive))
        elif len(fp_df_inactive) > len(fp_df_active):
            fp_df_inactive = fp_df_inactive.sample(len(fp_df_active))
        fp_df_down = pd.concat([fp_df_active, fp_df_inactive], ignore_index=True).sample(frac=1)
        fp_df_down.reset_index(drop=True, inplace=True)
        df_target = fp_df_down.copy()

    # Train and test sets
    X = df_target.drop(['activity'], axis=1)
    X = X[fingerprint]
    y = df_target['activity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y,
                                                        random_state=seed)

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    X_test = X_test.tolist()

    return X_train, X_test, y_train, y_test


def param_grid_mod(params_grid):
    try:
        params_grid['max_samples'] = float(params_grid['max_samples'])
    except:
        params_grid['max_samples'] = None

    try:
        params_grid['max_features'] = float(params_grid['max_features'])
    except:
        params_grid['max_features'] = params_grid['max_features']
    return params_grid


def model_clf_grid_search(params_dict):
    import timeit, csv
    import numpy as np

    start = timeit.default_timer()
    results = list(params_dict.values())

    # Labels initialized with -1 for each data-point
    pred_train = -1 * np.ones(len(X_train))
    prediction_prob_train = -1 * np.ones(len(X_train))

    model.set_params(**params_dict)

    # N-SPLITS - 80% train set
    # Shuffle the indices
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_index, test_index in skf.split(X_train, y_train):
        x_train_fold, x_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        x_train_fold, x_test_fold = x_train_fold.tolist(), x_test_fold.tolist()
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
        model.fit(x_train_fold, y_train_fold)
        # Save the predicted label and prob of each fold
        pred_train[train_index] = model.predict(x_train_fold)
        prediction_prob_train[train_index] = model.predict_proba(x_train_fold)[:, 1]

    # score TRAIN
    auc_score_train = roc_auc_score(y_train, prediction_prob_train)
    acc_score_train = accuracy_score(y_train, pred_train)
    sens_score_train = recall_score(y_train, pred_train)
    spec_score_train = (acc_score_train * len(y_train) - sens_score_train * sum(y_train)) / \
                       (len(y_train) - sum(y_train))
    prec_score_train = precision_score(y_train, pred_train, zero_division=1)
    confusion_train = confusion_matrix(y_train, pred_train)

    # score TEST
    pred_test = model.predict(X_test)
    prediction_prob_test = model.predict_proba(X_test)[:, 1]
    auc_score_test = roc_auc_score(y_test, prediction_prob_test)
    acc_score_test = accuracy_score(y_test, pred_test)
    sens_score_test = recall_score(y_test, pred_test)
    spec_score_test = (acc_score_test * len(y_test) - sens_score_test * sum(y_test)) / (len(y_test) - sum(y_test))
    prec_score_test = precision_score(y_test, pred_test, zero_division=1)
    f1_score_test = f1_score(y_test, pred_test)
    confusion_test = confusion_matrix(y_test, pred_test)

    stop = timeit.default_timer()

    results.extend([auc_score_train, acc_score_train, sens_score_train, spec_score_train, prec_score_train,
                    confusion_train.tolist(),
                    auc_score_test, acc_score_test, sens_score_test, spec_score_test, prec_score_test,
                    f1_score_test, confusion_test.tolist(), round(stop - start, 2)])

    with open(f'{uniprot_id}_results.csv', 'a') as file:
        writer = csv.writer(file, lineterminator='\n', delimiter=",")
        writer.writerow(results)

    return results


df = pd.read_csv('params_grid_RFC.csv', sep=',')
list_dict_params = df.to_dict('records')

uniprot_id = 'P49841'
fingerprint = 'maccs'
seed = 1
n_splits = 3
X_train, X_test, y_train, y_test = carga_datos(uniprot_id)
model = RandomForestClassifier()
