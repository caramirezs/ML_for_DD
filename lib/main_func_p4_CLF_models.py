from lib.main_func_p1 import path
import numpy as np
import pandas as pd

# sklearn:
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def create_param_grid(param_grid, file_name):
    results = list()
    for i_0 in param_grid[list(param_grid)[0]]:
        for i_1 in param_grid[list(param_grid)[1]]:
            for i_2 in param_grid[list(param_grid)[2]]:
                for i_3 in param_grid[list(param_grid)[3]]:
                    for i_4 in param_grid[list(param_grid)[4]]:
                        for i_5 in param_grid[list(param_grid)[5]]:
                            for i_6 in param_grid[list(param_grid)[6]]:
                                for i_7 in param_grid[list(param_grid)[7]]:
                                    results.append([i_0, i_1, i_2, i_3, i_4, i_5, i_6, i_7])
    # print(len(results))
    # print(list(param_grid.values()))
    # print(list(map(len, list(param_grid.values()))))
    df = pd.DataFrame(results, columns=list(param_grid.keys()))
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(file_name, sep=',', na_rep='None', index=False)
    return None


def create_results_file(param_grid, file_name):
    import csv
    head_params = list(param_grid.keys())
    columns_train = ['auc_score_train', 'accuracy_score_train', 'sens_score_train',
                     'spec_score_train', 'prec_score_train', 'confusion_m_train']
    columns_test = ['auc_score_test', 'accuracy_score_test', 'sens_score_test',
                    'spec_score_test', 'prec_score_test', 'f1_score_test', 'confusion_m_test', 'time']
    columns_name = list(head_params)
    columns_name.extend(columns_train)
    columns_name.extend(columns_test)

    with open(file_name, 'w') as file:
        writer = csv.writer(file, lineterminator='\n', delimiter=",")
        writer.writerow(columns_name)
    return None


def param_grid_mod(dict_params):
    try:
        dict_params['max_samples'] = float(dict_params['max_samples'])
    except:
        dict_params['max_samples'] = None

    try:
        dict_params['max_features'] = float(dict_params['max_features'])
    except:
        dict_params['max_features'] = dict_params['max_features']
    return dict_params


def model_clf_fp(model, fp_df, fp_list, uniprot_id, params_dict=None, seed=1, n_splits=5):
    import timeit

    columns = ['FINGERPRINT', 'AUC_train', 'acc_train', 'sen_train', 'spe_train', 'pre_train',
               'AUC_test', 'acc_test', 'sen_test', 'spe_test', 'pre_test', 'f1_score', 'confusion_m', 'time']
    list_results = list()
    results_ROC_fp = list()

    best_AUC = 0
    best_fp = list()

    # Separar atributos y etiqueta
    X = fp_df.drop(['activity'], axis=1)
    y = fp_df['activity']

    # Conjunto de entrenamiento y prueba (80% train / 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y,
                                                        random_state=seed)
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    if params_dict is not None:
        model.set_params(**params_dict)

    for fp_name in fp_list:
        start = timeit.default_timer()
        X_train_fp = X_train[fp_name]
        X_test_fp = X_test[fp_name].tolist()

        # Labels initialized with -1 for each data-point
        pred_train = -1 * np.ones(len(X_train_fp))
        prediction_prob_train = -1 * np.ones(len(X_train_fp))
        # Shuffle the indices
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        # skf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        i = 1
        for train_index, test_index in skf.split(X_train_fp, y_train):
            x_train_fold, x_test_fold = X_train_fp.iloc[train_index], X_train_fp.iloc[test_index]
            x_train_fold, x_test_fold = x_train_fold.tolist(), x_test_fold.tolist()
            y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
            model.fit(x_train_fold, y_train_fold)
            # Save the predicted label and prob of each fold
            pred_train[train_index] = model.predict(x_train_fold)
            prediction_prob_train[train_index] = model.predict_proba(x_train_fold)[:, 1]
            # print(f'fp: {fp_name}, fold: {i} ok')
            i += 1

        # score TRAIN
        auc_score_train = roc_auc_score(y_train, prediction_prob_train)
        acc_score_train = accuracy_score(y_train, pred_train)
        sens_score_train = recall_score(y_train, pred_train)
        spec_score_train = (acc_score_train * len(y_train) - sens_score_train * sum(y_train)) / \
                           (len(y_train) - sum(y_train))
        prec_score_train = precision_score(y_train, pred_train, zero_division=1)

        # score TEST
        pred_test = model.predict(X_test_fp)
        prediction_prob_test = model.predict_proba(X_test_fp)[:, 1]
        # Get fpr, tpr and roc_auc TEST for each fp
        fpr_test, tpr_test, _ = roc_curve(y_test, prediction_prob_test)
        auc_score_test = auc(fpr_test, tpr_test)
        acc_score_test = accuracy_score(y_test, pred_test)
        sens_score_test = recall_score(y_test, pred_test)
        spec_score_test = (acc_score_test * len(y_test) - sens_score_test * sum(y_test)) / (len(y_test) - sum(y_test))
        prec_score_test = precision_score(y_test, pred_test, zero_division=1)
        f1_score_test = f1_score(y_test, pred_test)
        confusion = confusion_matrix(y_test, pred_test)

        results_ROF_test = (fpr_test, tpr_test, auc_score_test)

        # Metrica de comparación AUC_TEST
        if auc_score_test > best_AUC:
            best_fp = [fp_name, auc_score_train, auc_score_test, acc_score_train, acc_score_test, f1_score_test]
            best_AUC = auc_score_test

        stop = timeit.default_timer()

        results_ROC_fp.append(results_ROF_test)

        list_results.append([fp_name, auc_score_train, acc_score_train, sens_score_train, spec_score_train,
                             prec_score_train, auc_score_test, acc_score_test, sens_score_test, spec_score_test,
                             prec_score_test, f1_score_test, confusion, stop - start])

    print('Results %s:' % str(model).split('(')[0],
          '\n-------------------------------------')
    df_model = pd.DataFrame(list_results, columns=columns)
    # print('best model found (auc_score_test):')
    # print('Fingerprint: {}| auc_score_train: {:.3f}| auc_score_test: {:.3f}|'
    #       ' acc_score_train {:.3f}| acc_score_test: {:.3f}|'.format(best_fp[0], best_fp[1], best_fp[2],
    #                                                                 best_fp[3], best_fp[4]))
    # All ROC resulst
    return df_model, results_ROC_fp


def model_clf(model, fp_df, fp_name, uniprot_id, params_dict=None, seed=1, n_splits=5):
    # Separar atributos y etiqueta
    X = fp_df.drop(['activity'], axis=1)
    X = X[fp_name]
    y = fp_df['activity']

    # Conjunto de entrenamiento y prueba (80% train / 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y,
                                                        random_state=seed)
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    path_file = path(uniprot_id)
    test_set = pd.concat([X_test, y_test], axis=1)
    test_set.to_pickle(f'{path_file}_dataset_test')

    X_test = X_test.tolist()

    if params_dict is not None:
        model.set_params(**params_dict)

    # Labels initialized with -1 for each data-point
    pred_train = -1 * np.ones(len(X_train))
    prediction_prob_train = -1 * np.ones(len(X_train))

    # Shuffle the indices
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # skf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
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
    f1_score_train = f1_score(y_train, pred_train)
    fpr_train, tpr_train, _ = roc_curve(y_train, prediction_prob_train)
    results_ROF_train = (fpr_train, tpr_train, auc_score_train)

    # score TEST
    pred_test = model.predict(X_test)
    prediction_prob_test = model.predict_proba(X_test)[:, 1]
    auc_score_test = roc_auc_score(y_test, prediction_prob_test)
    acc_score_test = accuracy_score(y_test, pred_test)
    sens_score_test = recall_score(y_test, pred_test)
    spec_score_test = (acc_score_test * len(y_test) - sens_score_test * sum(y_test)) / \
                      (len(y_test) - sum(y_test))
    prec_score_test = precision_score(y_test, pred_test, zero_division=1)
    f1_score_test = f1_score(y_test, pred_test)
    confusion_test = confusion_matrix(y_test, pred_test)
    fpr_test, tpr_test, _ = roc_curve(y_test, prediction_prob_test)
    results_ROF_test = (fpr_test, tpr_test, auc_score_test)

    results_ROC = [results_ROF_train, results_ROF_test]

    list_results = [['AUC', auc_score_train, auc_score_test],
                    ['accuracy', acc_score_train, acc_score_test],
                    ['sensitivity (recall)', sens_score_train, sens_score_train],
                    ['specificity', spec_score_train, spec_score_test],
                    ['precision', prec_score_train, prec_score_test],
                    ['f1_score', f1_score_train, f1_score_test],
                    ['confusion_matrix', confusion_train, confusion_test]]

    print('Results %s:' % str(model).split('(')[0],
          '\n-------------------------------------')
    print(classification_report(y_test, pred_test))
    df_model = pd.DataFrame(list_results, columns=['Metric', 'Train', 'Test'])

    return df_model, results_ROC


def plot_ROC_curve(metrics_ROC, metrics_ROC_name, model_name):
    # matplotlib:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    cmap = cm.get_cmap('Blues')
    colors = [cmap(i) for i in np.linspace(0.3, 1.0, len(metrics_ROC))]

    plt.figure(1, figsize=(7, 7))

    for i, metrics in enumerate(metrics_ROC):
        fpr, tpr, roc_auc = metrics[0], metrics[1], metrics[2]
        name = metrics_ROC_name[i]
        plt.plot(fpr, tpr, lw=2, color=colors[i], label='AUC_{} = {:.3f}'.format(name, roc_auc))
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random', lw=2, color="black")  # Random curve
    plt.xlabel('False positive rate', size=24)
    plt.ylabel('True positive rate', size=24)
    plt.title(model_name + ' ROC curve', size=24)
    plt.tick_params(labelsize=16)
    plt.legend(fontsize=16, loc=(1.04, 0))
    return None
