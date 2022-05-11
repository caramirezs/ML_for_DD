import numpy as np
import pandas as pd

# sklearn:
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def resampling_set(fp_df, mode='over_sampling', ratio=0):
    seed = 142857
    if ratio == 0:
        return fp_df

    fp_df_active = fp_df[fp_df.activity == 1]
    fp_df_inactive = fp_df[fp_df.activity == 0]
    old_ratio = round(len(fp_df_inactive) / len(fp_df_active), 2)
    new_ratio = min(old_ratio, ratio)

    if mode == 'under_sampling':
        # under_sampling
        n_sample = round(new_ratio * min(len(fp_df_inactive), len(fp_df_active)))
        if len(fp_df_active) > len(fp_df_inactive):
            fp_df_active = fp_df_active.sample(n_sample, random_state=seed, ignore_index=True)
        elif len(fp_df_inactive) > len(fp_df_active):
            fp_df_inactive = fp_df_inactive.sample(n_sample, random_state=seed, ignore_index=True)

    if mode == 'over_sampling':
        # over_sampling
        n_sample = round(max(len(fp_df_inactive), len(fp_df_active)) / new_ratio)
        if len(fp_df_active) > len(fp_df_inactive):
            fp_df_inactive = fp_df_inactive.sample(n_sample, random_state=seed, replace=True, ignore_index=True)
        elif len(fp_df_inactive) > len(fp_df_active):
            fp_df_active = fp_df_active.sample(n_sample, random_state=seed, replace=True, ignore_index=True)

    fp_df_down = pd.concat([fp_df_active, fp_df_inactive], ignore_index=True).sample(frac=1, random_state=seed)
    fp_df_down.reset_index(drop=True, inplace=True)
    return fp_df_down


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


# datos_iniciales = uniprot_id, fp_name, nombre_algoritmo, params_dict, n_splits,
# log = 'path/data/log.txt
def log_model(datos_iniciales, df_results_model, log=f'./data/log.csv', algoritmo='nombre_algoritmo'):
    import csv
    from datetime import datetime
    now = datetime.now()  # current date and tim
    date_time = now.strftime("%Y/%m/%d, %H:%M:%S")
    new_dato = [date_time, datos_iniciales[0], datos_iniciales[1], algoritmo, datos_iniciales[2], datos_iniciales[3],
                df_results_model.loc['AUC']['Train'], df_results_model.loc['accuracy']['Train'],
                df_results_model.loc['sensitivity_(recall)']['Train'], df_results_model.loc['specificity']['Train'],
                df_results_model.loc['precision']['Train'], df_results_model.loc['f1_score']['Train'],
                df_results_model.loc['confusion_matrix']['Train'].tolist(),
                df_results_model.loc['AUC']['Test'], df_results_model.loc['accuracy']['Test'],
                df_results_model.loc['sensitivity_(recall)']['Test'], df_results_model.loc['specificity']['Test'],
                df_results_model.loc['precision']['Test'], df_results_model.loc['f1_score']['Test'],
                df_results_model.loc['confusion_matrix']['Test'].tolist()]
    with open(log, 'a') as file:
        writer = csv.writer(file, lineterminator='\n', delimiter=",")
        writer.writerow(new_dato)
    return None


def model_clf(model, fp_name, uniprot_id, params_dict=None, seed=142857, n_splits=5, save_log=False,
              resample_factor=0, resample_mode='under_sampling'):
    from lib.main_func_p1 import path
    import timeit, csv
    import numpy as np

    path_file = path(uniprot_id)
    # Import train/test dataset
    df_set = pd.read_pickle(f'{path_file}_dataset_train')
    ori_compounds_len = len(df_set)

    # Balanced samplers
    if resample_factor != 0:
        df_set = resampling_set(df_set, mode=resample_mode, ratio=resample_factor)
        print(f'{resample_mode} - {resample_factor}: {ori_compounds_len} to {len(df_set)}')

    # train test split
    X_set, y_set = df_set[fp_name], df_set['activity']
    X_train, X_test, y_train, y_test = train_test_split(X_set, y_set, test_size=0.2, random_state=seed, stratify=y_set)
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
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
        model.fit(x_train_fold, y_train_fold)
        # Save the predicted label and prob of each fold
        pred_train[train_index] = model.predict(x_train_fold)
        prediction_prob_train[train_index] = model.predict_proba(x_train_fold)[:, 1]

    # scores TRAIN
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

    # scores TEST
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
                    ['sensitivity_(recall)', sens_score_train, sens_score_train],
                    ['specificity', spec_score_train, spec_score_test],
                    ['precision', prec_score_train, prec_score_test],
                    ['f1_score', f1_score_train, f1_score_test],
                    ['confusion_matrix', confusion_train, confusion_test]]

    print('Results %s:' % str(model).split('(')[0],
          '\n-------------------------------------')
    print(classification_report(y_test, pred_test))
    df_results_model = pd.DataFrame(list_results, columns=['Metric', 'Train', 'Test'])
    df_results_model.set_index('Metric', inplace=True)

    # Archivar resultados en el log
    if save_log:
        nombre_algoritmo = str(type(model))[:-2].split('.')[-1]
        log = f'./log/log.csv'
        datos_iniciales = [uniprot_id, fp_name, params_dict, n_splits]
        log_model(datos_iniciales, df_results_model, log=log, algoritmo=nombre_algoritmo)

    return model, df_results_model, results_ROC


def plot_ROC_curve(metrics_ROC, metrics_ROC_name, model_name, path_file=None, save_fig=False):
    # matplotlib:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    cmap = cm.get_cmap('Blues')
    colors = [cmap(i) for i in np.linspace(0.3, 1.0, len(metrics_ROC))]

    plt.figure(1, figsize=(10, 10))

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
    if save_fig:
        plt.savefig(f'{path_file}_ROC_curve.png', bbox_inches='tight')
    plt.show()
    plt.close()
    return None


def plot_calibration_curve(df_prediction_prob, model_name, path_file=None, save_fig=False):
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    fop, mpv = calibration_curve(df_prediction_prob['activity'], df_prediction_prob['prediction_prob'], n_bins=10,
                                 normalize=True)
    plt.figure(1, figsize=(10, 10))

    # plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle='--', label='perfectly_calibrated')
    # plot model reliability

    plt.plot(mpv, fop, marker='.', label=f'model classifier')
    plt.xlabel('Mean predicted probability', size=24)
    plt.ylabel('Fraction of positives', size=24)
    plt.title(model_name + ' calibration curve', size=24)
    plt.tick_params(labelsize=16)
    plt.legend(fontsize=16, loc=(1.04, 0))
    if save_fig:
        plt.savefig(f'{path_file}_calibration_curve.png', bbox_inches='tight')
    plt.show()
    return None
