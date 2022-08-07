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
    if old_ratio <= 1:
        old_ratio = 1 / old_ratio
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
    new_dato = [date_time, datos_iniciales[0], datos_iniciales[1], algoritmo, datos_iniciales[2],
                df_results_model.loc['AUC']['Train'], df_results_model.loc['accuracy']['Train'],
                df_results_model.loc['recall_(sens)']['Train'], df_results_model.loc['specificity']['Train'],
                df_results_model.loc['precision']['Train'], df_results_model.loc['f1_score']['Train'],
                df_results_model.loc['confusion_matrix']['Train'],
                df_results_model.loc['AUC']['Test'], df_results_model.loc['accuracy']['Test'],
                df_results_model.loc['recall_(sens)']['Test'], df_results_model.loc['specificity']['Test'],
                df_results_model.loc['precision']['Test'], df_results_model.loc['f1_score']['Test'],
                df_results_model.loc['confusion_matrix']['Test']]
    with open(log, 'a') as file:
        writer = csv.writer(file, lineterminator='\n', delimiter=",")
        writer.writerow(new_dato)
    return None


def model_metrics_score(y_true, y_pred, y_prob_pred):
    auc_score = round(roc_auc_score(y_true, y_prob_pred), 3)
    acc_score = round(accuracy_score(y_true, y_pred), 3)
    sens_score = round(recall_score(y_true, y_pred), 3)
    spec_score = round((acc_score * len(y_true) - sens_score * sum(y_true)) / \
                       (len(y_true) - sum(y_true)), 3)
    prec_score = round(precision_score(y_true, y_pred, zero_division=1), 3)
    f1_s = round(f1_score(y_true, y_pred, average='weighted'), 3)
    conf_m_score = confusion_matrix(y_true, y_pred).tolist()
    fpr, tpr, _ = roc_curve(y_true, y_prob_pred)
    results_ROF = (fpr, tpr, auc_score)
    return auc_score, acc_score, sens_score, spec_score, prec_score, f1_s, conf_m_score, results_ROF


def model_clf(model, fp_name, uniprot_id, seed=142857, save_log=False, verbose=True):
    from lib.main_func_p1 import path

    path_file = path(uniprot_id)
    # Import train/test dataset
    df_set = pd.read_pickle(f'{path_file}_dataset_train')
    ori_compounds_len = len(df_set)
    activity_type = df_set.activity.value_counts().keys()
    activity_count = list(df_set.activity.value_counts())

    ratio = activity_count[0] / activity_count[1]
    if ratio < 1:
        ratio = 1 / ratio

    if verbose:
        print(f'Protein: {uniprot_id}\n'
              f'Total ={ori_compounds_len}\n'
              f' [{activity_type[0]}({activity_count[0]})/{activity_type[1]}({activity_count[1]})] -'
              f' ratio={round(ratio, 2)}')

    # train test split
    X_set, y_set = df_set[fp_name], df_set['activity']
    X_train, X_test, y_train, y_test = train_test_split(X_set, y_set, test_size=0.15,
                                                        random_state=seed, stratify=y_set)
    X_train, X_test = X_train.to_list(), X_test.tolist()

    pred_train = model.predict(X_train)
    pred_prob_train = model.predict_proba(X_train)[:, 1]

    # scores TRAIN/TEST sataset
    auc_score_train, acc_score_train, \
    sens_score_train, spec_score_train, \
    prec_score_train, f1_score_train, \
    confusion_train, results_ROF_train = model_metrics_score(y_train, pred_train, pred_prob_train)

    # scores Validation dataset
    pred_test = model.predict(X_test)
    pred_prob_test = model.predict_proba(X_test)[:, 1]

    auc_score_test, acc_score_test, \
    sens_score_test, spec_score_test, \
    prec_score_test, f1_score_test, \
    confusion_test, results_ROF_test = model_metrics_score(y_test, pred_test, pred_prob_test)

    results_ROC = [results_ROF_train, results_ROF_test]

    list_results = [['AUC', auc_score_train, auc_score_test],
                    ['accuracy', acc_score_train, acc_score_test],
                    ['recall_(sens)', sens_score_train, sens_score_test],
                    ['specificity', spec_score_train, spec_score_test],
                    ['precision', prec_score_train, prec_score_test],
                    ['f1_score', f1_score_train, f1_score_test],
                    ['confusion_matrix', confusion_train, confusion_test]]

    if verbose:
        print('Results %s:' % str(model).split('(')[0],
              '\n-------------------------------------')
        print(classification_report(y_test, pred_test))

    df_results_model = pd.DataFrame(list_results, columns=['Metric', 'Train', 'Test'])
    df_results_model.set_index('Metric', inplace=True)

    # Archivar resultados en el log
    if save_log:
        params_dict = model.get_params()
        nombre_algoritmo = str(type(model))[:-2].split('.')[-1]
        log = f'./log/log.csv'
        datos_iniciales = [uniprot_id, fp_name, params_dict]
        log_model(datos_iniciales, df_results_model, log=log, algoritmo=nombre_algoritmo)

    return model, df_results_model, results_ROC


def modelXGBoost_fit_scores(xgb_clf, fp_name, df_set, df_valid,
                            resample_factor=0, resample_mode='under_sampling', verbose=True):
    from sklearn.model_selection import train_test_split

    # Balanced samplers
    if resample_factor != 0:
        ori_compounds_len = len(df_set)
        df_set_rsmp = resampling_set(df_set, mode=resample_mode, ratio=resample_factor)
        if verbose:
            print(f'{resample_mode} - {resample_factor}: {ori_compounds_len} to {len(df_set_rsmp)}')
    else:
        df_set_rsmp = df_set

    X_set, y_set = df_set_rsmp[fp_name], df_set_rsmp['activity']
    X_train, X_test, y_train, y_test = train_test_split(X_set, y_set, test_size=0.15,
                                                        random_state=142857, stratify=y_set)
    X_train, X_test = X_train.to_list(), X_test.to_list()
    X_valid, y_valid = df_valid[fp_name].to_list(), df_valid['activity']

    xgb_clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # scores train set
    X_set, y_set = df_set[fp_name], df_set['activity']
    pred_train = xgb_clf.predict(X_set.to_list())
    df_set['prediction'] = pred_train
    pred_prob_train = xgb_clf.predict_proba(X_set.to_list())[:, 1]
    df_set['prediction_prob'] = pred_prob_train
    scores_train = model_metrics_score(y_set, pred_train, pred_prob_train)

    # scores validation set
    pred_valid = xgb_clf.predict(X_valid)
    df_valid['prediction'] = pred_valid
    pred_prob_valid = xgb_clf.predict_proba(X_valid)[:, 1]
    df_valid['prediction_prob'] = pred_prob_valid
    scores_valid = model_metrics_score(y_valid, pred_valid, pred_prob_valid)
    return xgb_clf, scores_train, scores_valid


def plot_ROC_curve(metrics_ROC_list, metrics_ROC_name, model_name,
                   path_file=None, name_mod=None, save_fig=False):
    # matplotlib:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    cmap = cm.get_cmap('Blues')
    colors = [cmap(i) for i in np.linspace(0.3, 1.0, len(metrics_ROC_list))]

    plt.figure(1, figsize=(10, 10))

    for i, metrics in enumerate(metrics_ROC_list):
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
        if name_mod is not None:
            plt.savefig(f'{path_file}_IMG04_ROC_curve_{name_mod}.png', bbox_inches='tight')
        else:
            plt.savefig(f'{path_file}_IMG04_ROC_curve.png', bbox_inches='tight')
    plt.show()
    plt.close()
    return None


def plot_calibration_curve(df_list, df_list_name, model_name,
                           path_file=None, name_mod=None, save_fig=False):
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    plt.figure(1, figsize=(10, 10))
    # plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle='--', label='perfectly_calibrated')

    for i, df in enumerate(df_list):
        clf_score = round(brier_score_loss(df['activity'], df['prediction_prob'], pos_label=1), 3)
        fop, mpv = calibration_curve(df['activity'], df['prediction_prob'], n_bins=5)
        name = df_list_name[i]
        # plot model reliability
        plt.plot(mpv, fop, marker='.', label=f'{name}: {clf_score}')

    plt.xlabel('Mean predicted probability', size=24)
    plt.ylabel('Fraction of positives', size=24)
    plt.title(model_name + ' calibration curve', size=24)
    plt.tick_params(labelsize=16)
    plt.legend(fontsize=16, loc=(1.04, 0))
    if save_fig:
        if name_mod is not None:
            plt.savefig(f'{path_file}_IMG05_calibration_curve_{name_mod}.png', bbox_inches='tight')
        else:
            plt.savefig(f'{path_file}_IMG05_calibration_curve.png', bbox_inches='tight')
    plt.show()
    plt.close()
    return None


def brier_score(df_list):
    from sklearn.metrics import brier_score_loss
    clf_score_list = list()
    for i, df in enumerate(df_list):
        clf_score = brier_score_loss(df['activity'], df['prediction_prob'], pos_label=1)
        clf_score_list.append(clf_score)
    return clf_score_list


def plot_probability_curve(df, uniprot_id, hue_order=None,
                           path_file=None, name_mod=None, save_fig=False):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig = plt.figure()
    fig.set_size_inches(15, 15)
    sns.displot(data=df, x='prediction_prob', hue="type", kind="kde", bw_adjust=.1, common_norm=False,
                hue_order=hue_order)

    plt.axvline(x=0.5, ymax=0.95, color='k', linestyle='--')
    plt.axvspan(0, 0.5, ymax=0.95, facecolor='oldlace', alpha=0.8, zorder=-100)
    plt.axvspan(0.5, 1, ymax=0.95, facecolor='lightcyan', alpha=0.8, zorder=-100)

    plt.text(0.08, 0.88 * plt.gca().get_ylim()[1], 'Inactive zone', fontsize=13, fontdict={"weight": "bold"})
    plt.text(0.6, 0.88 * plt.gca().get_ylim()[1], 'Active zone', fontsize=13, fontdict={"weight": "bold"})
    plt.title(f'{uniprot_id}  Active/Inactive probability',
              fontsize=16, fontdict={"weight": "bold"})
    if save_fig:
        if name_mod is not None:
            plt.savefig(f'{path_file}_IMG06_compounds_probability_{name_mod}.png', dpi=150, bbox_inches='tight')
        else:
            plt.savefig(f'{path_file}_IMG06_compounds_probability.png', dpi=150, bbox_inches='tight')
    plt.show()


# convert list of images to a single pdf
def imgs_to_pdf(img_dir, save_dir=None, save_name=None, res=400):
    import os
    from img2pdf import convert
    list_images = list()
    for file in os.listdir(img_dir):
        try:
            checker = file.rsplit('_')[1]
        except IndexError:
            checker = ''
        if checker == 'summary':
            list_images.append(f'{img_dir}/{file}')
    with open(f'{save_dir}/{save_name}', 'wb') as pdf:
        pdf.write(convert(list_images))
    return None


# AutouploadGIT

def git_init():
    import git
    master = 'https://github.com/caramirezs/ML_for_DD'
    repo = git.Repo('.')
    repository.create_remote('master', 'https://github.com/foo/test.git')


def git_push(list_files, server):
    import git
    for file in list_files:
        repo.git.add(file)
    repo.git.commit('-m', f'Archivos actualizados a master desde {server}', author=f'{server}')
