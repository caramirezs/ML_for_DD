def model_cv(params_dict, df_train, fp_name, verbose=False):
    """
    Optimiza el modelo realizando una validacion cruzada y mejorando el número de estimadores

    :param params_dict: diccionario de parámetros del modelo
    :param df_train: dataser con los datos de entrenamiento
    :param fp_name: fingerprint name
    :return: XGBoost model optimized
    """

    # XGBoost library
    import xgboost as xgb

    seed = 142857

    X_train, y_train = df_train[fp_name].to_list(), df_train['activity']
    xgtrain = xgb.DMatrix(X_train, y_train)

    xgb_clf = xgb.XGBClassifier(**params_dict)
    metrics_xgb_clf = ['error', 'auc']
    cvresult = xgb.cv(params_dict, xgtrain, nfold=5, num_boost_round=5000, metrics=metrics_xgb_clf,
                      early_stopping_rounds=11, stratified=True, seed=seed, verbose_eval=False)
    old_n_estimators = xgb_clf.get_params()['n_estimators']
    new_n_estimators = cvresult.shape[0]
    xgb_clf.set_params(n_estimators=new_n_estimators)
    if verbose:
        print('Model optimized. n_estimators changed: {} to {}'.format(old_n_estimators, new_n_estimators))
    return xgb_clf
