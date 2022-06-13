def test_concepto(lista_smiles):
    from lib.main_func_p3 import calculate_onefp
    from lib.main_func_p1 import path
    import numpy as np
    import pandas as pd

    #XGBoost library
    import xgboost as xgb

    #####################################
    # proteina (uniprot_ID)
    uniprot_id = 'P56817'

    path_file = path(uniprot_id)

    # Parametros
    seed = 142854
    fp_name = 'morgan2_c'

    # convertir en un datarframe
    df = pd.DataFrame(lista_smiles, columns=['smiles'])

    # Clacular finger print
    calculate_onefp(df, fp_name)
    df.drop_duplicates(subset=['smiles'], inplace=True)
    df.reset_index(inplace=True, drop=True)
    df = df.drop(['mol'], axis=1)


    # Instanciar el modelo
    xgbc_model = xgb.XGBClassifier()
    # Cargar el modelo
    xgbc_model.load_model(f'./models/{uniprot_id}_model.ubj')

    # Calcular probabilidades del conjunto de datos externo
    X = df[fp_name].tolist()

    # Calcular probabilidades del conjunto de datos interno
    prediction_prob_test = np.array(xgbc_model.predict_proba(X)[:,1])
    df['prediction_prob'] = prediction_prob_test

    pred_test = xgbc_model.predict(X)
    df['prediction'] = pred_test

    df = df[['smiles', 'prediction_prob', 'prediction']]
    return df