from lib.main_func_p3 import calculate_onefp
import numpy as np
import pandas as pd
import os, json

#XGBoost library
import xgboost as xgb

def test_concepto(lista_smiles):
    # Parametros
    seed = 142854
    fp_name = 'morgan2_c'

    #####################################
    # convertir en un datarframe
    df = pd.DataFrame(lista_smiles, columns=['smiles'])

    # Clacular finger print
    calculate_onefp(df, fp_name)
    df.drop_duplicates(subset=['smiles'], inplace=True)
    df.reset_index(inplace=True, drop=True)
    df = df.drop(['mol'], axis=1)

    uniprot_id_list = os.listdir('./models')
    for name in uniprot_id_list:

        # proteina (uniprot_ID)
        uniprot_id = name.split('_')[0]

        # Instanciar el modelo
        xgbc_model = xgb.XGBClassifier()
        # Cargar el modelo
        xgbc_model.load_model(f'./models/{uniprot_id}_model.ubj')

        # Calcular probabilidades del conjunto de datos externo
        X = df[fp_name].tolist()

        # Calcular probabilidades del conjunto de datos interno
        prediction_prob_test = np.array(xgbc_model.predict_proba(X)[:,1])
        df[f'{uniprot_id}_prediction_prob'] = prediction_prob_test

        pred_test = xgbc_model.predict(X)
        df[f'{uniprot_id}_prediction'] = pred_test

    df = df.drop([fp_name], axis=1)
    result = df.to_json(orient="table")
    parsed = json.loads(result)
    return parsed['data']

