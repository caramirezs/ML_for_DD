def path(uniprot_id):
    """
    :param uniprot_id: identificador de la proteina
    :return: path de la carpeta de la proteina
    """
    import os
    dirName = f'./data/{uniprot_id}'
    try:
        os.mkdir(dirName)
    except FileExistsError:
        pass
    return f'./data/{uniprot_id}/{uniprot_id}'


def df_rule_of_five(df):
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    import pandas as pd

    smi = df['smiles']
    m = Chem.MolFromSmiles(smi)

    # Calculate rule of five chemical properties
    MW = Descriptors.ExactMolWt(m)
    HBA = Descriptors.NumHAcceptors(m)
    HBD = Descriptors.NumHDonors(m)
    LogP = Descriptors.MolLogP(m)

    # Rule of five conditions
    conditions = [MW <= 500, HBA <= 10, HBD <= 5, LogP <= 5]

    # Create pandas row for conditions results with values and information whether rule of five is violated
    return pd.Series([MW, HBA, HBD, LogP, 'yes']) if conditions.count(True) >= 3 else pd.Series(
        [MW, HBA, HBD, LogP, 'no'])


def get_lipinski_Ro5(df):
    print('> Recuperando ADME y aplicando la regla de los 5')
    rule5_prop_df = df.apply(df_rule_of_five, axis=1)
    rule5_prop_df.columns = ['MW', 'HBA', 'HBD', 'LogP', 'rule_of_five_conform']
    df = df.join(rule5_prop_df)
    filtered_df = df[df['rule_of_five_conform'] == 'yes']
    # Info about data
    print('>> # compuestos en data set: ', len(df))
    print(">> # compuestos que cumplen Lipinski's rule of five:", len(filtered_df))
    print(">> # compuetos que NO cumplen Lipinski's rule of five:", (len(df) - len(filtered_df)))
    print(">>> Filtrando compuestos que cumplen Lipinski's rule of five")
    return filtered_df


def get_info_target(uniprot_data, verbose=True):
    """
    :param uniprot_data: [uniprot_id, group]
    :return:
    """
    # Import dependences
    import pandas as pd
    uniprot_id = uniprot_data[0]
    path_file = path(uniprot_id)

    """-----------------------------------------------------------------------------------------------------------------
    Parte 1. Cargar los datos
    -----------------------------------------------------------------------------------------------------------------"""
    try:
        activity_df = pd.read_csv(f'data/activity_type_per_target/{uniprot_id}.csv')
    except FileNotFoundError:
        print(f'Actividad del target {uniprot_id} no encontrada, proceso finalizado')
        return None
    activity_df = activity_df.rename(columns={'canonical_smiles (Canonical)': 'smiles', 'Activity_type': 'activity_type'})
    smiles_df = activity_df[['chembl_id_ligand', 'smiles', 'activity_type']]
    smiles_df.to_csv(f'{path_file}_01_ligands_smiles.csv', index=False)

    """-----------------------------------------------------------------------------------------------------------------
    Parte 2. Lipinski Ro5
    -----------------------------------------------------------------------------------------------------------------"""
    ro5_df = get_lipinski_Ro5(smiles_df)
    ro5_df = ro5_df[['chembl_id_ligand', 'smiles', 'MW', 'HBA', 'HBD', 'LogP', 'activity_type']]
    ro5_df.reset_index(drop=True, inplace=True)
    ro5_df.to_csv(f'{path_file}_02_ligands_smiles_ADME_lipinski.csv', index=False)
    print(f'>>> SAVED: {uniprot_id}_02_ligands_smiles_ADME_lipinski.csv')
    print('------------------------------------------------------------------')
    """-----------------------------------------------------------------------------------------------------------------
    Parte 3. Guardar archivo final: ['chembl_id_ligand', 'smiles', 'activity_type']
    -----------------------------------------------------------------------------------------------------------------"""
    activity_df = ro5_df[['chembl_id_ligand', 'smiles', 'activity_type']]
    print(f'>>> SAVED: {uniprot_id}_03_ligands_simles_activity_type.csv')
    activity_df.to_csv(f'{path_file}_03_ligands_smiles_activity_type.csv', index=False)
    print(f'>>>>>> Resumen: total({len(activity_df)})')
    print(activity_df.activity_type.value_counts())
    print('------------------------------------------------------------------')
    return activity_df
