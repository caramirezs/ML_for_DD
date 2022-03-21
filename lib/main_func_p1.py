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


def convert_to_nM(unit, bioactivity):
    """
    Conversor de unidades estandar
    :param unit: unidad estandar
    :param bioactivity: valor estandar
    :return: valor convertido a nM
    """
    if unit != "nM":
        if unit == "10^-12M" or unit == "10'-12M" or unit == "pM":
            value = float(bioactivity) * 1e-3
        elif unit == "10^-11M" or unit == "10'-11M" or unit == "10pM":
            value = float(bioactivity) * 1e-2
        elif unit == "10^-10M" or unit == "10'-10M" or unit == "10'-4microM" or unit == "10^-4microM" or unit == "10'2pM":
            value = float(bioactivity) * 1e-1
        elif unit == "10^-9M" or unit == "10'-9M" or unit == "10'-3microM" or unit == "10^-3microM" or unit == "10'3pM":
            value = float(bioactivity)
        elif unit == "10^-8M" or unit == "10'-8M" or unit == "10'-2microM" or unit == "10^-2microM" or unit == "10'4pM":
            value = float(bioactivity) * 1e1
        elif unit == "10^-7M" or unit == "10'-7M" or unit == "10'-1microM" or unit == "10^-1microM" or unit == "10'5pM":
            value = float(bioactivity) * 1e2
        elif unit == "10^-6M" or unit == "10'-6M" or unit == "uM" or unit == "/uM" or unit == "10'-6M":
            value = float(bioactivity) * 1e3
        elif unit == "microM" or unit == "µM" or unit == "10'6pM":
            value = float(bioactivity) * 1e3
        elif unit == "10^-5M" or unit == "10'-5M" or unit == "10'7pM" or unit == "10'1 uM":
            value = float(bioactivity) * 1e4
        elif unit == "10^-4M" or unit == "10'-4M" or unit == "10'2 uM":
            value = float(bioactivity) * 1e5
        elif unit == "10^-3M" or unit == "10'-3M" or unit == "mM":
            value = float(bioactivity) * 1e6
        elif unit == "10^-2M" or unit == "10'-2M":
            value = float(bioactivity) * 1e7
        elif unit == "10^-1M" or unit == "10'-1M":
            value = float(bioactivity) * 1e8
        elif unit == "M":
            value = float(bioactivity) * 1e9
        elif unit == "10^8M":
            value = float(bioactivity) * 1e9 * 1e8
        else:
            print('unit not recognized...', unit)
            value = 1e10
        return value
    else:
        return float(bioactivity)


def activity_threshold_type(df, group):
    import pandas as pd
    """
    Agrega columna de actividad conociendo el grupo
    :param df: data frame de la proteina
    :param group: tipo de grupo de la proteina
    :return: None - agrega una nueva columna
    """

    activity_thresholds = {'kinase': [30, 300], 'GPCR': [100, 1000], 'Nuclear Receptor': [100, 1000],
                           'Ion Channel': [10000, 100000], 'Non-IDG Family Targets': [1000, 10000]}
    bottom, upper = activity_thresholds[group]

    df.dropna(inplace=True)
    # Add column for target
    df['activity_type'] = "intermediate"

    # active / inactive
    df_equal = df[df['relation'] == '=']
    df_equal.loc[df_equal[df_equal.value_nM <= bottom].index, 'activity_type'] = "active"
    df_equal.loc[df_equal[df_equal.value_nM >= upper].index, 'activity_type'] = "inactive"
    df_not_equal = df[df['relation'] == '>']
    df_not_equal.loc[df_not_equal[df_not_equal.value_nM >= upper].index, 'activity_type'] = "inactive"
    df = pd.concat([df_equal, df_not_equal])

    print(f'> Columna -activity_type- creada\n'
          f'>> # de compuestos: {len(df)}, [{dict(df.activity_type.value_counts())}]')
    return df


def get_smiles(df):
    import csv
    import pandas as pd
    from chembl_webresource_client.new_client import new_client
    compounds = new_client.molecule

    cmpd_id_list = list(df['molecule_chembl_id'])
    print('> Recuperando SMILES de CHEMBL (esto puede tardar unos minutos)')
    compound_list = compounds.filter(molecule_chembl_id__in=cmpd_id_list).only('molecule_chembl_id',
                                                                               'molecule_structures')
    compound_df = pd.DataFrame(compound_list)
    compound_df = compound_df.drop_duplicates('molecule_chembl_id', keep='first')
    for i, cmpd in compound_df.iterrows():
        if compound_df.loc[i]['molecule_structures'] is not None:
            compound_df.loc[i]['molecule_structures'] = cmpd['molecule_structures']['canonical_smiles']
    output_df = pd.merge(df[['molecule_chembl_id', 'relation', 'value', 'activity_type']], compound_df,
                         on='molecule_chembl_id')
    output_df = output_df.rename(columns={'molecule_structures': 'smiles'})
    old_shape = output_df.shape
    output_df = output_df[~output_df['smiles'].isnull()]
    print(
        f'>>> Eliminando compuestos sin SMILES. # compuestos: {output_df.shape[0]}, {old_shape[0] - output_df.shape[0]} droped')
    output_df = output_df[['molecule_chembl_id', 'smiles', 'relation', 'value', 'activity_type']]
    output_df.reset_index(drop=True, inplace=True)
    return output_df


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


def get_ligands(uniprot_data, verbose=True):
    """
    :param uniprot_data: [uniprot_id, group]
    :return:
    """
    # Import dependences
    from math import isnan
    import pandas as pd
    import numpy as np
    from chembl_webresource_client.new_client import new_client

    # Create resource objects for API access
    targets = new_client.target
    bioactivities = new_client.activity

    uniprot_id = uniprot_data[0]
    group = uniprot_data[1]

    path_file = path(uniprot_id)

    """---------------------------------------------------------------------------------------------------------------------------
    Parte 1. Identificar el id de la proteina en CHEMBL
    ---------------------------------------------------------------------------------------------------------------------------"""
    # Get target information from ChEMBL but restrict to specified values only
    print('Proteina: ', uniprot_id)
    target_id = targets.get(target_components__accession=uniprot_id).only('target_chembl_id', 'organism', 'pref_name',
                                                                          'target_type')
    target = target_id[0]

    # proteina no encontrada en CHEMBL
    if target_id[0] is None:
        print(F'>>> ERROR: {uniprot_id} no encontrada en CHEMBL')
        print('proceso terminado')
        return 'NOFILE'

    print('> Proteina encontrada en CHEMBL, parametros:')
    print(target)
    chembl_id = target['target_chembl_id']
    print(f'>> target_chembl_id: {chembl_id}')
    print('------------------------------------------------------------------')

    """-----------------------------------------------------------------------------------------------------------------
    Parte 2a. Descargar todos los tipos de bioactividad
    Filtro inicial: {target_chembl_id = chembl_id, target_organism = 'Homo sapiens'}
    -----------------------------------------------------------------------------------------------------------------"""
    print('> Recuperando información de CHEMBL (esto puede tardar unos minutos)')
    bioact = list(bioactivities.filter(target_chembl_id=chembl_id).filter(target_organism='Homo sapiens')
                  .only('molecule_chembl_id', 'type', 'relation', 'value', 'units', 'pchembl_value'))
    bioact_df = pd.DataFrame(bioact)
    # Save initial dataframe
    bioact_df.to_csv(f'{path_file}_00_bioactivity.csv', index=False)
    print(f'>> Bioactividad descargada. Filtros: [target_chembl_id = {chembl_id}, target_organism = Homo sapiens]')
    bioact_df = bioact_df[['molecule_chembl_id', 'type', 'relation', 'value', 'units']]
    # Convertir la columna 'value' en floats
    bioact_df['value'] = bioact_df['value'].astype(float)

    print(f'>> número de registros: {bioact_df.shape[0]}, atributos: {bioact_df.shape[1]}')
    print('------------------------------------------------------------------')

    """-----------------------------------------------------------------------------------------------------------------
    Parte 2b. Filtro por tipo de actividad (IC50, Ki, EC50)
    ---------------------------------------------------------------------------------------------------------------- """
    IC50_df = bioact_df[bioact_df['type'] == 'IC50'].reset_index(drop=True)
    Ki_df = bioact_df[bioact_df['type'] == 'Ki'].reset_index(drop=True)
    EC50_df = bioact_df[bioact_df['type'] == 'EC50'].reset_index(drop=True)
    bioact_df = pd.concat([IC50_df, Ki_df, EC50_df]).reset_index(drop=True)
    """------------------------------------------------------------------------------------------------------------- """
    # Drop actividad cuya unidad de medida es diferente a *Molar
    bioact_df.dropna(subset=['units'], inplace=True)
    bioact_df = bioact_df.drop(bioact_df.index[~bioact_df.units.str.contains('M')])
    bioact_df = bioact_df.drop(bioact_df.index[bioact_df.units.str.contains('/')])
    """--------------------------------------------------------------------------------------------------------------"""
    print('> Limpiando / convirtiendo valores a nM')
    print(f'>>> Tipo de actividad filtrada: {bioact_df.type.unique()}')
    print(f'>>> # de compuestos: {bioact_df.shape[0]}')
    print(f'>>>>>> Resumen por tipo de actividad')
    print(bioact_df.type.value_counts())
    print('------------------------------------------------------------------')

    # Convertir a nM
    bioactivity_nM = []
    for i, row in bioact_df.iterrows():
        bioact_nM = convert_to_nM(row['units'], row['value'])
        bioactivity_nM.append(bioact_nM)
    bioact_df['value_nM'] = bioactivity_nM

    # Save intermedia dataframe
    bioact_df.to_csv(f'{path_file}_01_convert_units.csv', index=False)

    """-----------------------------------------------------------------------------------------------------------------
    Parte 3. Detectar ligandos activos e inactivos conociendo el valor de actividad
    Parte 3a. Determinar actividad del ligando con la proteina
              agregar columna activity_type: inactive, intermediate, active
    -----------------------------------------------------------------------------------------------------------------"""
    value_df = activity_threshold_type(bioact_df, group=group)
    # Drop compuestos con actividad intermedia
    value_df.drop(value_df.index[value_df.activity_type == 'intermediate'], inplace=True)
    print(f'>>> Moleculas con tipo de actividad intermedia eliminadas.\n'
          f'>>> # de compuestos: {len(value_df)}')
    print('------------------------------------------------------------------')
    print(value_df.activity_type.value_counts())
    """-----------------------------------------------------------------------------------------------------------------
    Parte 3b. Quitar duplicados
    -----------------------------------------------------------------------------------------------------------------"""
    #  QUEDARSE CON EL menor valor (mayor actividad)
    # todo > REVISAR ESTA PARTE DEL CODIGO
    s = value_df.duplicated(subset=['molecule_chembl_id'], keep=False)
    lista_iloc_duplicated = list(s[s].index)
    print(f'> Eliminado duplicados')
    print(f'>> # entradas duplicadas encontradas: {len(lista_iloc_duplicated)}')
    duplicated_df = value_df.loc[lista_iloc_duplicated].sort_values(by=['molecule_chembl_id', 'value'], ascending=True)
    s = duplicated_df.duplicated(subset=['molecule_chembl_id'], keep='first')
    lista_iloc_duplicated = list(s[s].index)
    value_df = value_df.drop(index=lista_iloc_duplicated)
    value_df.reset_index(drop=True, inplace=True)
    print(f'>>> # de compuestos: {len(value_df)}')
    print('------------------------------------------------------------------')
    """-----------------------------------------------------------------------------------------------------------------
    Parte 4. SIMLES
    -----------------------------------------------------------------------------------------------------------------"""
    smiles_df = get_smiles(value_df)
    smiles_df.rename(columns={'value': 'value_nM'}, inplace=True)
    # save document
    smiles_df.to_csv(f'{path_file}_01_ligands_smiles.csv', index=False)
    print(f'>>> SAVED: {uniprot_id}_01_ligands_smiles.csv')
    print('------------------------------------------------------------------')
    """-----------------------------------------------------------------------------------------------------------------
    Parte 6. Lipinski Ro5
    -----------------------------------------------------------------------------------------------------------------"""
    ro5_df = get_lipinski_Ro5(smiles_df)
    ro5_df = ro5_df[['molecule_chembl_id', 'smiles', 'MW', 'HBA', 'HBD', 'LogP', 'activity_type']]
    ro5_df.reset_index(drop=True, inplace=True)
    ro5_df.to_csv(f'{path_file}_02_ligands_smiles_ADME_lipinski.csv', index=False)
    print(f'>>> SAVED: {uniprot_id}_02_ligands_smiles_ADME_lipinski.csv')
    print('------------------------------------------------------------------')
    """-----------------------------------------------------------------------------------------------------------------
    Parte 6. Guardar archivo final: ['molecule_chembl_id', 'smiles', 'activity_type']
    -----------------------------------------------------------------------------------------------------------------"""
    activity_df = ro5_df[['molecule_chembl_id', 'smiles', 'activity_type']]
    print(f'>>> SAVED: {uniprot_id}_03_ligands_simles_activity_type.csv')
    activity_df.to_csv(f'{path_file}_03_ligands_smiles_activity_type.csv', index=False)
    print(f'>>>>>> Resumen: total({len(activity_df)})')
    print(activity_df.activity_type.value_counts())
    print('------------------------------------------------------------------')
    return activity_df
