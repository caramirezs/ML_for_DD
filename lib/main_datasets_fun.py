from lib.main_func_p1 import path
from lib.main_func_p1 import get_info_target
from lib.main_func_p3 import export_dataset

import pandas as pd


def uniprot_id_datasets(uniprot_id, fp_name='morgan2_c', seed=142857):
    print('---------------------------------------------------------')
    print(f'Proceso iniciado, uniprot_id: {uniprot_id}\n')
    path_file = path(uniprot_id)
    try:
        with open(f'{path_file}_03_ligands_smiles_activity_type.csv') as f:
            print('Ya se cuenta con la información del target. No es necesario hacer este proceso')
            activity_df = pd.read_csv(f'{path_file}_03_ligands_smiles_activity_type.csv')
    except FileNotFoundError:
        activity_df = get_info_target(uniprot_id)

    activity_filtered_df = pd.read_csv(f'{path_file}_03_ligands_smiles_activity_type.csv')
    activity_filtered_df = activity_filtered_df[activity_df['activity_type'] != 'Intermediate']

    fp_list = [fp_name]
    try:
        with open(f'{path_file}_dataset_full') as f:
            print('El archivo ya está en la base. No es necesario hacer este proceso')
            fp_df = pd.read_pickle(f'{path_file}_dataset_full')
    except FileNotFoundError:
        # Construct a molecule from a SMILES string
        fp_df = export_dataset(activity_filtered_df, fp_list, verbose=False)
        fp_df.to_pickle(f'{path_file}_dataset_full')
        print(f'>>> SAVED: {path_file}_dataset_full, compounds: {len(fp_df)}')

    # Train a test set
    test_size = 0.15  # test_size: 15%
    from sklearn.model_selection import train_test_split
    fp_df_train, fp_df_valid = train_test_split(fp_df, test_size=test_size, shuffle=True, stratify=fp_df['activity'],
                                               random_state=seed)
    fp_df_train.reset_index(drop=True, inplace=True)
    fp_df_valid.reset_index(drop=True, inplace=True)
    fp_df_train.to_pickle(f'{path_file}_dataset_train')
    print(f'>>> SAVED: {path_file}_dataset_train, compounds: {len(fp_df_train)}')
    fp_df_valid.to_pickle(f'{path_file}_dataset_valid')
    print(f'>>> SAVED: {path_file}_dataset_valid, compounds: {len(fp_df_valid)}')
    return None
