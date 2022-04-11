from importlib import reload
from lib.old_main_func_p1 import path

import pandas as pd
import numpy as np

# uniprot_data = ['P49841', 'kinase']
uniprot_data = ['P22303', 'Non-IDG Family Targets']
uniprot_id = uniprot_data[0]
path_file = path(uniprot_id)

import lib.old_main_func_p1
reload(lib.old_main_func_p1)
from lib.old_main_func_p1 import get_ligands

activity_df = get_ligands(uniprot_data)


import lib.main_func_p3
reload(lib.main_func_p3)
from lib.main_func_p3 import export_train_set_pickle
import pandas as pd

path_file = path(uniprot_id)
activity_df = pd.read_csv(f'{path_file}_03_ligands_smiles_activity_type.csv')

fp_list = ['maccs', 'morgan2_c', 'morgan3_c', 'topological_torsions_b', 'rdkit5_b', 'avalon_512_b']

# Construct a molecule from a SMILES string
fp_df = export_train_set_pickle(activity_df, fp_list)


fp_df.to_pickle(f'{path_file}_dataset')
print(f'>>> SAVED: {uniprot_id}_dataset')

