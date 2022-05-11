from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import RDKFingerprint
from rdkit.Avalon import pyAvalonTools

# Lista de todas las fingerprint
all_fp_list = ['maccs',
               'morgan0_c', 'morgan1_c', 'morgan2_c', 'morgan3_c', 'morgan0_b', 'morgan1_b', 'morgan2_b', 'morgan3_b',
               'feat_morgan0_c', 'feat_morgan1_c', 'feat_morgan2_c', 'feat_morgan3_c',
               'feat_morgan0_b', 'feat_morgan1_b', 'feat_morgan2_b', 'feat_morgan3_b',
               'rdkit4_b', 'rdkit5_b', 'rdkit6_b', 'rdkit7_b',
               'linear_rdkit4_b', 'linear_rdkit5_b', 'linear_rdkit6_b', 'linear_rdkit7_b',
               'atom_pairs_b', 'topological_torsions_b',
               'avalon_512_b', 'avalon_1024_b', 'avalon_512_c', 'avalon_1024_c']


# Finger prints excluidas porque toman mucho timepo
#  'atom_pairs_c', 'topological_torsions_c


def calculate_fp(mol, method='maccs'):
    # mol = Chem molecule object
    # Function to calculate molecular fingerprints given the number of bits and the method
    if method == 'maccs':
        return rdMolDescriptors.GetMACCSKeysFingerprint(mol)
    if method == 'morgan0_c':
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 0)
    if method == 'morgan1_c':
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 1)
    if method == 'morgan2_c':
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2)  # ecfp4
    if method == 'morgan3_c':
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 3)  # ecfp6
    if method == 'morgan0_b':
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 0, 1024)
    if method == 'morgan1_b':
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 1, 1024)
    if method == 'morgan2_b':
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, 1024)
    if method == 'morgan3_b':
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 3, 1024)
    if method == 'feat_morgan0_c':
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 0, 1024, useFeatures=True)
    if method == 'feat_morgan1_c':
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 1, 1024, useFeatures=True)
    if method == 'feat_morgan2_c':
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, 1024, useFeatures=True)
    if method == 'feat_morgan3_c':
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 3, 1024, useFeatures=True)
    if method == 'feat_morgan0_b':
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 0, 1024, useFeatures=True)
    if method == 'feat_morgan1_b':
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 1, 1024, useFeatures=True)
    if method == 'feat_morgan2_b':
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, 1024, useFeatures=True)
    if method == 'feat_morgan3_b':
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 3, 1024, useFeatures=True)
    if method == 'rdkit4_b':
        return RDKFingerprint(mol, maxPath=4)
    if method == 'rdkit5_b':
        return RDKFingerprint(mol, maxPath=5, fpSize=1024)
    if method == 'rdkit6_b':
        return RDKFingerprint(mol, maxPath=6)
    if method == 'rdkit7_b':
        return RDKFingerprint(mol, maxPath=7)
    if method == 'linear_rdkit4_b':
        return RDKFingerprint(mol, maxPath=4, branchedPaths=False)
    if method == 'linear_rdkit5_b':
        return RDKFingerprint(mol, maxPath=5, branchedPaths=False)
    if method == 'linear_rdkit6_b':
        return RDKFingerprint(mol, maxPath=6, branchedPaths=False)
    if method == 'linear_rdkit7_b':
        return RDKFingerprint(mol, maxPath=7, branchedPaths=False)
    # if method == 'atom_pairs_c':
    #     return rdMolDescriptors.GetAtomPairFingerprint(mol)
    # if method == 'topological_torsions_c':
    #     return rdMolDescriptors.GetTopologicalTorsionFingerprint(mol)
    if method == 'atom_pairs_b':
        return rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol)
    if method == 'topological_torsions_b':
        return rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=2048)  # torsion
    if method == 'avalon_512_b':
        return pyAvalonTools.GetAvalonFP(mol, 512)
    if method == 'avalon_1024_b':
        return pyAvalonTools.GetAvalonFP(mol, 1024)
    if method == 'avalon_512_c':
        return pyAvalonTools.GetAvalonCountFP(mol, 512)
    if method == 'avalon_1024_c':
        return pyAvalonTools.GetAvalonCountFP(mol, 1024)


def calculate_onefp(df, fp_name):
    from rdkit import Chem
    df['mol'] = df.smiles.map(lambda smile: Chem.MolFromSmiles(smile))
    df[fp_name] = df.mol.apply(calculate_fp, args=[fp_name])


def create_mol(df):
    # Construct a molecule from a SMILES string
    # Generate mol column: Returns a Mol object, None on failure.
    df['mol'] = df.smiles.map(lambda smile: Chem.MolFromSmiles(smile))
    return None


def create_fp_bv(df, method):
    df[f'{method}'] = df.mol.map(
        # Apply the lambda function "calculate_fp" for each molecule
        lambda x: list(calculate_fp(x, method=method)))
    return None


def export_dataset(df, fp_list, verbose=False):
    import numpy as np
    if verbose: print('> Construyendo una forma molecular a partir de los SMILES')
    create_mol(df)

    if verbose: print('> Creating fingerprints')
    for fp in fp_list:
        create_fp_bv(df, fp)

    df_final = df.copy()

    if verbose: print('Add column for activity_type')
    df_final['activity'] = np.zeros(len(df_final))

    if verbose: print('Mark every molecule as active (1.0) if target is active')

    df_final.loc[df_final[df_final.activity_type == 'Active'].index, 'activity'] = 1.0
    df_final.drop(['smiles', 'activity_type', 'mol'], axis=1, inplace=True)
    df_final.reset_index(drop=True, inplace=True)

    return df_final
