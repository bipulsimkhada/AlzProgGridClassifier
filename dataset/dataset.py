import pandas as pd
import numpy as np
from pathlib import Path

regions = ['Ventricles/ICV', 'Hippocampus/ICV', 'WholeBrain/ICV', 'Entorhinal/ICV', 'Fusiform/ICV', 'MidTemp/ICV',
           'FDG', 'AV45',
           'CDRSB', 'MMSE', 'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting','RAVLT_perc_forgetting','FAQ', 'MOCA', 'LDELTOTAL', 'DIGITSCOR','TRABSCOR',
           'ABETA', 'PTAU', 'TAU', 'AGE', 'PTEDUCAT', 'APOE4', 'PTGENDER']
targets = ['DX', 'DX6', 'DX12', 'DX24']

DATASET_DIR = Path(__file__).resolve().parent

def create_dataset():
    file_path = DATASET_DIR / "dataset_with_outlier_removed.csv"
    df = pd.read_csv(file_path, low_memory=False, index_col="ID")
    df['PTGENDER'] = df['PTGENDER'].map({
        'Male': 0,
        'Female': 1
    })

    X = df[regions]
    y = df[targets].copy()
    groups = df['RID'].to_numpy()

    y["stable"] = (y.nunique(axis=1) == 1).astype(int)

    return X, y, groups

def createOutputLabels(Labels):
    Labels = Labels[targets]
    
    output = np.zeros((len(Labels), 4, 3))
    
    for i in range(len(Labels)):
        for j in range(4):
            if Labels.iloc[i, j] == 'CN':
                output[i, j, 0] = 1
            elif Labels.iloc[i, j] == 'MCI':
                output[i, j, 1] = 1
            elif Labels.iloc[i, j] == 'AD':
                output[i, j, 2] = 1
                
    return output

