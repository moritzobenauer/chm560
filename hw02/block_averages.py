import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib

def block_averages(data: np.array, block_size: int) -> np.array:
    n_blocks = len(data) // block_size
    blocks = np.array_split(data, n_blocks)
    block_averages = np.array([block.mean() for block in blocks])
    return block_averages

def calc_std_error(block_averages: np.array) -> float:
    n_blocks = len(block_averages)
    return np.std(block_averages) / np.sqrt(n_blocks)


def load_data(folder: pathlib.Path) -> pd.DataFrame:
    dict_final = {'J': [], 'M': [], 'X': [], 'E': [], 'M2': [], 'M4': [], 'dM': [], 'dE': [], 'dE2': []}
    for file in folder.iterdir():
        df_local = pd.read_csv(file)
        J = df_local['J'][0]
        average_mag = np.abs(df_local['M'].mean())
        average_energy = (df_local['E']).mean()
        average_energy_squared = average_energy**2
        average_energy2 = (df_local['E2']).mean()
        X = (average_energy2 - average_energy_squared)
        average_m2 = (df_local['M2']).mean()
        average_m4 = (df_local['M4']).mean()
        dict_final['J'].append(J)
        dict_final['M'].append(average_mag)
        dict_final['E'].append(average_energy)
        dict_final['X'].append(X)
        dict_final['M2'].append(average_m2)
        dict_final['M4'].append(average_m4)

        data = df_local['M'].values
        block_counts, errors, error_est = iterative_coarse_graining(data)
        dict_final['dM'].append(error_est)
        data = df_local['E'].values
        block_counts, errors, error_est = iterative_coarse_graining(data)
        dict_final['dE'].append(error_est)
        data = df_local['E2'].values
        block_counts, errors, error_est = iterative_coarse_graining(data)
        dict_final['dE2'].append(error_est)

    df = pd.DataFrame.from_dict(dict_final)
    df = df.sort_values(by='J')
    return df

def iterative_coarse_graining(data):
    current_data = np.array(data)
    errors = []
    block_counts = []
    
    while len(current_data) > 1:
        n_blocks = len(current_data)
        block_counts.append(n_blocks)
        se = np.std(current_data, ddof=1) / np.sqrt(n_blocks-1)
        errors.append(se)
        
        if n_blocks % 2 == 1:
            current_data = current_data[:-1]
        
        current_data = 0.5 * (current_data[0::2] + current_data[1::2])
    
    error_estimate = np.average(errors[0:2])
    
    return block_counts, errors, error_estimate


data = load_data(pathlib.Path('/home/moritz/Projects/chm560/output_16x16'))
plt.errorbar(data['J'], data['E'], yerr=data['dE'], capsize=10, markersize=5, ls='none', marker='o')
data = load_data(pathlib.Path('/home/moritz/Projects/chm560/output_32x32'))
plt.errorbar(data['J'], data['E'], yerr=data['dE'], capsize=10, markersize=5, ls='none', marker='s')
plt.xlabel('J')
plt.ylabel('Magnetization')
plt.show()
