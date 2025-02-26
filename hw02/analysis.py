import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import argparse

def block_avg(data):
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



def main():

    parser = argparse.ArgumentParser(description="Analysis")
    
    parser.add_argument('-g', '--gridsize', type=int, help='gridsize', default=16)
    parser.add_argument('-b', '--external', type=float, help='external field', default=0.0)
    parser.add_argument('-f', '--folder', type=str, default='data')

    args = parser.parse_args()
    N = int(args.gridsize ** 2)
    B = args.external

    folder = pathlib.Path(args.folder).resolve()

    outname = 'binder'
    pathlib.Path(f'{outname}').mkdir(exist_ok=True)

    dict_final = {'J': [], 'M': [], 'X': [], 'E': [], 'M2': [], 'M4': [], 'dM': [], 'dE': [], 'dE2': [], 'dM2': [], 'dM4': [], 'time_correlation_mag': [], 'dX': [], 'h': []}

    for file in folder.iterdir():
        print(file)
        df_local = pd.read_csv(file)

        J = df_local['J'][0]

        average_mag = df_local['M'].mean()
        average_energy = (df_local['E']).mean()
        average_energy_squared = average_energy**2
        average_energy2 = (df_local['E2']).mean()

        X = (average_energy2 - average_energy_squared)

        dict_final['J'].append(J)
        dict_final['M'].append(average_mag)
        dict_final['E'].append(average_energy)
        dict_final['X'].append(X)

        average_m2 = (df_local['M2']).mean()
        average_m4 = (df_local['M4']).mean()

        dict_final['M2'].append(average_m2)
        dict_final['M4'].append(average_m4)

        data = df_local['M'].values
        block_counts, errors, error_est = block_avg(data)
        dict_final['dM'].append(error_est)

        correlated_error_mag = np.std(data, ddof=1) / np.sqrt(N)
        time_correlation_mag = 0.5 * (error_est/correlated_error_mag)**2
        dict_final['time_correlation_mag'].append(time_correlation_mag)

        data = df_local['M2'].values
        block_counts, errors, error_est = block_avg(data)
        dict_final['dM2'].append(error_est)

        data = df_local['M4'].values
        block_counts, errors, error_est = block_avg(data)
        dict_final['dM4'].append(error_est)

        data = df_local['E'].values
        block_counts, errors, error_est = block_avg(data)
        dict_final['dE'].append(error_est)
        data = df_local['E2'].values
        block_counts, errors, error_est = block_avg(data)
        dict_final['dE2'].append(error_est)

        data = df_local['E2'].values - df_local['E'].values**2
        block_counts, errors, error_est = block_avg(data)
        dict_final['dX'].append(error_est)

        dict_final['h'].append(args.external)

    df = pd.DataFrame.from_dict(dict_final)
    df = df.sort_values(by='J')
    df.to_csv(f'{outname}/{args.gridsize}x{args.gridsize}_{args.external}.csv')

if __name__ == '__main__':
    main()
