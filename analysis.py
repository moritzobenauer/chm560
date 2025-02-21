import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import argparse



def main():

    parser = argparse.ArgumentParser(description="Analysis")
    
    parser.add_argument('-g', '--gridsize', type=int, help='gridsize', default=16)
    parser.add_argument('-b', '--external', type=float, help='external field', default=0.0)
    parser.add_argument('-f', '--folder', type=str, default='data')

    args = parser.parse_args()
    N = int(args.gridsize ** 2)
    B = args.external

    folder = pathlib.Path(args.folder).resolve()

    dict_final = {'J': [], 'M': [], 'X': [], 'E': []}

    for file in folder.iterdir():
        df_local = pd.read_csv(file)

        J = df_local['J'][0]

        average_mag = np.abs(df_local['M'].mean())
        average_energy = (df_local['E']).mean()
        average_energy_squared = average_energy**2
        average_energy2 = (df_local['E2']).mean()

        X = (average_energy2 - average_energy_squared)

        dict_final['J'].append(J)
        dict_final['M'].append(average_mag)
        dict_final['E'].append(average_energy)
        dict_final['X'].append(X)


    df = pd.DataFrame.from_dict(dict_final)
    df = df.sort_values(by='J')
    df.to_csv(f'out_{args.gridsize}x{args.gridsize}.csv')

if __name__ == '__main__':
    main()
