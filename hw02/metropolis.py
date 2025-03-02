#MLO @ Princeton 2024

import numpy as np
from numpy.random import rand
import random
from scipy.stats import norm
import pandas as pd
import argparse
from numba import jit
import time
import pathlib

@jit(nopython=True)
def GetNNL(lattice: np.array, i: int, j: int):
    ''' 
    Returns nearest neighbors (version 1.1 with easier boundary conditions)
        nnl: nearest neighbor list
        nns: nearest neighbor states
    '''
    N = np.shape(lattice)[0]
    
    nnl = [[(i+1)%N, j], [(i-1)%N, j], [i, (j+1)%N], [i, (j-1)%N]]

    nns = [lattice[index[0], index[1]] for index in nnl]  
    return nnl, np.array(nns)

@jit(nopython=True)
def CaclulateMagnetization(lattice: np.array):
    """Calculate the magnetization of the entire lattice."""
    N = lattice.shape[0] ** 2
    return np.sum(lattice)

@jit(nopython=True)
def CalculateEnergy(lattice: np.array, J:float, h: float, M: float):
    energy = 0.0
    # M = CaclulateMagnetization(lattice)
    for x in range(lattice.shape[0]):
        for y in range(lattice.shape[1]):
            #print(x,y)
            s = lattice[x,y]
            nnl, nns = GetNNL(lattice, x, y)
            energy += -J * s* np.sum(nns)
    N = lattice.shape[0] ** 2
    return ((energy / 2.0) - h*M)


@jit(nopython=True)
def calc_quantities(lattice: np.array, J:float, h: float) -> list[float, float, float]:
    M = CaclulateMagnetization(lattice)
    energy = CalculateEnergy(lattice, J, h, M)
    energy_squared = energy**2
    msquared = M**2
    mpower4 = M**4
    

    return [M, energy, energy_squared, msquared, mpower4]


@jit(nopython=True)
def mc_sweep(lattice: np.array, beta: float, J: float, B:float, mu:float=1):
    N = lattice.shape[0] ** 2
    for i in range(N):
        lattice = CheckAcceptance(lattice, beta, J, B, mu=1)
    return lattice

@jit(nopython=True)
def CheckAcceptance(lattice: np.array, beta: float, J: float, B:float, mu:float=1):

    # eold = CalculateEnergy(lattice, J)

    dim_i, dim_j = np.shape(lattice)
    i = np.random.randint(dim_i)
    j = np.random.randint(dim_j)

    nnl, nb = GetNNL(lattice, i, j)
    k = lattice[i,j]
    magnetic_contribution = 2 * mu * B * lattice[i, j]
    dE = 2 * k * np.sum(nb)*J + magnetic_contribution

    rand = np.random.rand()

    if dE <= 0 or np.random.rand() <= np.exp(-dE * beta):
        lattice[i,j] *= -1
    else:
         pass
    
    return lattice


def main():

    parser = argparse.ArgumentParser(description="2DIsingModel2024-MLO")
    
    parser.add_argument('-g', '--gridsize', type=int, help='gridsize', default=16)
    parser.add_argument('-b', '--external', type=float, help='external field', default=0.0)
    parser.add_argument('-j', '--coupling', type=float, help='coupling constant', default=1)
    parser.add_argument('-f', '--file', type=str, default='data')

    parser.add_argument('--eq', type=int, help='eq time', default=5000)
    parser.add_argument('--prod', type=int, help='eq time', default=5000) 

    args = parser.parse_args()


    gridsize = args.gridsize # lattice dimensions
    J = args.coupling
    B = args.external # B=0 --> no external field

    current_dir = pathlib.Path.cwd()
    subfolder_path = current_dir / f'{args.file}_{gridsize}x{gridsize}'
    subfolder_path.mkdir(parents=True, exist_ok=True)

    filename = f'out_{gridsize}x{gridsize}_h_{B}_J_{J}.txt'

    filename = subfolder_path / filename



    MCTIME = gridsize ** 2

    eq_time = args.eq
    simulation_time = args.prod

    beta = 1.


    lattice = np.zeros((gridsize, gridsize))
    print(lattice.shape)
    print(f'{J}')
    print(f'{B}')

    # Generates random grid
    
    with np.nditer(lattice, op_flags=['readwrite']) as it:
        for x in it:
            x[...] = random.choice([-1,1])
     
    # Sampling Interval for Production Run

    interval = 10

    output_dic = {'T': [], 'M': [], 'E': [], 'E2': [], 'X': [], 'J': [], 'B': [], 'M2': [], 'M4': []}

    # EQ run
    for step in range(eq_time):
        lattice = mc_sweep(lattice, beta, J, B, mu=1)
        # lattice = CheckAcceptance(lattice, beta, J, B, mu=1)

        if step % interval == 0:
            print(f'EQ step {step}/{eq_time}')
    
    # Production run
    for step in range(simulation_time):
        lattice = mc_sweep(lattice, beta, J, B, mu=1)
        # lattice = CheckAcceptance(lattice, beta, J, B, mu=1)

        if step % interval == 0:
            print(step)
            M, E, E2, M2, M4 = calc_quantities(lattice, J, B)
            output_dic['T'].append(step)
            output_dic['J'].append(J)
            output_dic['B'].append(B)
            output_dic['X'].append(gridsize)
            output_dic['M'].append(M)
            output_dic['E'].append(E)
            output_dic['E2'].append(E2)
            output_dic['M2'].append(M2)
            output_dic['M4'].append(M4)
        
    print(np.average(output_dic['E'])**2)
    print(np.average(output_dic['E2']))

    df = pd.DataFrame.from_dict(output_dic)
    df.to_csv(filename)

if __name__ == "__main__":
    main()