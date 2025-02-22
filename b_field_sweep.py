from metropolis import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt





if __name__ == '__main__':

    gridsize = 32
    beta = 1.
    J = 0.75

    mags = []
    mags_errors = []
    field = []

    # Generating intial lattice
    lattice = np.zeros((gridsize, gridsize))
    with np.nditer(lattice, op_flags=['readwrite']) as it:
        for x in it:
            x[...] = random.choice([-1,1])
    
    for step in range(500):
        lattice = mc_sweep(lattice, beta, J, B=-1., mu=1)

    for B in np.arange(-1.0,1.,0.15):
        
        m_current = []

        for step in range(10):
            lattice = mc_sweep(lattice, beta, J, B, mu=1)
        # lattice = CheckAcceptance(lattice, beta, J, B, mu=1)

    # Production run
        for step in range(30):
            lattice = mc_sweep(lattice, beta, J, B, mu=1)
            M, E, E2, M2, M4 = calc_quantities(lattice, J, B)
            m_current.append(M)

        field.append(B)
        mags.append(np.mean(m_current))
        mags_errors.append(np.std(m_current, ddof=1) / np.sqrt(len(m_current)-1))

    for B in np.flip(np.arange(-1.0,1.,0.15)):
        
        m_current = []

        for step in range(10):
            lattice = mc_sweep(lattice, beta, J, B, mu=1)
        # lattice = CheckAcceptance(lattice, beta, J, B, mu=1)

    # Production run
        for step in range(30):
            lattice = mc_sweep(lattice, beta, J, B, mu=1)
            M, E, E2, M2, M4 = calc_quantities(lattice, J, B)
            m_current.append(M)
    

        field.append(B)
        mags.append(np.mean(m_current))
        mags_errors.append(np.std(m_current, ddof=1) / np.sqrt(len(m_current)-1))

    plt.errorbar(field, mags,yerr=mags_errors, marker='o', markersize=5, ls='-', lw=1, capsize=10)
    plt.show()