import numpy as np


def bma(RUNS, N):
    for i in range(RUNS):
        np.random.seed(i)
        xbar = []
        for _ in range(N):
            random_numbers_tmp = []

            # Get xbar from the mean of 12 random uniform numbers
            for _ in range(12):
                random_numbers_tmp.append(np.random.random())
            xbar.append((sum(random_numbers_tmp)-6))

    xbar = np.array(xbar)
    return xbar