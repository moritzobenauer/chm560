import numpy as np


def bma(RUNS=1, N=24):
    u = []
    for _ in range(N):
        random_variable = np.random.random()
        u.append(-np.log(random_variable))
    r = [np.sqrt(2*x) for x in u]


    theta = []
    for _ in range(N):
        random_variable = np.random.random()
        theta.append(2*np.pi*(random_variable))

    r = np.array(r)
    theta = np.array(theta)


    return r*np.cos(theta)[0]
