import numpy as np
import matplotlib.pyplot as plt

for i in range(19):
    fes=np.genfromtxt("fes_" + str(i) + ".dat")
    plt.plot(fes[:,0],fes[:,1],label=str((i+1)*0.25) + " ns")
fes=np.genfromtxt("fes.dat")
plt.plot(fes[:,0],fes[:,1],label="5 ns")
plt.xlabel("Collective variable")
plt.ylabel("Free energy (kJ/mol)")
# plt.xlim([0,216])
plt.legend()
#plt.ylim([0,1500])
plt.savefig('test.png')