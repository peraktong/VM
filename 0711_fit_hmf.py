import emcee
import numpy as np
from hmf import sample
from hmf import cosmo
import matplotlib.pyplot as plt
import matplotlib
import math

def log10(x):

    return math.log10(x)

log10= np.vectorize(log10)



path = "HMF_8_to_16/"+"all_plots/"+"mVector_PLANCK-SMT .txt"

fusion = np.loadtxt(path)


# print(fusion.shape)

"""
# [1] m:            [M_sun/h] 
# [2] sigma 
# [3] ln(1/sigma) 
# [4] n_eff 
# [5] f(sigma) 
# [6] dn/dm:        [h^4/(Mpc^3*M_sun)] 
# [7] dn/dlnm:      [h^3/Mpc^3] 
# [8] dn/dlog10m:   [h^3/Mpc^3] 
# [9] n(>m):        [h^3/Mpc^3] 
# [11] rho(>m):     [M_sun*h^2/Mpc^3] 
# [11] rho(<m):     [M_sun*h^2/Mpc^3] 
# [12] Lbox(N=1):   [Mpc/h]

"""

m = fusion[:,0]


logm = log10(m)

dndlog = np.array(fusion[:,7]).ravel()
n = np.array(fusion[:,8]).ravel()

logn = log10(n)


z = np.polyfit(logm,logn, 10)

p = np.poly1d(z)

target = np.arange(11.05,14.15,0.1)

# use linear interpolation

n_interpolate = np.interp(target,xp=logm,fp=n)

"""
plt.plot(target,n_interpolate,"ro",alpha=0.3)
plt.plot(logm,n,"kx",alpha=0.3)
plt.show()
"""
print(len(n_interpolate))
print(target)