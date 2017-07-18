import numpy as np
from scipy import integrate
import time
from termcolor import colored
import math
import os
from helpers.SimulationAnalysis import SimulationAnalysis, readHlist, iterTrees, getMainBranch
import pickle
import emcee
import scipy.optimize as op
import matplotlib.pyplot as plt


### start:

def log10(x):
    if x > 0:
        return math.log10(x)
    else:
        return -np.inf


def exp(x):
    try:
        return math.exp(x)
    except:
        return np.inf


exp = np.vectorize(exp)
log10 = np.vectorize(log10)

omega_m = 0.272
omega_gamma = 0.728


def E(z):
    return (omega_m * (1 + z) ** 3 + omega_gamma) ** 0.5


def a_to_time_Hogg(a):
    z = 1 / a - 1

    result = integrate.quad(lambda x: ((1 + x) * E(x)) ** (-1), z, np.inf)

    return result[0]


a_to_time_Hogg = np.vectorize(a_to_time_Hogg)

# Quenching fraction:
tinker = np.loadtxt("tinker_fqcen_SDSS_M9.7.dat")

ms_tinker = tinker[:, 0]
fraction_tinker = tinker[:, 1]
error_tinker = tinker[:, 2]


# start from logMh=11 halos:

class Interpolate_median_Bolshoi_tree():
    def update_kwargs(self, kwargs):

        self.kwargs = kwargs

    def interpolate_to_behroozi(self):

        ## Interpolate for M11.0 to 14.0 with 0.1 dex bin

        # read a_target:


        pkl_file = open("a_Behroozi.pkl", 'rb')
        a_target = pickle.load(pkl_file)
        pkl_file.close()

        min = 11.0
        # also need to calculate HMF weight factor for these halos:
        for k in range(0,31):

            mass = min+k*0.1
            mass_tree = min+0.05+k*0.1
            mass = "{0:.1f}".format(mass)

            mass_tree = "{0:.2f}".format(mass_tree)

            # read analytic trees from 11.05 to 14.05 with 0.1 dex bin size:

            data_path = "M10.0/"

            data_path = data_path.replace("M10.0", "M_"+str(mass_tree))

            # Only use 100 halos sometimes. Make it quicker
            n_halo = 50

            # result all is z + Mass_stellar

            Mh_array = []
            count=0

            for halo in range(1, 300):
                file_i = "output_fulltree" + str(halo) + ".txt"

                result_i = np.loadtxt(data_path + file_i)


                z = result_i[:, 5]

                a = 1 / (1 + z)

                a = np.array(a)

                a = a[::-1]

                M_h = result_i[:, 1]

                M_h = np.array(M_h)

                M_h = M_h[::-1]

                # Let's only choose trees with length = 101:
                if len(a)==101:
                    Mh_array.append(M_h)
                    count += 1


                if count==n_halo:
                    break


            Mh_array = np.array(Mh_array).T
                
            print(Mh_array.shape)


            output = open("a_target_analytic.pkl", 'wb')
            pickle.dump(a, output)
            output.close()

            # save trees:

            save_path = "Analytic_trees_100"+"M_"+str(mass_tree)+".pkl"

            output = open(save_path, 'wb')
            pickle.dump(Mh_array, output)
            output.close()

    def calculate_hmf(self):

        path = "HMF_8_to_16/" + "all_plots/" + "mVector_PLANCK-SMT .txt"

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

        m = fusion[:, 0]

        logm = log10(m)

        dndlog = np.array(fusion[:, 7]).ravel()
        n = np.array(fusion[:, 8]).ravel()

        logn = log10(n)

        z = np.polyfit(logm, logn, 10)

        p = np.poly1d(z)

        target = np.arange(11.05, 14.15, 0.1)

        # use linear interpolation

        n_interpolate = np.interp(target, xp=logm, fp=n)

        """
        plt.plot(target,n_interpolate,"ro",alpha=0.3)
        plt.plot(logm,n,"kx",alpha=0.3)
        plt.show()
        """

        output = open("HMF_11_14.pkl", 'wb')
        pickle.dump(n_interpolate, output)
        output.close()

        print(n_interpolate.shape)

        # normalize them:
        print(n_interpolate*len(n_interpolate)/np.sum(n_interpolate))





        # check median



model = Interpolate_median_Bolshoi_tree()

# model.read_tree()

model.interpolate_to_behroozi()

model.calculate_hmf()