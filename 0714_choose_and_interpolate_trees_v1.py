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
import matplotlib

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


    def choose_some_trees(self,number):

        if number>300000:
            print("Choose a number<300000")


        # Load index for fitting;
        fb = 0.17

        index_array = []
        mi = 11.0
        for i in range(0, 41):
            mi = float("{0:.2f}".format(mi))

            index_array.append(mi)
            mi += 0.1

        index_array = np.array(index_array, dtype=str)
        print(index_array)

        # calculate halos:


        pkl_file = open('Bolshoi_tree_300000.pkl', 'rb')
        Bolhshoi_tree_300000 = pickle.load(pkl_file)
        pkl_file.close()

        Mh_max = []
        results_all = []
        n_halo = number

        for i in range(0, len(Bolhshoi_tree_300000[:n_halo, 0])):
            # How to read tree!
            # scale factor + M_h
            tree_i = np.array(Bolhshoi_tree_300000[i, 1], dtype=float).T
            # print(tree_i.shape)
            results_all.append(tree_i)
            Mh_max.append(np.nanmax(tree_i[1, :]))

        Mh_max = np.array(Mh_max)
        Mh_max = [log10(x) for x in Mh_max]
        log_Mh_max = np.array(Mh_max)

        results_all = np.array(results_all)

        # print(log_Mh_max.shape)
        # print(results_all.shape)

        print("min and max")
        print(np.min(log_Mh_max))
        print(np.max(log_Mh_max))

        total = 0

        # start from 11.0 and stop at 15.0
        for i in range(0, 31):
            # see max Mh:

            delta = 0.05

            number = 11.0 + 0.1 * i

            mask = abs(log_Mh_max - (number)) < delta
            print("For %.2f" % (number))
            print(log_Mh_max[mask].shape)
            total += len(log_Mh_max[mask])
            # print(log_Mh_max[mask])

            results_i = results_all[mask]
            print(results_i.shape)

            number = "{0:.1f}".format(number)

            output = open("Bolshoi_tree_" + str(number) + ".pkl", 'wb')
            pickle.dump(results_i, output)
            output.close()

        print("tree numer is = %d" % total)

    def interpolate_to_behroozi(self):

        ## Interpolate for M11.0 to 14.0 with 0.1 dex bin

        # read a_target:


        pkl_file = open("a_Behroozi.pkl", 'rb')
        a_target = pickle.load(pkl_file)
        pkl_file.close()

        min = 11.0
        for k in range(0,31):

            mass = min+k*0.1
            mass = "{0:.1f}".format(mass)

            print("doing interpolation for %.2f"%float(mass))


            pkl_file = open("Bolshoi_tree_" + str(mass) + ".pkl", 'rb')
            Bolshoi_tree_i = pickle.load(pkl_file)
            pkl_file.close()

            # use numpy.interp

            mh_int_array = []

            for j in range(0,len(Bolshoi_tree_i)):

                tree_j = np.array(Bolshoi_tree_i[j],dtype=float)

                # Remember to reverse them...

                # interpolation

                mh_int_array.append(np.interp(x=a_target,xp=tree_j[:,0][::-1],fp=tree_j[:,1][::-1]))



            mh_int_array = np.array(mh_int_array)

            # save median halos!
            """
            

            output = open(save_path.replace("interpolated_","interpolated_median_"), 'wb')
            pickle.dump(np.nanmedian(mh_int_array,axis=0), output)
            output.close()

            
            """

            # check median



model = Interpolate_median_Bolshoi_tree()

model.choose_some_trees(number=30000)

model.interpolate_to_behroozi()

