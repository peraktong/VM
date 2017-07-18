import numpy as np
from scipy import integrate
import time
import matplotlib.pyplot as plt
from termcolor import colored
import math
import os
from helpers.SimulationAnalysis import SimulationAnalysis, readHlist, iterTrees, getMainBranch
import pickle
import emcee
import scipy.optimize as op


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

class Fit_Bolshoi_tree():
    def update_kwargs(self, kwargs):

        self.kwargs = kwargs

    def read_tree(self):

        # From 11.0 to 15 with bin size=0.1. Your f_con works well for 9.75 to 12.55...


        merger_tree_median = []


        merger_tree_all = []

        min = 11.0

        # construct merger tree median;

        Behroozi_median = []
        for i in range(0, 31):
            number = min + 0.1 * i
            number = "{0:.1f}".format(number)

            merger_tree_median.append("M" + str(number) + "=None")

        # Split
        merger_tree_median = dict(s.split("=") for s in merger_tree_median)

        for i in range(0, 31):
            number = min + 0.1 * i
            number = "{0:.1f}".format(number)

            # load trees:

            save_path = "Bolshoi_tree_interpolated_" + str(number) + ".pkl"

            pkl_file = open(save_path, 'rb')
            array_i = pickle.load(pkl_file)
            pkl_file.close()

            merger_tree_all.extend(array_i)

            # load median trees:

            pkl_file = open(save_path.replace("interpolated_", "interpolated_median_"), 'rb')
            median_i = pickle.load(pkl_file)
            pkl_file.close()

            merger_tree_median["M" + str(number)] = median_i
        merger_tree_all = np.array(merger_tree_all).T

        # prepare merger_tree for the future fitting with HMF weighted.
        self.merger_tree_median = merger_tree_median
        self.merger_tree_all = merger_tree_all

        # Let's process them:

        # calculate delta M_h:



        zero = np.ones(len(merger_tree_all[0, :]))

        low = np.vstack((zero, merger_tree_all))

        up = np.vstack((merger_tree_all, zero))

        delta_Mh_all = up - low

        delta_Mh_all = delta_Mh_all[:-1, :]

        delta_Mh_all[0, :] = 0

        self.delta_Mh_all = delta_Mh_all

        # calculate delta_Ms multiplication array:
        ones_2d = np.ones([178, 178])

        ones_2d = np.tril(ones_2d, 0)

        ones_3d = np.repeat(ones_2d[:, np.newaxis, :], len(delta_Mh_all[0, :]), axis=1)

        self.ones_3d = ones_3d

        # print(ones_3d[:,2,:])


        # read a:



        pkl_file = open("a_Behroozi.pkl", 'rb')
        a_target = pickle.load(pkl_file)
        pkl_file.close()

        self.a_target = a_target

        a_2d = np.tile(a_target, (len(merger_tree_all[0, :]), 1)).T

        self.a_2d = a_2d

        # t target:

        t_target = a_to_time_Hogg(a_target)

        self.t_target = t_target

        t_2d = np.tile(t_target, (len(merger_tree_all[0, :]), 1)).T

        self.t_2d = t_2d

        # calculate delta_t

        delta_t = self.t_to_delta_t(t_target)

        delta_t_2d = np.tile(delta_t, (len(merger_tree_all[0, :]), 1)).T

        self.delta_t_2d = delta_t_2d


    def read_tree_sep(self,lower,upper):

        # From 11.0 to 15 with bin size=0.1. Your f_con works well for 9.75 to 12.55...


        merger_tree_median = []


        merger_tree_all = []

        min = 11.0

        # construct merger tree median;

        Behroozi_median = []
        for i in range(lower, upper):
            number = min + 0.1 * i
            number = "{0:.1f}".format(number)

            merger_tree_median.append("M" + str(number) + "=None")

        # Split
        merger_tree_median = dict(s.split("=") for s in merger_tree_median)

        for i in range(lower, upper):
            number = min + 0.1 * i
            number = "{0:.1f}".format(number)

            # load trees:

            save_path = "Bolshoi_tree_interpolated_" + str(number) + ".pkl"

            pkl_file = open(save_path, 'rb')
            array_i = pickle.load(pkl_file)
            pkl_file.close()

            merger_tree_all.extend(array_i)

            # load median trees:

            pkl_file = open(save_path.replace("interpolated_", "interpolated_median_"), 'rb')
            median_i = pickle.load(pkl_file)
            pkl_file.close()

            merger_tree_median["M" + str(number)] = median_i
        merger_tree_all = np.array(merger_tree_all).T

        # prepare merger_tree for the future fitting with HMF weighted.
        self.merger_tree_median = merger_tree_median
        self.merger_tree_all = merger_tree_all

        # Let's process them:

        # calculate delta M_h:



        zero = np.ones(len(merger_tree_all[0, :]))

        low = np.vstack((zero, merger_tree_all))

        up = np.vstack((merger_tree_all, zero))

        delta_Mh_all = up - low

        delta_Mh_all = delta_Mh_all[:-1, :]

        delta_Mh_all[0, :] = 0

        self.delta_Mh_all = delta_Mh_all

        # calculate delta_Ms multiplication array:
        ones_2d = np.ones([178, 178])

        ones_2d = np.tril(ones_2d, 0)

        ones_3d = np.repeat(ones_2d[:, np.newaxis, :], len(delta_Mh_all[0, :]), axis=1)

        self.ones_3d = ones_3d

        # print(ones_3d[:,2,:])


        # read a:



        pkl_file = open("a_Behroozi.pkl", 'rb')
        a_target = pickle.load(pkl_file)
        pkl_file.close()

        self.a_target = a_target

        a_2d = np.tile(a_target, (len(merger_tree_all[0, :]), 1)).T

        self.a_2d = a_2d

        # t target:

        t_target = a_to_time_Hogg(a_target)

        self.t_target = t_target

        t_2d = np.tile(t_target, (len(merger_tree_all[0, :]), 1)).T

        self.t_2d = t_2d

        # calculate delta_t

        delta_t = self.t_to_delta_t(t_target)

        delta_t_2d = np.tile(delta_t, (len(merger_tree_all[0, :]), 1)).T

        self.delta_t_2d = delta_t_2d


    def delta_Ms_to_Ms(self, delta_Ms):

        array_all = np.tile(delta_Ms, (len(delta_Ms), 1))

        array_all = np.tril(array_all, k=0)
        sum = np.sum(array_all, axis=1)

        return sum

    def Mh_to_delta_Mh(self, Mh):

        up = np.append(Mh, [0])
        low = np.append([0], Mh)

        delta_Mh = np.array((up - low)[:-1])

        # Attention!!
        delta_Mh[0] = 0

        return delta_Mh

    def t_to_delta_t(self, t):

        up = np.append(t, [0])
        low = np.append([0], t)

        delta_t = np.array((up - low)[:-1])

        # Attention!!
        delta_t[0] = 0

        return delta_t

    def f_con_Jason(self, a, Mh):

        # a and Mh are both 2d arrays


        kwargs = self.kwargs

        f0_log = kwargs["f0_log"]

        f0 = 10 ** (f0_log)

        A1 = kwargs["A1"]
        A2 = kwargs["A2"]
        A3 = kwargs["A3"]
        mht = kwargs["mht"]
        A4 = kwargs["A4"]
        A5 = kwargs["A5"]

        A6 = kwargs["A6"]

        # I think we need to calculate M_critical from reading the data:

        At = A3 * (log10(Mh)) ** 2 + A4 * (log10(Mh)) + A5

        # fix a_s
        As = A6

        # matrix

        f_con_original = f0 * ((Mh / mht) ** A1) * (log10((Mh / mht))) ** A2 * exp(-(At - a) ** 2 / As)

        # Quenched fraction part:

        mhc_log = kwargs["mhc_log"]

        mhc = 10 ** (mhc_log)

        sigmah = kwargs["sigmah"]

        alphah = kwargs["alphah"]

        f_q = (exp((mhc_log - log10(Mh)) / sigmah)) ** alphah

        mask = Mh < mhc

        f_q[mask] = 1

        return f_q * f_con_original

    # return a chi_quenched
    def fit_quenched_fraction(self):

        merger_tree_all = self.merger_tree_all

        delta_Mh_all = self.delta_Mh_all

        f_con_2d = self.f_con_Jason(a=self.a_2d, Mh=merger_tree_all)

        delta_ms_all = f_con_2d * delta_Mh_all

        """

        for mi in range(0,1000):

            plt.plot(self.a_target,log10(delta_ms_all[:,mi]),alpha=0.1)
        plt.show()

        """

        delta_ms_3d = np.repeat(delta_ms_all[:, :, np.newaxis], 178, axis=2)

        ms_all = delta_ms_3d * self.ones_3d

        ms_all = np.nansum(ms_all, axis=2)

        # calculate index where the quenching starts:


        mhc_log = kwargs["mhc_log"]

        mhc = 10 ** (mhc_log)

        mask = merger_tree_all < mhc

        mask = np.array(mask, dtype=int)

        index_all = np.sum(mask, axis=0) - 1

        # calculate SFR array:

        SFR_2d = delta_ms_all / self.delta_t_2d

        index_all = tuple(zip(index_all, np.arange(len(index_all))))

        rate_array = SFR_2d[-1, :] / [SFR_2d[x] for x in index_all]

        judge_array = np.array(rate_array < 0.1, dtype=int)

        # print(judge_array)


        final_mass_log = np.array(log10(ms_all[-1, :]))

        fraction_array = []

        length_array = []

        for m in range(0, 19):
            lower = 9.7 + 0.1 * m
            upper = 9.8 + 0.1 * m

            mask = (final_mass_log > lower) & (final_mass_log < upper)

            fraction_array.append(np.nansum(judge_array[mask]) / np.nansum(np.array(mask,dtype=int)))

            length_array.append(np.nansum(np.array(mask,dtype=int)))

        fraction_array = np.array(fraction_array)

        # print(length_array)

        # Another mask here;


        # print("fraction array")

        self.fraction_array = fraction_array

        # calculate chi-quenched:

        ivar = (error_tinker)**(-2)
        ### Need to normalize ivar!!!

        ivar = ivar*len(ivar)/np.sum(ivar)

        chi_quenched = np.nansum((fraction_array-fraction_tinker)**2*ivar)
        self.chi_quenched = chi_quenched

        return chi_quenched

    # return a behroozi


    # return a behroozi chi
    def fit_behroozi(self):

        chi_B_array = []

        # fit from 11.0 to 13.0 with bin size=0.5

        for i in range(0,5):

            number = 11.0 + 0.5 * i
            number = "{0:.1f}".format(number)


            B_path = "Behroozi_revised_M11.0.pkl"

            B_path = B_path.replace("M11.0", "M" +str(number))


            pkl_file = open(B_path, 'rb')
            Behroozi_fusion = pickle.load(pkl_file)
            pkl_file.close()


            # calculate M* of the median Bolshoi:

            # fusion = np.c_[a, s, h,ds,dh,f_con_array]


            data_path = "Median_a_Mh_300_M10.0.pkl"

            data_path = data_path.replace("M10.0", "M" + str(number))

            pkl_file = open(data_path, 'rb')
            median_fusion = pickle.load(pkl_file)
            pkl_file.close()

            a_us = median_fusion[:,0]
            Mh = median_fusion[:, 1]

            delta_Mh = self.Mh_to_delta_Mh(Mh=Mh)

            f_con_2d = self.f_con_Jason(a=np.atleast_2d(a_us), Mh=np.atleast_2d(Mh))

            delta_Ms = (f_con_2d * delta_Mh).ravel()

            Ms = model.delta_Ms_to_Ms(delta_Ms)

            # calculate
            log_median_Ms = [log10(x) for x in Ms]

            log_Behroozi_Ms = [log10(x) for x in Behroozi_fusion[:, 1]]

            log_median_Ms = np.array(log_median_Ms)
            log_Behroozi_Ms = np.array(log_Behroozi_Ms)

            slope = np.sum(log_Behroozi_Ms[a_index_Behroozi] * log_median_Ms[a_index_us]) / np.sum(
                log_median_Ms[a_index_us] ** 2)


            chi = np.sum((slope*log_median_Ms[a_index_us] - log_Behroozi_Ms[a_index_Behroozi]) ** 2 * a_weight_factor)

            chi_B_array.append(chi)
        chi_B_array = np.array(chi_B_array)

        self.chi_B_array = chi_B_array

        return np.sum(chi_B_array)




## Initialize;


kwargs = {"MCMC": None, "A1": None, "A2": None, "A3": None, "f0_log": None, "A4": None, "A5": None, "A6": None,
              "mht": None,
              "mst": None, "zc": None, "sigmaz": None,
              "alphaz": None, "mhc_log": None, "msc_log": None, "sigmah": None, "sigmag": None, "alphah": None,
              "alphag": None}



model = Fit_Bolshoi_tree()


counter=0

def lnlike(theta, x, y):
    global counter

        # introduce our model here.

        # Theta is a tuple
        # f0_log, mhc_log,sigmah,alphah = theta

    mhc_log, sigmah, alphah = theta

        # kwargs["f0_log"] = f0_log
    kwargs["mhc_log"] = mhc_log

    kwargs["sigmah"] = sigmah
    kwargs["alphah"] = alphah

    model.update_kwargs(kwargs=kwargs)

        # Let's change the model from simple linear to our complex model.

        # y_model is the value from our method

        # Now we define chi as the abs of the fraction of the quenched galaxies


    counter = counter + 1
    print("Doing %d" % counter)
    print(mhc_log, sigmah, alphah)

    chi = model.fit_quenched_fraction()

    print("Tinker fraction")
    print(fraction_tinker)

    print("Our fraction")
    print(model.fraction_array)

    print(colored("chi", "red"))
    print(colored(chi, "red"))

    return -0.5 * (chi)




        # Set the range of the parameters
def lnprior(theta):
    mhc_log, sigmah, alphah = theta

        #
        # -3 < f0_log < 0 and
    if 8 < mhc_log < 13 and 0 < sigmah < 3 and -5 < alphah < 5:
        return 0.0
    return -np.inf


    # Define the initial condition of the MCMC chain: Position/initial values

def lnprob(theta, x, y):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf, str(lnlike(theta, x, y) / (-0.5))
    return lp + lnlike(theta, x, y), str(lnlike(theta, x, y) / (-0.5))


start_time = time.time()

Mh_array = []

parameter_array = []

fraction_array = []




for ii in range(0,14):

    # Initial values:
    # From scipy best fitting model. Not sure whether to use this or use EMCEE.


    kwargs["f0_log"] = log10(0.00251552734235)

    kwargs["A1"] = -5.00156921458
    kwargs["A2"] = 36.7909234394
    kwargs["A3"] = -1.45406047315

    kwargs["A4"] = 31.1501956219

    kwargs["A5"] = -163.028606123

    kwargs["A6"] = 2.83787333037

    kwargs["mht"] = 10 ** (8)
    kwargs["mst"] = 10 ** (8)

    ###### Now we add Jeremy's stellar quenching model:

    # Here we fit fo, mst,sigmag and alphag.


    kwargs["mhc_log"] = 11.5 - (10-ii)*0.2

    kwargs["sigmah"] = 0.52

    # 3.63
    kwargs["alphah"] = 0

    model = Fit_Bolshoi_tree()

    # update kwargs:

    model.update_kwargs(kwargs=kwargs)
    # read trees

    model.read_tree_sep(lower=int(0 + ii * 2), upper=int(1 + ii * 2))

    # Let's append:

    Mh_array.append(11.0+ii*0.2)

    # read parameters:


    save_path = "Bolshoi_trees_Matrix_best_fit_v1.pkl".replace("best_fit_", "best_fit_" + str(11.0+ii*2) + "_sep_")
    save_path_backup = save_path

    save_path = save_path.replace("best_fit_", "EMCEE")



    pkl_file = open(save_path, 'rb')
    results = pickle.load(pkl_file)
    pkl_file.close()

    save_path = save_path.replace("EMCEE", "EMCEE_chi_")

    pkl_file = open(save_path, 'rb')
    chi = pickle.load(pkl_file)
    pkl_file.close()

    chi = np.array(chi.T,dtype=float)

    mask = chi > 1.2

    chi = chi[mask]
    results = results[mask,:]

    model.fit_behroozi()
"""


    try:

        index = np.argmin(chi)

        print(chi[index])
        print(results[index, :])

        parameter_array.append(results[index, :])

        # calculate fraction:




        kwargs["mhc_log"] = results[index, 0]

        kwargs["sigmah"] = results[index, 1]

        # 3.63
        kwargs["alphah"] = results[index, 2]

        model.update_kwargs(kwargs=kwargs)

        model.fit_quenched_fraction()

        fraction_i = model.fraction_array

        print(fraction_i)

        fraction_array.append(fraction_i)
    except:
        on=1







"""

