import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import math

def log10(x):

    if x>0:
        return math.log10(x)

    else:
        return

# 11


index_array = ["10.0","10.5",'11.0',"11.5","12.0","12.5","13.0","13.5","14.0","14.5","15.0"]



def read_file(index):

    index = str(index)

    data_path = "/Users/caojunzhi/Desktop/NYU/Laboratory/My_code/Cosmology_2017.4-8_Jason/Behroozi_2013_data/sfh_stats/"

    file_path = "stats_a1.002310_absolute_mpeak_10.000_cen.dat"

    path = data_path + file_path

    path = path.replace("10.0",index)

    # print(path)

    with open(path) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    content = np.array(content,dtype=str)


    N = len(content)

    a = []
    s = []
    h = []

    for i in range(0,N):

        row = content[i]
        row = row.split(" ")

        # print(row)

        # structure is a+stellar mass+halo mass

        check = content[i].find("sm_mp")

        check_2 = content[i].find("hm_mp")
        if check!=-1:

            # print(row[0],row[1],row[2])
            a = np.append(a,row[1])
            s = np.append(s,row[2])

        elif check_2!=-1:

            # print(row[0],row[1],row[2])
            h = np.append(h,row[2])

    # save in pickle

    a = np.array(a, dtype=float)
    s = np.array(s, dtype=float)
    h = np.array(h, dtype=float)

    # from stellar mass and halo mass, we can derive f_con, too:

    # f_con = (DeltaM*)/(DeltaMh)/0.17

    # Let's do everything in a

    ds = [s[0]]
    dh = [h[0]]


    for k in range(1,len(s)):

        ds.extend([s[k]-s[k-1]])
        dh.extend([h[k] - h[k - 1]])

    ds = np.array(ds)
    dh = np.array(dh)

    f_con_array = ds/dh/0.17


    # fusion: a+s+h+ds+dh+f_con

    fusion = np.c_[a, s, h,ds,dh,f_con_array]

    fusion = np.array(fusion)

    print(fusion.shape)

    save_path = "Behroozi_revised_M" + index + ".pkl"

    output = open(save_path, 'wb')
    pickle.dump(fusion, output)
    output.close()

def plot_Behroozi_revised(index):

    load_path = "Behroozi_revised_M" + index + ".pkl"

    pkl_file = open(load_path, 'rb')
    fusion = pickle.load(pkl_file)
    pkl_file.close()

    figure_path = "/Users/caojunzhi/Downloads/upload_0530_Jeremy/"


    # Let's plot:


    font = {'family': 'normal',
            'weight': 'bold',
            'size': 20}

    matplotlib.rc('font', **font)

    stellar_mass = [log10(x) for x in fusion[:,1]]

    plt.subplot(1, 2, 1)

    plt.plot(fusion[:,0],stellar_mass,"ro",label="$Behroozi\quad revised$")
    plt.plot([],[],label = "Log[M_h] = %s"%index)
    plt.xlabel("$Scale\quad factor$")
    plt.ylabel("$Log[M_*]\quad (dex)$")
    plt.suptitle("$Stellar\quad mass\quad vs\quad scale\quad factor$")

    plt.legend()

    plt.subplot(1, 2, 2)


    plt.plot(fusion[:,0],fusion[:,5],"ro",label="$Behroozi\quad revised$")
    plt.plot([],[],label = "Log[M_h] = %s"%index)
    plt.xlabel("$Scale\quad factor$")
    plt.ylabel("$f_con\quad (dex)$")
    plt.suptitle("$f_con\quad vs\quad scale\quad factor$")

    plt.legend()



    fig = matplotlib.pyplot.gcf()

    # adjust the size based on the number of visit

    fig.set_size_inches(24.5, 9.5)

    save_path = figure_path+"Behroozi_revised_"+index + ".png"


    fig.savefig(save_path, dpi=500)

    plt.close()


for j in range(0,len(index_array)-1):

    print("calculating result %d" % j)

    read_file(index_array[j])

for j in range(0, len(index_array)-1):

    print("Plotting result %d"%j)

    plot_Behroozi_revised(index_array[j])

