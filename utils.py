# %matplotlib inline
import math
import numpy as np
import matplotlib.pyplot as plt

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 12, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plots(*args,legends = []):
    for data in args:
        plt.plot(data)
    if legends:
        plt.legend(legends)
    plt.yscale("log")
    plt.show()


def decision_stub(train_data,verbose=False):
    F_star = math.inf
    m = len(train_data)
    d = len(train_data[0]) - 1
    b_star = 0

    F0_p = sum([1 for data in train_data if data[-1] == 1])
    for j in range(d):
        train_data.sort(key=sortKeyGenerator(j))
        F_p = F0_p
        if F_p < F_star:
            F_star = F_p
            theta_star = train_data[0][j] - 1
            j_star = j
            b_star = 1
        for i in range(m - 1):
            F_p -= train_data[i][d]
            if F_p < F_star and train_data[i][j] != train_data[i + 1][j]:
                F_star = F_p
                theta_star = (train_data[i][j] + train_data[i + 1][j]) / 2
                j_star = j
                b_star = 1
        i = m - 1
        F_p -= train_data[i][-1]
        if F_p < F_star:
            F_star = F_p
            theta_star = train_data[i][j] + 0.5
            j_star = j
            b_star = 1

    F0_n = sum([1 for data in train_data if data[-1] == -1])

    for j in range(d):
        train_data.sort(key=sortKeyGenerator(j))
        F_n = F0_n
        if F_n < F_star:
            F_star = F_n
            theta_star = train_data[0][j] - 1
            j_star = j
            b_star = -1
        for i in range(m - 1):
            F_n += train_data[i][d]
            if F_n < F_star and train_data[i][j] != train_data[i + 1][j]:
                F_star = F_n
                theta_star = (train_data[i][j] + train_data[i + 1][j]) / 2
                j_star = j
                b_star = -1
        i = m - 1
        F_n += train_data[i][-1]
        if F_n < F_star:
            F_star = F_n
            theta_star = train_data[i][j] + 0.5
            j_star = j
            b_star = -1
    if verbose:
            train_data = np.array(train_data)
            print("+1/-1 ratrio:%.2f/%.2f"%(0.5+np.sum(train_data[:,-1])/m/2,0.5-np.sum(train_data[:,-1])/m/2))
            print("+1 features max:%.2f\t min:%.2f\t mean:%.2f\t median %.2f" %(
                np.max(train_data[train_data[:,-1]==1,j_star]),
                np.min(train_data[train_data[:,-1]==1,j_star]),
                np.mean(train_data[train_data[:,-1]==1,j_star]),
                np.median(train_data[train_data[:,-1]==1,j_star])
            ))            
            print("-1 features max:%.2f\t min:%.2f\t mean:%.2f\t median %.2f" %(
                np.max(train_data[train_data[:,-1]==-1,j_star]),
                np.min(train_data[train_data[:,-1]==-1,j_star]),
                np.mean(train_data[train_data[:,-1]==-1,j_star]),
                np.median(train_data[train_data[:,-1]==-1,j_star])
            ))
            
            print(
                "Feature = %d\tThreshold = %f\tPolorization = %d" % (
                    j_star, theta_star, b_star))
    return F_star / m


def sortKeyGenerator(i):
    def sortKey(v):
        return v[i]

    return sortKey
