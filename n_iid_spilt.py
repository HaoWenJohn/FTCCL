
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def draw_heatmap(data,save_file,cbar=True):
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(data = data,cmap="YlGnBu",fmt=".0%" ,vmin=0.0, vmax=1.0,annot=True,cbar=cbar)
    if cbar:
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)
    plt.ylabel('clients',fontsize=20)
    plt.xlabel('labels',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.rcParams.update({'font.size':20})
    plt.savefig(f"{save_file}")


def matrix(client_datas,clazz_num):
    res = []
    for key in client_datas:
        labels = client_datas[key]["y_train"]
        _,counts = np.unique(labels,return_counts=True)
        res.append(counts)
    res =np.array(res)
    print(res.shape)
    res = res/np.sum(res,axis=0)
    print(res)
    return res


