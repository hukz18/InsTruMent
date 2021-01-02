from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import warnings
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn import manifold
import dataset
import feature


def pca_plot(X):  # 画PCA的拐点图，差不多选前3个或者前2的变量没啥问题
    PCA_model = PCA(n_components=10)
    PCA_model.fit(X)
    PCA_var = PCA_model.explained_variance_
    plt.plot(np.arange(0, 10), PCA_var)
    plt.show()


def pca_decomposition(X, n, labels, n_count=100, picshow=0):  # 撒豆子警告
    warnings.filterwarnings('ignore')
    sns.set()
    PCA_model = PCA(n_components=n)
    PCA_model.fit(X)
    X_new = PCA_model.fit_transform(X)
    fig = plt.figure(figsize=(20, 15))
    if n == 3 and picshow:
        colors = sns.color_palette("Spectral", len(labels))
        # colors=["#9E0142","#D53E4F","#F46D43","#FDAE61", "#FEE08B","#FFFFBF","#E6F598","#ABDDA4","#66C2A5","#3288BD","#5E4FA2"]
        ax = fig.add_subplot(111, projection='3d')
        for i in range(0, len(labels)):
            x = X_new[i * n_count:(i + 1) * n_count, 0]
            y = X_new[i * n_count:(i + 1) * n_count, 1]
            z = X_new[i * n_count:(i + 1) * n_count, 2]
            ax.scatter(x, y, z, c=colors[i], marker='o', label=labels[i], s=10)
        plt.legend()
        plt.show()

    if n == 2 & picshow:
        colors = sns.color_palette("Spectral", len(labels))
        for i in range(0, len(labels)):
            x = X_new[i * n_count:(i + 1) * n_count, 0]
            y = X_new[i * n_count:(i + 1) * n_count, 1]
            plt.scatter(x, y, c=colors[i], marker='o', label=labels[i], s=10)
        plt.legend()
        plt.show()

    return X_new


def outcome_print(Y_predict, labels, n_count=100):  # 聚类结果可视化
    count_martix = np.zeros([len(labels), max(Y_predict) + 1], dtype=int)
    for i in range(0, len(labels)):
        label_count = np.bincount(Y_predict[i * n_count:(i + 1) * n_count])
        count_martix[i, 0:len(label_count)] = label_count
        print(labels[i], "| max class =", np.argmax(label_count), "| count =", label_count[np.argmax(label_count)])
        # print(labels[i],np.argmax(label_count),label_count)
    ax1 = sns.heatmap(count_martix, cmap='rainbow', annot=True)
    ax1.set_yticklabels(labels)
    return count_martix


def gmm_predict(X, n_centers=11, type="diag"):  # 高斯混合模型
    """
    :param n_centers: 聚类数目
    :param type: 表现最好的方法
    :return:
    """
    gmm = GaussianMixture(n_components=11, covariance_type=type).fit(X)
    Y_predict = GaussianMixture.predict(gmm, X)  # predict_proba
    return Y_predict


def hierarchical_predict(X, n_centers=30, linkage="ward"):  # 层次聚类
    ac = AgglomerativeClustering(n_clusters=n_centers, affinity='euclidean', linkage=linkage)
    # {"ward", "complete", "average", "single"}
    ac.fit(X)
    Y_predict = ac.fit_predict(X)
    return Y_predict


def t_SNE2(X):
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X)
    return X_tsne


def get_tsne():
    labels = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']  # 需要识别的乐器
    mfccs = {}

    for i in range(11):
        label = labels[i]
        label_dict = {}
        for ii in range(11):
            j = labels[ii]
            if (j == label):
                label_dict[j] = (True,500)
            else:
                label_dict[j] = (False,0)
        mfcc,true = dataset.get_data_list_weighted_npy('./dataset/IRMAS-TrainingData',label_dict)
        mfccs[label] = get_feature_tsne(mfcc) 
    
    X = []
    y = []
    for label in labels:
        X += mfccs[label].tolist()
        y += [label]*500
    X_tsne = t_SNE2(X)

    for i in range(len(X)):
        if y[i] == labels[0]:
            plt.plot(X_tsne[i,0],X_tsne[i,1],'r')
        elif y[i] == labels[1]:
            plt.plot(X_tsne[i,0],X_tsne[i,1],'y')
        elif y[i] == labels[2]:
            plt.plot(X_tsne[i,0],X_tsne[i,1],'g')
        elif y[i] == labels[3]:
            plt.plot(X_tsne[i,0],X_tsne[i,1],'b')
        elif y[i] == labels[4]:
            plt.plot(X_tsne[i,0],X_tsne[i,1],'k')
        elif y[i] == labels[5]:
            plt.plot(X_tsne[i,0],X_tsne[i,1],'p')
        elif y[i] == labels[6]:
            plt.plot(X_tsne[i,0],X_tsne[i,1],'m')
        elif y[i] == labels[7]:
            plt.plot(X_tsne[i,0],X_tsne[i,1],'c')
        elif y[i] == labels[8]:
            plt.plot(X_tsne[i,0],X_tsne[i,1],'peachpuff')
        elif y[i] == labels[9]:
            plt.plot(X_tsne[i,0],X_tsne[i,1],'crimson')
        elif y[i] == labels[10]:
            plt.plot(X_tsne[i,0],X_tsne[i,1],'purple')
    plt.show()
    return X_tsne,y

def get_feature_tsne(mfccs):
    '''
    input:
        mfccs : N * F * L , a set of mfccs
    return:
        mfcc_result : N * F
    '''
    freq_sum = np.sum(mfccs, axis=1)
    # print(freq_sum)
    sort_ind = np.argsort(freq_sum, axis=1)
    # print(sort_ind)
    sort_ind_ext = np.tile(sort_ind, (mfccs.shape[1], 1, 1))
    sort_ind_ext = np.moveaxis(sort_ind_ext, 1, 0)
    # print(sort_ind_ext)
    result = np.take_along_axis(mfccs, sort_ind_ext, axis=2)
    # print(result)
    avg_max_result = np.mean(result[:, :, 50:], axis=2)
    # print(avg_max_result)
    return avg_max_result

if __name__ == '__main__':
    get_tsne()