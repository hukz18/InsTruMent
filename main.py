from dataset import get_data
from feature import get_feature
from supervised import svm_train
import unsuperivesd
import numpy as np


if __name__ == '__main__':
    labels = ['cel', 'cla', 'flu', 'gac', 'gel', 'org',
              'pia', 'sax', 'tru', 'vio', 'voi']  # 需要识别的乐器
    svm_classifiers = []  # 针对各种乐器的svm
    
    for label in labels:
        waves, samples, labels = get_data(label, 100)
        features = get_feature(waves, samples)
        svm_classifiers.append(svm_train(features, labels))
    
    # 无监督部分
    n_centers=len(labels)
    n_count = 100
    '''
    X=np.zeros([n_count*n_centers,24])    
    Y=np.zeros([n_count*n_centers],dtype=int)
    i=0
    for label in labels:
        waves, samples, labels = get_data(label, n_count,true_p=1)
        features = get_feature(waves, samples) #100*24*130
        features_mean =np.mean(features,axis=2)
        X[i*n_count:(i+1)*n_count,]=features_mean
        Y[i*n_count:(i+1)*n_count]=i
        i=i+1
    '''
    # 从文件读取mfcc 这个快一点，代码用的是上面那样的
    X = np.loadtxt("read_mfcc.txt", dtype=int, delimiter=" ")
    Y_predict = unsuperivesd.gmm_predict(X,n_centers=11,type="spherical") #撒豆子警告
    #Y_predict = unsuperivesd.hierarchical_predict(X,n_centers=11,linkage="average") #撒豆子警告
    count_martix = unsuperivesd.outcome_print(Y_predict,labels,n_count=100)
    #unsuperivesd.pca_plot(X)
    #X_new =unsuperivesd.pca_decomposition(X,3,labels,100,1) # PCA降维