from dataset import get_data, get_test_data
from feature import get_feature
from supervised import SVM
import unsuperivesd
import numpy as np

'''
    Annotations: The annotation of the predominant instrument of 
    each excerpt is both in the name of the containing folder, 
    and in the file name: cello (cel), clarinet (cla), flute (flu), 
    acoustic guitar (gac), electric guitar (gel), organ (org), piano (pia), 
    saxophone (sax), trumpet (tru), violin (vio), and human singing voice (voi). 
    The number of files per instrument are: cel(388), cla(505), flu(451), 
    gac(637), gel(760), org(682), pia(721), sax(626), tru(577), vio(580), voi(778). 
'''

if __name__ == '__main__':
    # meta_labels=['wind','']
    labels = ['cel', 'cla', 'flu', 'gac', 'gel', 'org',
              'pia', 'sax', 'tru', 'vio', 'voi']  # 需要识别的乐器
    svm_classifiers = SVM()

    for label in labels:
        waves, samples, labels = get_data(label, 50, 0.3)
        features = get_feature(waves, samples)
        svm_classifiers.add_svm(features, labels)
    test_iter = get_test_data()
    svm_classifiers.evaluate(test_iter)
    # 无监督部分

    # 从文件读取mfcc 这个快一点，代码用的是上面那样的

    # Y_predict = unsuperivesd.gmm_predict(X,n_centers=11,type="spherical") #撒豆子警告
    # Y_predict = unsuperivesd.hierarchical_predict(X,n_centers=11,linkage="average") #撒豆子警告
    # count_martix = unsuperivesd.outcome_print(Y_predict,labels,n_count=100)
    # unsuperivesd.pca_plot(X)
    # X_new =unsuperivesd.pca_decomposition(X,3,labels,100,1) # PCA降维

    # 试了一下t-SNE
