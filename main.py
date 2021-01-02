from dataset import *
from feature import *
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

dataset_path = './IRMAS-TestingData-Part1'
is_gen_mfcc = False

if is_gen_mfcc:
    generate_features(dataset_path)

if __name__ == '__main__':
    # meta_labels=['wind','']
    labels = ['cel', 'cla', 'flu', 'gac', 'gel', 'org',
              'pia', 'sax', 'tru', 'vio', 'voi']  # 需要识别的乐器
    svm_classifiers = SVM()

    mfccs, label_list = get_test_data_whole(dataset_path)
    print(f'mfccs = {mfccs.shape}')
    result_mfcc = get_feature_npy(mfccs)
    print(f'result mfccs = {result_mfcc.shape}')

    for label in labels:
        print('training:' + label)
        # label_dict = {l: (True, 500) if l == label else (False, 500)
        #               for l in labels}
        # # print(label_dict)
        # mfccs, lbs = get_data_list_weighted_npy(dataset_path, label_dict)
        lbs = [True if label in l else False for l in label_list]
        lbs = np.array(lbs)
        print(f'lbs  = {lbs.shape}')
        svm_classifiers.add_svm(result_mfcc, lbs)
    test_iter = get_test_data_iter()
    # svm_classifiers.evaluate(test_iter)

    # 无监督部分

    # 从文件读取mfcc 这个快一点，代码用的是上面那样的

    # Y_predict = unsuperivesd.gmm_predict(X,n_centers=11,type="spherical") #撒豆子警告
    # Y_predict = unsuperivesd.hierarchical_predict(X,n_centers=11,linkage="average") #撒豆子警告
    # count_martix = unsuperivesd.outcome_print(Y_predict,labels,n_count=100)
    # unsuperivesd.pca_plot(X)
    # X_new =unsuperivesd.pca_decomposition(X,3,labels,100,1) # PCA降维

    # 试了一下t-SNE
