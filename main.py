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

dataset_path = './dataset/IRMAS-TestingData-Part[2-3]'
is_gen_mfcc = False

if is_gen_mfcc:
    generate_features(dataset_path)

if __name__ == '__main__':
    # meta_labels=['wind','']

    svm_classifiers = SVM()

    mfccs, label_list = get_test_data_whole(dataset_path)

    result_mfcc = get_feature_npy(mfccs)

    #
    # for labels in svm_classifiers.meta_labels:
    #     lbs = np.array([True if np.any(np.in1d(np.array(labels), np.array(l))) else False for l in label_list])
    #     svm_classifiers.add_meta_svm(result_mfcc, lbs)
    #     sub_label_list, sub_result_mfcc = np.array(label_list)[lbs].tolist(), result_mfcc[lbs]
    #     for label in labels:
    #         print('training:' + label)
    #         lbs = np.array([True if label in l else False for l in sub_label_list])
    #         svm_classifiers.add_svm(sub_result_mfcc, lbs, label)

    for label in svm_classifiers.labels:
        print('training:' + label)
        lbs = np.array([True if label in l else False for l in label_list])
        svm_classifiers.add_svm(result_mfcc, lbs, label)
    test_iter = get_test_data_iter()
    svm_classifiers.evaluate(test_iter)


    # 从文件读取mfcc 这个快一点，代码用的是上面那样的

    # Y_predict = unsuperivesd.gmm_predict(X,n_centers=11,type="spherical") #撒豆子警告
    # Y_predict = unsuperivesd.hierarchical_predict(X,n_centers=11,linkage="average") #撒豆子警告
    # count_martix = unsuperivesd.outcome_print(Y_predict,labels,n_count=100)
    # unsuperivesd.pca_plot(X)
    # X_new =unsuperivesd.pca_decomposition(X,3,labels,100,1) # PCA降维

    # 试了一下t-SNE
