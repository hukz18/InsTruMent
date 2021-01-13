from dataset import *
from feature import *
from supervised import ML_Classifier, ML_Regressor
import unsuperivesd
import numpy as np
import pickle

'''
    Annotations: The annotation of the predominant instrument of
    each excerpt is both in the name of the containing folder,
    and in the file name: cello (cel), clarinet (cla), flute (flu),
    acoustic guitar (gac), electric guitar (gel), organ (org), piano (pia),
    saxophone (sax), trumpet (tru), violin (vio), and human singing voice (voi).
    The number of files per instrument are: cel(388), cla(505), flu(451),
    gac(637), gel(760), org(682), pia(721), sax(626), tru(577), vio(580), voi(778).
'''

dataset_path = './dataset/TrainingData'
is_gen_mfcc = False
is_gen_dataset = False
if is_gen_dataset:
    for i in range(1, 4):
        generate_dataset('./dataset/IRMAS-TestingData-Part%d' % i, i)
if is_gen_mfcc:
    generate_features(dataset_path)


def train_classifier():  # 使用单层判别器训练
    ml_classifier = ML_Classifier('SVM')  # use 'SVM', 'RF', or 'XGB'
    mfccs, label_list = get_test_data_whole(dataset_path)
    result_mfcc = get_feature_npy(mfccs)
    for label in ml_classifier.labels:
        print('training:' + label)
        lbs = np.array([True if label in l else False for l in label_list])
        print(result_mfcc.shape)
        print(len(lbs))
        print(label)
        ml_classifier.add_clf(result_mfcc, lbs, label)
    test_iter = get_test_data_iter()
    ml_classifier.evaluate(test_iter)
    print('input y to save the current model:')
    if input() == 'y':
        with open('classifier.pkl', 'wb') as f:
            pickle.dump(ml_classifier, f)


def train_meta_classifier():  # 使用双层判别器训练
    ml_classifier = ML_Classifier('SVM')  # use 'SVM', 'RF', or 'XGB'
    mfccs, label_list = get_test_data_whole(dataset_path)
    result_mfcc = get_feature_npy(mfccs)
    for labels in ml_classifier.meta_labels:
        lbs = np.array([True if np.any(np.in1d(np.array(labels), np.array(l))) else False for l in label_list])
        ml_classifier.add_meta_clf(result_mfcc, lbs)
        sub_label_list, sub_result_mfcc = np.array(label_list)[lbs].tolist(), result_mfcc[lbs]
        if len(labels) > 1:
            for label in labels:
                print('training:' + label)
                lbs = np.array([True if label in l else False for l in sub_label_list])
                ml_classifier.add_clf(sub_result_mfcc, lbs, label)
    test_iter = get_test_data_iter()
    ml_classifier.evaluate(test_iter)

    print('input y to save the current model:')
    if input() == 'y':
        with open('meta_classifier.pkl', 'wb') as f:
            pickle.dump(ml_classifier, f)


def train_regressor():  # 使用回归器训练
    ml_regressor = ML_Regressor()
    mfccs, label_list = get_test_data_whole(dataset_path)
    result_mfcc = get_feature_npy(mfccs)
    for label in ml_regressor.labels:
        print('training:' + label)
        lbs = np.array([1 if label in l else -1 for l in label_list])
        ml_regressor.add_rgs(result_mfcc, lbs, label)
    test_iter = get_test_data_iter()
    ml_regressor.evaluate(test_iter)
    print('input y o save the current model:')
    if input() == 'y':
        with open('regressor.pkl', 'wb') as f:
            pickle.dump(ml_regressor, f)


if __name__ == '__main__':
    train_classifier()
    # 从文件读取mfcc 这个快一点，代码用的是上面那样的

    # Y_predict = unsuperivesd.gmm_predict(X,n_centers=11,type="spherical") #撒豆子警告
    # Y_predict = unsuperivesd.hierarchical_predict(X,n_centers=11,linkage="average") #撒豆子警告
    # count_martix = unsuperivesd.outcome_print(Y_predict,labels,n_count=100)
    # unsuperivesd.pca_plot(X)
    # X_new = unsuperivesd.pca_decomposition(X,3,labels,100,1) # PCA降维

    # 试了一下t-SNE
