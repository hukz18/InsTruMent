from dataset import get_data
from feature import get_feature
from supervised import *

if __name__ == '__main__':
    labels = []  # 需要识别的乐器
    svm_classifiers = [] # 针对各种乐器的svm
    for label in labels:
        pos_files, neg_files = get_data(label)
        features = get_feature(pos_files, neg_files)
        svm_classifiers.append(svm_train(features))
