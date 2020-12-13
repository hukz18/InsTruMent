from dataset import get_data
from feature import get_feature
from supervised import *

if __name__ == '__main__':
    labels = ['cel']  # 需要识别的乐器
    svm_classifiers = []  # 针对各种乐器的svm
    for label in labels:
        waves, samples, labels = get_data(label, 10)
        features = get_feature(waves, samples)
        svm_classifiers.append(svm_train(features, labels))
