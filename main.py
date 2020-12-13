from dataset import get_data
from feature import get_feature
from supervised import *

if __name__ == '__main__':
    labels = ['cel']  # 需要识别的乐器
    svm_classifiers = []  # 针对各种乐器的svm
    for label in labels:
        (pos_wav, pos_sr), (neg_wav, neg_sr) = get_data(label)
        pos_feature, neg_feature = get_feature(
            (pos_wav, pos_sr), (neg_wav, neg_sr))
        svm_classifiers.append(svm_train(pos_feature, neg_feature))
