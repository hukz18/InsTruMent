import numpy as np
from tqdm import tqdm
from sklearn import svm
import matplotlib.pyplot as plt
from feature import get_feature_npy
from evaluation import *
from dataset import get_data_list_weighted
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV, train_test_split


class SVM:
    def __init__(self):
        self.labels = ['cel', 'cla', 'flu', 'gac', 'gel',
                       'org', 'pia', 'sax', 'tru', 'vio', 'voi']
        self.label_to_idx = {v: i for (i, v) in enumerate(self.labels)}
        self.meta_labels = {}
        self.discriminators = []
        self.meta_discriminators = []
        self.class_num = 11
        self.meta_class_num = 4
        # self.meta_labels = []
        # self.meta_classifiers = []

    def add_meta_svm(self, labels):
        waves, samples, labels = get_data_list(labels, 2000, 0.5)
        features = get_feature(waves, samples)
        meta_svm = train_svm(features, labels)
        self.meta_discriminators.append(meta_svm)

    def add_svm(self, features, labels):
        sub_svm = train_svm(features, labels)
        self.discriminators.append(sub_svm)

    def predict_meta(self, wave, sample):
        wave = wave[np.newaxis, :]
        sample = sample[np.newaxis]
        feature = get_feature(wave, sample)
        feature = np.mean(feature, axis=2)
        result = np.zeros(self.meta_class_num)
        for i in range(self.meta_class_num):
            result[i] = self.meta_discriminators[i].predict(feature)
        return result

    def predict_whole(self, wave, sample):
        num_item = np.shape(wave, 0)
        result = np.zeros(num_item, self.class_num)
        feature = np.mean(get_feature(wave, sample), axis=2)
        for i in range(self.class_num):
            result[:, i] = self.discriminators[i].predict(feature)
        return result

    def predict_one(self, mfcc, meta_label=None):
        mfcc = mfcc[np.newaxis, :]
        result = np.zeros(self.class_num)
        feature = get_feature_npy(mfcc)
        for i in range(self.class_num):
            result[i] = self.discriminators[i].predict(feature)
        return result

    def evaluate(self, test_iter):
        true_labels, predict_labels = [], []
        for mfcc, labels in tqdm(test_iter):
            true_label = np.zeros(self.class_num)
            # meta_label = self.predict_one(wave, sample)
            # meta_label = np.where(meta_label)[0].tolist()  # 含有哪些基类
            for label in labels:
                true_label[self.label_to_idx[label]] = True
            true_labels.append(true_label)
            # predict_labels.append(self.predict_one(mfcc, meta_label))
            predict_labels.append(self.predict_one(mfcc))
        predict_labels = np.array(predict_labels)
        true_labels = np.array(true_labels)
        print(accuracy_score(true_labels, predict_labels))
        confusion = confusion_mat(np.mat(true_labels), np.mat(predict_labels))
        plt.matshow(confusion)
        plt.show()


def svm_param_select(X, Y, n_folds=3):
    # Cs = [0.01, 0.1, 1, 10]
    Cs = [5, 10, 20, 50]
    gammas = ['scale', 'auto', 0.0001, 0.005, 0.001]
    kernels = ['rbf', 'sigmoid']
    param_grid = {'C': Cs, 'gamma': gammas, 'kernel': kernels}
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=n_folds)
    grid_search.fit(X, Y)
    return grid_search.best_params_


def train_svm(X, Y):
    # X = np.mean(X, axis=2)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    param = svm_param_select(X_train, Y_train)
    print(param)
    svm_classifier = svm.SVC(**param)
    svm_classifier.fit(X_train, Y_train)
    predicted_train = svm_classifier.predict(X_train)
    predicted_test = svm_classifier.predict(X_test)
    print(accuracy_score(predicted_train, Y_train))
    print(accuracy_score(predicted_test, Y_test))
    return svm_classifier


def RF_train(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    RF_classifier = RandomForestClassifier(
        n_estimators=500, random_state=1993, max_depth=18)
    RF_classifier.fit(X_train, y_train)
    predicted_train = RF_classifier.predict(X_train)
    predicted_test = RF_classifier.predict(Y_train)
    print(accuracy_score(predicted_train, Y_train))
    print(accuracy_score(predicted_test, Y_test))
    print(accuracy_score(X_train, Y_train))
    print(accuracy_score(X_test, Y_test))


def XGB_train(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    XGB_classifier = XGBClassifier(
        n_estimators=300, max_depth=70, learning_rate=1)
    XGB_classifier.fit(X_train, y_train)
    predicted_train = XGB_classifier.predict(X_train)
    predicted_test = XGB_classifier.predict(Y_train)
    print(accuracy_score(predicted_train, Y_train))
    print(accuracy_score(predicted_test, Y_test))
    print(accuracy_score(X_train, Y_train))
    print(accuracy_score(X_test, Y_test))
