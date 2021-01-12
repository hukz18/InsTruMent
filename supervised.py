import seaborn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import svm
import matplotlib.pyplot as plt
from feature import get_feature_npy
from evaluation import *
from dataset import get_data_list_weighted
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split


class ML_Classifier:
    def __init__(self, clf_type):
        self.labels = ['cel', 'cla', 'flu', 'gac', 'gel',
                       'org', 'pia', 'sax', 'tru', 'vio']
        self.clf_type = clf_type
        self.label_to_idx = {v: i for (i, v) in enumerate(self.labels)}
        self.meta_labels = [['pia', 'gel'], ['sax', 'org', 'cel', 'cla', 'flu', 'gac', 'vio', 'tru']]
        self.discriminators = {}
        self.meta_discriminators = []
        self.class_num = 10
        self.meta_class_num = 2

    def add_meta_clf(self, features, labels):
        if self.clf_type == 'SVM':
            sub_clf = train_SVM(features, labels)
        elif self.clf_type == 'RF':
            sub_clf = train_RF(features, labels)
        elif self.clf_type == 'XGB':
            sub_clf = train_XGB(features, labels)
        else:
            raise ValueError('choose clf_type from SVM, RF(random forest), or XGB(XGBoost)')
        self.meta_discriminators.append(sub_clf)

    def add_clf(self, features, labels, label):
        if self.clf_type == 'SVM':
            sub_clf = train_SVM(features, labels)
        elif self.clf_type == 'RF':
            sub_clf = train_RF(features, labels)
        elif self.clf_type == 'XGB':
            sub_clf = train_XGB(features, labels)
        else:
            raise ValueError('choose clf_type from SVM, RF(random forest), or XGB(XGBoost)')
        self.discriminators[label] = sub_clf

    def predict_meta(self, mfcc):
        mfcc = mfcc[np.newaxis, :]
        feature = get_feature_npy(mfcc)
        result = np.zeros(self.meta_class_num)
        for i in range(self.meta_class_num):
            result[i] = self.meta_discriminators[i].predict(feature)
        return result

    def predict_whole(self, mfccs):
        num_item = np.shape(mfccs, 0)
        result = np.zeros(num_item, self.class_num)
        feature = get_feature_npy(mfccs)
        for label in self.labels:
            result[:, self.label_to_idx[label]] = self.discriminators[label].predict(feature)
        return result

    def predict_one(self, mfcc, meta_label=None):
        labels = []
        if not len(self.meta_discriminators):
            labels = self.labels
        else:
            for i in range(self.meta_class_num):
                labels += self.meta_labels[i] if meta_label[i] else []
        mfcc = mfcc[np.newaxis, :]
        result = np.zeros(self.class_num)
        feature = get_feature_npy(mfcc)
        for label in labels:
            result[self.label_to_idx[label]] = self.discriminators[label].predict(feature)
        return result

    def evaluate(self, test_iter):
        true_labels, predict_labels = [], []
        for mfcc, labels in tqdm(test_iter):
            true_label = np.zeros(self.class_num)
            # meta_label = self.predict_meta(mfcc)
            meta_label = None
            for label in labels:
                true_label[self.label_to_idx[label]] = True
            true_labels.append(true_label)
            # predict_labels.append(self.predict_one(mfcc, meta_label))
            predict_labels.append(self.predict_one(mfcc, meta_label))
        predict_labels = np.array(predict_labels)
        true_labels = np.array(true_labels)
        print('accuracy score: %.3f' % accuracy_score(true_labels, predict_labels))
        print('recall score: %.3f' % recall_score(true_labels.reshape(-1, 1), predict_labels.reshape(-1, 1)))
        print('precision score: %.3f' % precision_score(true_labels.reshape(-1, 1), predict_labels.reshape(-1, 1)))
        print('f1 score: %.3f' % f1_score(true_labels.reshape(-1, 1), predict_labels.reshape(-1, 1)))
        confusion = co_est_mat(np.mat(true_labels), np.mat(predict_labels))
        confusion = pd.DataFrame(confusion, columns=self.labels, index=self.labels)
        seaborn.heatmap(confusion, annot=True)
        plt.show()


def svm_param_select(X, Y, n_folds=3):
    # Cs = [0.01, 0.1, 1, 10]
    Cs = [10, 20, 50, 100, 200]
    gammas = ['scale', 0.0003, 0.0001, 0.0005, 0.0008, 0.001]
    kernels = ['rbf']
    param_grid = {'C': Cs, 'gamma': gammas, 'kernel': kernels}
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=n_folds)
    grid_search.fit(X, Y)
    return grid_search.best_params_


def train_SVM(X, Y):
    # X = np.mean(X, axis=2)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    param = svm_param_select(X_train, Y_train)
    print(param)
    svm_classifier = svm.SVC(**param)
    svm_classifier.fit(X_train, Y_train)
    predicted_train = svm_classifier.predict(X_train)
    predicted_test = svm_classifier.predict(X_test)
    print('train acc:%.2f, test acc:%.2f' % (
        accuracy_score(predicted_train, Y_train), accuracy_score(predicted_test, Y_test)))
    return svm_classifier


def RF_param_select(X, Y, n_folds=3):
    criterion = ['gini', 'entropy']
    n_estimators = [20, 50, 100, 200]
    max_depth = [15, 18, 21]
    param_grid = {'n_estimators': n_estimators, 'criterion': criterion, 'max_depth': max_depth}
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=n_folds)
    grid_search.fit(X, Y)
    return grid_search.best_params_


def train_RF(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    param = RF_param_select(X_train, Y_train)
    print(param)
    RF_classifier = RandomForestClassifier(n_jobs=-1, **param)
    RF_classifier.fit(X_train, y=Y_train)
    predicted_train = RF_classifier.predict(X_train)
    predicted_test = RF_classifier.predict(X_test)
    print('train acc:%.2f, test acc:%.2f' % (
        accuracy_score(predicted_train, Y_train), accuracy_score(predicted_test, Y_test)))
    return RF_classifier


def XGB_param_select(X, Y, n_folds=3):
    n_estimators = [100, 300, 500]
    max_depth = [50, 70, 90]
    min_child_weight = [1, 1.5, 2]
    param_grid = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_child_weight': min_child_weight}
    grid_search = GridSearchCV(XGBClassifier(), param_grid, cv=n_folds)
    grid_search.fit(X, Y)
    return grid_search.best_params_


def train_XGB(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    param = XGB_param_select(X_train, Y_train)
    print(param)
    XGB_classifier = XGBClassifier(verbosity=1, learning_rate=1, **param)
    XGB_classifier.fit(X_train, Y_train)
    predicted_train = XGB_classifier.predict(X_train)
    predicted_test = XGB_classifier.predict(X_test)
    print('train acc:%.2f, test acc:%.2f' % (
        accuracy_score(predicted_train, Y_train), accuracy_score(predicted_test, Y_test)))
    return XGB_classifier
