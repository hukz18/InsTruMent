import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split


def svm_param_select(X, Y, n_folds=3):
    Cs = [0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    kernels = ['rbf', 'sigmoid']
    param_grid = {'C': Cs, 'gamma': gammas, 'kernel': kernels}
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=n_folds)
    grid_search.fit(X, Y)
    return grid_search.best_params_


def svm_train(X, Y):
    X = np.mean(X, axis=2)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    param = svm_param_select(X_train, Y_train)
    print(param)
    svm_classifier = svm.SVC(**param)
    svm_classifier.fit(X_train, Y_train)
    predicted_train = svm_classifier.predict(X_train)
    predicted_test = svm_classifier.predict(X_test)
    print(accuracy_score(predicted_train, Y_train))
    print(accuracy_score(predicted_test, Y_test))
    return svm


def svm_predict(x, labels, classifiers):
    pass


def RF_train(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    RF_classifier = RandomForestClassifier(n_estimators=500, random_state=1993, max_depth=18)
    RF_classifier.fit(X_train, y_train)
    predicted_train = RF_classifier.predict(X_train)
    predicted_test = RF_classifier.predict(Y_train)
    print(accuracy_score(predicted_train, Y_train))
    print(accuracy_score(predicted_test, Y_test))
    print(accuracy_score(X_train, Y_train))
    print(accuracy_score(X_test, Y_test))


def XGB_train(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    XGB_classifier = XGBClassifier(n_estimators=300, max_depth=70, learning_rate=1)
    XGB_classifier.fit(X_train, y_train)
    predicted_train = XGB_classifier.predict(X_train)
    predicted_test = XGB_classifier.predict(Y_train)
    print(accuracy_score(predicted_train, Y_train))
    print(accuracy_score(predicted_test, Y_test))
    print(accuracy_score(X_train, Y_train))
    print(accuracy_score(X_test, Y_test))
