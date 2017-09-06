import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from main import load_data, data_processing, drop_date
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def get_X_and_y(df):
    y = np.array(df.pop('churn'))
    X = np.array(df)
    return X, y

def score(model, y_true, y_pred):
    f1 =  f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return model.__class__.__name__, \
         'F1: {}, accuracy: {}, precision: {}, recall: {}' \
         .format(f1, accuracy, precision, recall)

def decision_tree(X, y):
    return DecisionTreeClassifier().fit(X,y)

def gd_model(X, y, n_estimators=1000, learning_rate=.2, max_depth=7, subsample=.5):
    return GradientBoostingClassifier(subsample=subsample, max_depth=max_depth,\
           learning_rate=learning_rate).fit(X, y)

def rf_model(X, y, n_estimators=1000, min_samples_split=2, min_samples_leaf=4, \
             max_features='sqrt'):
    return RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, \
           min_samples_leaf=min_samples_leaf, max_features=max_features).fit(X, y)

def adb_model(X, y, n_estimators=1000, learning_rate=0.1):
    adb = AdaBoostClassifier(n_estimators=n_estimators,learning_rate=learning_rate)
    return adb.fit(X, y)

def knn(X, y, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    return knn.fit(X, y)

def plot_imp(model, X):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure()
    plt.bar(range(X.shape[1]), importances[indices], color="b", align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])

def gd_gridsearch():
    gradient_boosting_grid = {'learning_rate': [.1, .2, .3],
                              'max_depth': [3, 5, 7, 9],
                              'n_estimators': [100, 1000, 5000],
                              'subsample': [.3, .5, .7, .9],
                              'random_state': [1]}
    gb_gridsearch = GridSearchCV(GradientBoostingClassifier(),
                                 gradient_boosting_grid,
                                 n_jobs=-1,
                                 verbose=True,
                                 scoring='f1')
    gb_gridsearch.fit(X_train, y_train)
    gb_gridsearch.best_params_

def rf_gridsearch():
    random_forest_grid = {'max_depth': [3, 5, 7, None],
                      'max_features': ['sqrt', 'log2', None],
                      'min_samples_split': [2, 4],
                      'min_samples_leaf': [1, 2, 4],
                      'bootstrap': [True, False],
                      'n_estimators': [10, 50, 100, 1000],
                      'random_state': [1]}
    rf_gridsearch = GridSearchCV(RandomForestClassifier(),
                                 random_forest_grid,
                                 n_jobs=-1,
                                 verbose=True,
                                 scoring='f1')
    rf_gridsearch.fit(X_train, y_train)
    rf_gridsearch.best_params



if __name__ == '__main__':
    df = load_data('data/churn_train.csv')
    df = data_processing(df)
    df = drop_date(df)
    X, y = get_X_and_y(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
    # DT
    dt = decision_tree(X_train, y_train)
    dt_pred = dt.predict(X_test)
    print score(dt, y_test, dt_pred)
    # GD
    gd = gd_model(X_train, y_train)
    gd_pred = gd.predict(X_test)
    print score(gd, y_test, gd_pred)
    # RF
    rf = rf_model(X_train, y_train)
    rf_pred = rf.predict(X_test)
    print score(rf, y_test, rf_pred)
    # adaboost
    adb = adb_model(X_train, y_train)
    adb_pred = adb.predict(X_test)
    print score(adb, y_test, adb_pred)
    # kNN
    knn = knn(X_train, y_train)
    knn_pred = knn.predict(X_test)
    print score(knn, y_test, knn_pred)

    '''
    - DecisionTreeClassifier:
      F1: 0.76205162588, accuracy: 0.70425,
      precision: 0.768596159048, recall: 0.755617604042
    - GradientBoostingClassifier:
      F1: 0.829737457759, accuracy: 0.781666666667,
      precision: 0.811491038515, recall: 0.848823294775
    - RandomForestClassifier:
      F1: 0.836654918295, accuracy: 0.787583333333,
      precision: 0.807521029193, recall: 0.867969684882
    - AdaBoostClassifier:
      F1: 0.83569823435, accuracy: 0.78675
      precision: 0.808045691582, recall: 0.865310464034
    - KNeighborsClassifier:
      F1: 0.798421529305, accuracy: 0.740333333333
      precision: 0.777497795137, recall: 0.82050259274
    '''

    # Plot feature importance
    plot_imp(gd, X_train)
    plt.title("Feature importances - Gradient Boosting")
    plt.show()

    plot_imp(dt, X_train)
    plt.title("Feature importances - Decision Tree")
    plt.show()

    plot_imp(rf, X_train)
    plt.title("Feature importances - Random Forest")
    plt.show()

    # best features for RF
    '''
    {'bootstrap': True,
     'max_depth': None,
     'max_features': 'sqrt',
     'min_samples_leaf': 4,
     'min_samples_split': 2,
     'n_estimators': 80,
     'random_state': 1}
     '''

     print 'RF:'
     print confusion_matrix(y_test, rf_pred)
     print 'ADB:'
     print confusion_matrix(y_test, adb_pred)

    # Run on test set
    test_df = load_data('data/churn_test.csv')
    test_df = data_processing(test_df)
    test_df = drop_date(test_df)
    test_X, test_y = get_X_and_y(test_df)
