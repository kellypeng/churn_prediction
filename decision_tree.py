# Decision Tree module for ML Case Study

import numpy as np
import pandas as pd
import pydotplus
from main import load_data, data_processing, drop_date
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

def get_X_and_y(df):
    y = np.array(df.pop('churn'))
    X = np.array(df)
    return X, y

def decision_tree(X, y):
    model = DecisionTreeClassifier()
    model = model.fit(X,y)
    return model

def print_tree(model):
    dot_data = export_graphviz(model, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf('DecisionTree.pdf')


if __name__ == '__main__':
    df = load_data()
    df = data_processing(df)
    df = drop_date(df)

    X, y = get_X_and_y(df)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
    model = decision_tree(X_train, y_train)

    y_pred = model.predict(X_test)
    f1score = f1_score(y_test, y_pred)
    print "F1 score: {}".format(f1score)

    print_tree(model)
