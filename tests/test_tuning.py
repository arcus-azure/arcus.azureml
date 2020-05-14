import os
from arcus.azureml.experimenting.tuning import LocalArcusGridSearchCV
# Imports for the excercise
import pandas as pd 
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model


def test_local_gridsearch():
    df = pd.read_csv('tests/resources/datasets/student-admission.csv')
    y = df.Admission.values
    X = np.asarray(df.drop(['Admission'],axis=1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    df.tail(5)
    logreg = linear_model.LogisticRegression(solver='liblinear')
    param_grid = {'C':[0.1,10]}
    grid = LocalArcusGridSearchCV(logreg, param_grid, scoring='accuracy')
    grid.fit(X_train, y_train)

