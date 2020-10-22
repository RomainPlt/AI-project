# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 13:34:20 2020

@author: Romain Pialat

FINAL ALGO

"""

from sklearn.datasets import load_files
from sklearn.metrics import confusion_matrix, make_scorer, roc_auc_score, roc_curve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.dummy import DummyClassifier
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm.classes import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from matplotlib import pyplot
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
import pandas as pd
import math
import utils


our_scorer = make_scorer(utils.g_mean_score, greater_is_better=True)


def train_model(X, y):
    """
    Return a trained model.

    Please keep the same arguments: X, y (to be able to import this function for evaluation)
    """
    assert 'X' in locals().keys()
    assert 'y' in locals().keys()
    assert len(locals().keys()) == 2

    sc = StandardScaler()
    clf = svm.SVC()
    pipe = Pipeline(steps=[
        ('sc', sc),
        ('clf', clf)])
    pipe.fit(X, y)
    return pipe


sc = StandardScaler()
    

SELECT_STAGE = 0  
NR_SEG = 1

df = utils.load_data_stage_last_k_segments(select_stage=SELECT_STAGE, nr_seg=NR_SEG)  
df = df.fillna(method = 'ffill')
df_train, df_test = utils.split_train_test_based_on_year(df)

    
X_train, y_train, _ = utils.get_X_y_from_dataframe(df_train)
X_test, y_test, _ = utils.get_X_y_from_dataframe(df_test)

def predict(model1, X):
    """
    Produce predictions for X using given filter.
    Please keep the same arguments: X, y (to be able to import this function for evaluation)
    """
    assert len(locals().keys()) == 2

    return model1.predict(X)


DTC = DecisionTreeClassifier(criterion='entropy', splitter = 'random')
steps = [('sc', sc), ('clf', DTC)]
pipe = Pipeline(steps)
pipe.fit(X_train, y_train)
y_tr_pred = predict(pipe, X_train)
print('\ng-mean on training data: ', utils.g_mean_score(y_train, y_tr_pred))
y_tst_pred = predict(pipe, X_test)
print('g-mean on test data: ', utils.g_mean_score(y_test, y_tst_pred))






