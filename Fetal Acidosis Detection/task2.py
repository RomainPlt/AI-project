# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 04:01:49 2020

@author: Romain
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

clf = [
    KNeighborsClassifier(),
    svm.classes.SVC(gamma = 'scale'),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    
    ]
    

SELECT_STAGE = 0  
NR_SEG = 1

df = utils.load_data_stage_last_k_segments(select_stage=SELECT_STAGE, nr_seg=NR_SEG)  
df = df.fillna(method = 'ffill')
df_train, df_test = utils.split_train_test_based_on_year(df)



    
X_train, y_train, _ = utils.get_X_y_from_dataframe(df_train)
X_test, y_test, _ = utils.get_X_y_from_dataframe(df_test)


#%% GRIDSEARCH


parametersKNN = {'clf__n_neighbors' : [1,2,3,4,5,6,7]}

parametersSVM = {
    'clf__kernel' : ['rbf'],
    'clf__C': [1,2,3,4,5,6,7,8,9,10],
    'clf__gamma': [0.0001,0.001,0.01,0.1,1]
}

parametersDTC = { 'clf__criterion' : ['gini', 'entropy'],
                 'clf__splitter' : ['best', 'random'] }

parametersMNB = {'clf__alpha' : [0,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]}


parametersRF = {'clf__n_estimators' : [50,100,150],
                'clf__criterion' : ['gini', 'entropy']}

parametersADA = [
        {'clf__algorithm' : ['SAMME'], 'clf__base_estimator' : [DecisionTreeClassifier(),
                                                          RandomForestClassifier(n_estimators=50)],
                'clf__n_estimators' : [50,60,70,80,90,100,120,150],
                'clf__learning_rate' : [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]},
        {'clf__algorithm' : ['SAMME.R'],'clf__base_estimator' : [DecisionTreeClassifier(), 
                                     DecisionTreeClassifier(max_depth=2),
                                     RandomForestClassifier(n_estimators=50)],
                'clf__n_estimators' : [50,60,70,80,90,100,120,150],
                'clf__learning_rate' : [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]}]




params = [parametersKNN, parametersSVM, parametersDTC,
          parametersRF, parametersADA]

classifier_name= ['KNN', 'SVC', 'Decision Tree', 'Random Forest', 'Adaboost']

param_clf = []

for i in range(len(params)):
    param_clf.append((classifier_name[i], clf[i],params[i]))
    
best_param = []

for name, classifier,param in param_clf: 
    print(name, '\n')
    clf = svm.SVC()
    pipe = Pipeline(steps=[
        ('sc', sc),
        ('clf', classifier)])
    pipe.fit(X_train, y_train) 
    gs_clf = GridSearchCV(pipe, param, cv = 5).fit(X_train, y_train)
    best_param.append(gs_clf.best_params_)


print(best_param)


#%%%

def plot_learning_curve_error(tr_sizes, tr_errors, tst_errors, tr_color='lime', tst_color='cyan'):
    """Plot the learning curve from pre-computed data."""

    fig, ax = plt.subplots()

    ax.plot(tr_sizes, tr_errors, lw=2, c=tr_color, label='training error')
    ax.plot(tr_sizes, tst_errors, lw=2, c=tst_color, label='cross validation '
                                                           'error')
    ax.set_xlabel('training set size')
    ax.set_ylabel('error')

    ax.legend(loc=0)
    ax.set_xlim(0, np.max(tr_sizes))
    ax.set_ylim(0, 1)
    ax.set_title('Learning Curve for a model (fixed parameters)')
    ax.grid(True)


def plot_learning_curve_accuracy(train_sizes, tr_mean, tst_mean, tr_std, tst_std):
    plt.plot(train_sizes, tr_mean, c="r",  label="Training score")
    plt.plot(train_sizes, tst_mean, color="orange", label="Cross-validation score")

    # Draw bands
    plt.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std, color="tomato")
    plt.fill_between(train_sizes, tst_mean - tst_std, tst_mean + tst_std, color="navajowhite")
                                      
    # Create plot
    plt.title("Learning Curve ")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

clf = [
    KNeighborsClassifier(2),
    svm.classes.SVC(C = 8, gamma = 0.0001, kernel = 'rbf'),
    DecisionTreeClassifier(criterion = 'entropy', splitter = 'best'),
    RandomForestClassifier(criterion='entropy', n_estimators=50)
    ]

classifier_name= ['KNN', 'SVC', 'Decision Tree', 'Random Forest']

param_clf = []
sc = StandardScaler()

for i in range(len(clf)):
    param_clf.append((classifier_name[i], clf[i]))

for name, classifier in param_clf : 
    print(name)
    steps = [('sc', sc), ('clf', classifier)]
    pipe = Pipeline(steps)
    
    train_sizes, train_scores, test_scores = learning_curve(pipe, 
                                                        X_train, 
                                                        y_train,
                                                        cv=10,
                                                        scoring='accuracy',
                                                        n_jobs=-1, 
                                                        train_sizes=np.linspace(0.01, 1, 60))
    tr_mean = np.mean(train_scores, axis=1)
    tr_std = np.std(train_scores, axis=1)
    tst_mean = np.mean(test_scores, axis=1)
    tst_std = np.std(test_scores, axis=1)
    tr_errors = 1 - tr_mean
    tst_errors = 1 - tst_mean
    
    
    
    plt.figure()
    plot_learning_curve_error(train_sizes, tr_errors, tst_errors)
    
    plt.figure()
    plot_learning_curve_accuracy(train_sizes, tr_mean, tst_mean, tr_std, tst_std)

#%% ROC curves
X_tr = X_train[0:2500]
y_tr = y_train[0:2500]
model = svc = svm.SVC(C = 1, gamma = 0.0001, kernel = 'rbf', probability=True)
model.fit(X_tr, y_tr)
# predict probabilities

yhat = model.predict_proba(X_test)
# keep probabilities for the positive outcome only
yhat = yhat[:, 1]
# calculate roc curves
fpr, tpr, thresholds = roc_curve(y_test, yhat)
# plot the roc curve for the model
pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
pyplot.plot(fpr, tpr, marker='.', label='SVC')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
# show the plot
pyplot.show()


gmeans = np.sqrt(tpr * (1-fpr))

ix = np.argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))


#%% Modified accuracy 
    
    
def predict(model1, X):
    """
    Produce predictions for X using given filter.
    Please keep the same arguments: X, y (to be able to import this function for evaluation)
    """
    assert len(locals().keys()) == 2

    return model1.predict(X)

sc = StandardScaler()


KNN = KNeighborsClassifier(2)
steps = [('sc', sc), ('clf', KNN)]
pipe = Pipeline(steps)
pipe.fit(X_train, y_train)
y_tr_pred = predict(pipe, X_train)
print('\ng-mean on training data: ', utils.g_mean_score(y_train, y_tr_pred))
y_tst_pred = predict(pipe, X_test)
print('g-mean on test data: ', utils.g_mean_score(y_test, y_tst_pred))

svc = svm.SVC(C = 1, gamma = 0.0001, kernel = 'rbf')
steps = [('sc', sc), ('clf', svc)]
pipe = Pipeline(steps)
pipe.fit(X_train, y_train)
y_tr_pred = predict(pipe, X_train)
print('\ng-mean on training data: ', utils.g_mean_score(y_train, y_tr_pred))
y_tst_pred = predict(pipe, X_test)
print('g-mean on test data: ', utils.g_mean_score(y_test, y_tst_pred))


DTC = DecisionTreeClassifier(criterion='entropy', splitter = 'random')
steps = [('sc', sc), ('clf', DTC)]
pipe = Pipeline(steps)
pipe.fit(X_train, y_train)
y_tr_pred = predict(pipe, X_train)
print('\ng-mean on training data: ', utils.g_mean_score(y_train, y_tr_pred))
y_tst_pred = predict(pipe, X_test)
print('g-mean on test data: ', utils.g_mean_score(y_test, y_tst_pred))


RF = RandomForestClassifier(criterion='gini', n_estimators=50) 
steps = [('sc', sc), ('clf', RF)]
pipe = Pipeline(steps)
pipe.fit(X_train, y_train)
y_tr_pred = predict(pipe, X_train)
print('\ng-mean on training data: ', utils.g_mean_score(y_train, y_tr_pred))
y_tst_pred = predict(pipe, X_test)
print('g-mean on test data: ', utils.g_mean_score(y_test, y_tst_pred))


















