# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 17:24:59 2020

@author: Romain Pialat
"""

#%% Data Representation


from sklearn.datasets import load_files
from sklearn.metrics import confusion_matrix, make_scorer
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
from sklearn.model_selection import learning_curve

TR_DATA = 'spam-data/spam-data-1'
TST_DATA = 'spam-data/spam-data-2'

data_tr = load_files(TR_DATA, encoding='utf-8')
X_train = data_tr.data
y_train = data_tr.target

data_tst = load_files(TST_DATA, encoding='utf-8')
X_test = data_tst.data
y_test = data_tst.target


def prediction(filter1, X):
    return filter1.predict(X)
    
def modified_accuracy(y, y_pred):
    """Return a modified accuracy score with larger weight of false positives."""
    cm = confusion_matrix(y, y_pred)
    if cm.shape != (2, 2):
        raise ValueError('The ground truth values and the predictions may contain at most 2 values (classes).')
    return (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + 10 * cm[0, 1] + cm[1, 0])

#%%  Model Tuning
    

vec = CountVectorizer()
clf = [
    KNeighborsClassifier(),
    svm.classes.SVC(gamma = 'scale'),
    DecisionTreeClassifier(),
    MultinomialNB(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    
    ]


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




params = [parametersKNN, parametersSVM, parametersDTC, parametersMNB,
          parametersRF, parametersADA]

classifier_name= ['KNN', 'SVC', 'Decision Tree', 'MultinomialNB', 'Random Forest', 'Adaboost']

param_clf = []

for i in range(len(params)):
    param_clf.append((classifier_name[i], clf[i],params[i]))
    
best_param = []

for name, classifier,param in param_clf: 
    print(name, '\n')
    steps = [('vectorizer', vec), ('clf', classifier)]
    pipe = Pipeline(steps)
    pipe.fit(X_train, y_train) 
    gs_clf = GridSearchCV(pipe, param, cv = 5).fit(X_train, y_train)
    best_param.append(gs_clf.best_params_)


print(best_param)


#%% Learning Curve

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

vec = CountVectorizer()
clf = [
    KNeighborsClassifier(2),
    svm.classes.SVC(C = 8, gamma = 0.0001, kernel = 'rbf'),
    DecisionTreeClassifier(criterion = 'entropy', splitter = 'best'),
    MultinomialNB(alpha=0.01),
    RandomForestClassifier(criterion='entropy', n_estimators=50),
    AdaBoostClassifier(algorithm='SAMME.R', 
                       base_estimator = DecisionTreeClassifier(criterion = 'gini', max_depth = 2), 
                       learning_rate = 1, n_estimators = 80)
    ]

classifier_name= ['KNN', 'SVC', 'Decision Tree', 'MultinomialNB', 'Random Forest', 'Adaboost']

param_clf = []

for i in range(len(clf)):
    param_clf.append((classifier_name[i], clf[i]))

for name, classifier in param_clf : 
    print(name)
    steps = [('vectorizer', vec), ('clf', classifier)]
    pipe = Pipeline(steps)
    
    train_sizes, train_scores, test_scores = learning_curve(pipe, 
                                                        X_train, 
                                                        y_train,
                                                        cv=10,
                                                        scoring='accuracy',
                                                        n_jobs=-1, 
                                                        train_sizes=np.linspace(0.1, 1, 60))
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



#%% Modified accuracy 
    
MNB = MultinomialNB(alpha=0.01)
vec = CountVectorizer()
 
steps = [('vectorizer', vec), ('clf', MNB)]
pipe = Pipeline(steps)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print("MNB modified accuracy : ", modified_accuracy(y_test, y_pred))


ADA = AdaBoostClassifier(algorithm='SAMME.R', 
                       base_estimator = DecisionTreeClassifier(criterion = 'gini', max_depth = 2), 
                       learning_rate = 1, n_estimators = 80)

vec = CountVectorizer()
 
steps = [('vectorizer', vec), ('clf', ADA)]
pipe = Pipeline(steps)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print("ADA modified accuracy : ", modified_accuracy(y_test, y_pred))

#%% MLPClassifier

parametersMLP = {'clf__hidden_layer_sizes' : [(100,), (50,), (150,)],
                'clf__activation' : ['relu', 'logistic'],
                'clf__solver' : ['lbfgs'],
                'clf__alpha' : [0.001,0.01,0.1],
                'clf__learning_rate' : ['constant', 'invscaling']}


clf1 = MLPClassifier(hidden_layer_sizes=50, activation= 'relu', alpha=0.1, learning_rate='constant')

steps = [('vectorizer', vec), ('clf', clf1)]
pipe = Pipeline(steps)
pipe.fit(X_train, y_train) 
y_pred = pipe.predict(X_test)

print("NN  modified accuracy : ", modified_accuracy(y_test, y_pred))



