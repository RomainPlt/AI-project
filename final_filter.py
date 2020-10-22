# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 03:44:09 2020

@author: Romain Pialat

FINAL SPAM FILTER
"""
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

TR_DATA = 'spam-data/spam-data-1'
TST_DATA = 'spam-data/spam-data-2'

data_tr = load_files(TR_DATA, encoding='utf-8')
X_train = data_tr.data
y_train = data_tr.target

data_tst = load_files(TST_DATA, encoding='utf-8')
X_test = data_tst.data
y_test = data_tst.target

    
def modified_accuracy(y, y_pred):
    """Return a modified accuracy score with larger weight of false positives."""
    cm = confusion_matrix(y, y_pred)
    if cm.shape != (2, 2):
        raise ValueError('The ground truth values and the predictions may contain at most 2 values (classes).')
    return (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + 10 * cm[0, 1] + cm[1, 0])


MNB = MultinomialNB(alpha=0.01)
vec = CountVectorizer()
 
steps = [('vectorizer', vec), ('clf', MNB)]
pipe = Pipeline(steps)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print("MNB modified accuracy : ", modified_accuracy(y_test, y_pred))



