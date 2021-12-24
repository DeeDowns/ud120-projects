#!/usr/bin/python3

"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
import numpy as np
from sklearn.naive_bayes import GaussianNB


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


##############################################################
# Enter Your Code Here
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# Y = np.array([1, 1, 1, 2, 2, 2])

# create the classifier
clf = GaussianNB()

# time the training process
t0 = time()
# # < your clf.fit() line of code >
# fit/train the classifier
clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

GaussianNB()

# time the predicting process
t0 = time()
# print the prediction (about 30x faster than training)
print('predict', clf.predict(features_test))
print("Predicting Time:", round(time()-t0, 3), "s")

# print the accuracy score of the model (about 97%)
print('score', clf.score(features_test, labels_test))

##############################################################

##############################################################
'''
You Will be Required to record time for Training and Predicting
The Code Given on Udacity Website is in Python-2
The Following Code is Python-3 version of the same code
'''

# t0 = time()
# # < your clf.fit() line of code >
# print("Training Time:", round(time()-t0, 3), "s")

# t0 = time()
# # < your clf.predict() line of code >
# print("Predicting Time:", round(time()-t0, 3), "s")

##############################################################
