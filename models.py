"""

In this module we are going to code a very simple Outliers Detection system, in perticular Novelty Detection
For reference, visit http://scikit-learn.org/stable/modules/outlier_detection.html

Author: Nilesh Chaudhari(github:nonlocal)
Date : 28/Jan/18/Sun
Copyright (c) 2018-* Nilesh Chaudhari aka nonlocal.
All Rights Reserved.

"""
from __future__ import division
import numpy as np
import time
def foo():
    print ("foo...")

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# This module contains classes for "models"
# Contains followind models:
#   1. One Class SVM model
#   2. KMeans
#   3. Custom Model based on Cosine Similarity
#   4. Dimensionality reduction and then One class SVM
#   5. tSNE and then one class SVM
class OneClassBase(object):
    """docstring for OneClassBase."""
    def __init__(self, kernel='poly'):
        super(OneClassBase, self).__init__()
        self.kernel = kernel
        self.model = OneClassSVM#(kernel=self.kernel)
        self.scaler = StandardScaler#()
        self.is_fitted = False

class SVMNovelty(OneClassBase):
    """docstring for SVMNovelty."""

    def __init__(self):
        super(SVMNovelty, self).__init__()
        self.input_shape = None
        self.model = self.model(kernel=self.kernel)
        self.scaler = self.scaler()
    def __name__(self):
        return "OneClassSVM_with_{}".format(self.kernel)

    def train_model(self, X):
        self.is_fitted = True
        start_time = time.time()
        print "Scaling features with StandardScaler..."
        X = self.scaler.fit_transform(X)#.toarray()
        self.input_shape = X.shape
        print "Training OneClassSVM with '{}' kernel...".format(self.kernel)
        self.model.fit(X)
        print "Time taken: {}s".format(time.time()-start_time)

    def infer(self, X):
        if not self.is_fitted:
            raise NotImplementedError("Model not fitted yet...")

        start_time = time.time()
        if len(X)!=len(self.input_shape) and X.shape[0] == self.input_shape[1]:
            X = X[np.newaxis, :]
        elif X.shape[1] == self.input_shape[1]:
            pass
        else:
            raise ValueError("Shape mismatch...")
        #scale features at predict time

        X = self.scaler.transform(X)#.toarray()
        predictions = self.model.predict(X)
        print "Time taken: {}s...".format(time.time()-start_time)
        return predictions
