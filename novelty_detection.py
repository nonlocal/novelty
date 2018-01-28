"""

In this module we are going to code a very simple Outliers Detection system, in perticular Novelty Detection
For reference, visit http://scikit-learn.org/stable/modules/outlier_detection.html

Author: Nilesh Chaudhari(github:nonlocal)
Date : 28/Jan/18/Sun
Copyright (c) 2018-* Nilesh Chaudhari aka nonlocal.
All Rights Reserved.

"""
from __future__ import division

#freak people out!
import os
if not set(['models.pyc', "novelty_detection.pyc", "vsm.pyc", "testing.pyc"]).intersection(os.listdir(".")):
    print(__doc__)

import numpy as np
from collections import Counter
from models import SVMNovelty
from data import data_model
from vsm import Tfidf, CountVect, W2V


def main():
    #load all kinds of data
    train, test, ood = data_model.load_data()
    #train data(in-domain), test data(in-domain), test data(out-of-domain)

    #import different feature representation/extraction techniques

    #initialize those feature extractors
    # tfidf = Tfidf(train) #eating up memory!!!
    # countv = CountVect(train) #eating up memory!!!
    w2v = W2V(train)
    print "Feature encodings initialised...\n"


    #X represents features and its subscript is how we got it
    # X_tfidf = tfidf.vectorize_docs(train)   #features obtained using tfidf modelling for train dataset
    # X_tfidf_test = tfidf.vectorize_docs(test) #features obtained using tfidf modelling for test dataset
    # X_tfidf_ood = tfidf.vectorize_docs(ood) #features obtained using tfidf modelling for out-of-domain dataset

    #similar notation follows for other VS modellers
    #CountVectorizer
    # X_cv = countv.vectorize_docs(train)
    # X_cv_test = countv.vectorize_docs(test)
    # X_cv_ood = countv.vectorize_docs(ood)

    #Word2Vec
    X_wv = w2v.vectorize_docs(train)
    X_wv_test = w2v.vectorize_docs(test)
    X_wv_ood = w2v.vectorize_docs(ood)

    #create compact references to features
    features = {
                # "tfidf":[X_tfidf, X_tfidf_test, X_tfidf_ood],
                # "countv":[X_cv, X_cv_test, X_cv_ood],
                "word2vec":[X_wv, X_wv_test, X_wv_ood]
                }#use first one for training, other two for testing

    #now we have 3 different kind of features...
    #let's pass'em through OneClassSVM...


    for key, value in features.iteritems():
        X_train, X_test_in, X_test_ood = value
        novelty = SVMNovelty()
        novelty.train_model(X_train)
        print "With {} model and with {} features".format(novelty.__name__(), key)
        print "\nTraining stats: ", Counter(novelty.infer(X_train))
        print "\nTesting stats: ", Counter(novelty.infer(X_test_in))
        print "\nOOD stats: ", Counter(novelty.infer(X_test_ood))

if __name__ == "__main__":
    main()
