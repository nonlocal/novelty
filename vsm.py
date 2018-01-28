"""

In this module we are going to code a very simple Outliers Detection system, in perticular Novelty Detection
For reference, visit http://scikit-learn.org/stable/modules/outlier_detection.html

Author: Nilesh Chaudhari(github:nonlocal)
Date : 28/Jan/18/Sun
Copyright (c) 2018-* Nilesh Chaudhari aka nonlocal.
All Rights Reserved.

"""
from __future__ import division
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models.word2vec import Word2Vec
from data import data_model
import numpy as np
# This module contains all types of Vector Space Modellings (VSMs i.e. feature representation)
# possible, relevant to our case.
# Has Following VSMs:
#   1. Count Vectorizer
#   2. TF-IDF
#   3. word2vec

def bar():
    print ("bar..")

class Tfidf(TfidfVectorizer):
    """docstring for Tdi."""
    def __init__(self, documents):
        super(Tfidf, self).__init__()
        self.documents = documents
        self.preprocess = data_model.preprocess
        self.train()
    def train(self):
        documents = [self.preprocess(doc) for doc in self.documents]
        self.fit(documents)
        print "TFIDF fitted to given documents"
    def vectorize_doc(self, doc):
        doc = self.preprocess(doc)
        return self.transform([doc]).toarray()
    def vectorize_docs(self, docs):
        docs = [self.preprocess(doc) for doc in docs]
        return self.transform(docs).toarray()


class CountVect(CountVectorizer):
    """docstring for CountVect."""
    def __init__(self, documents):
        super(CountVect, self).__init__()
        self.documents = documents
        self.preprocess = data_model.preprocess
        self.train()
    def train(self):
        documents = [self.preprocess(doc) for doc in self.documents]
        self.fit(documents)
        print "CountVectorizer fitted to given documents"

    def vectorize_doc(self, doc):
        doc = self.preprocess(doc)
        return self.transform([doc]).toarray()
    def vectorize_docs(self, docs):
        docs = [self.preprocess(doc) for doc in docs]
        return self.transform(docs).toarray()


class W2V(object):
    """docstring for Word2vec."""
    def __init__(self, documents):
        super(W2V, self).__init__()
        self.documents = documents
        self.preprocess = data_model.preprocess
        self.model = None
        self.train()
    def train(self):
        #pass documents as sentences to word2vec; really bad idea but what else to do?
        documents = [self.preprocess(doc).split() for doc in self.documents]
        self.model = Word2Vec(sentences=documents)
        print "Word2Vec Model trained!"
    def vectorize_doc(self, doc):
        tokens = self.preprocess(doc).split()
        doc_repr = np.array([self.model.wv[token] for token in tokens if token in self.model.wv.vocab])
        vector = doc_repr.mean(axis=0)
        return vector
    def vectorize_docs(self, docs):
        vectors = np.array([self.vectorize_doc(doc) for doc in docs])
        return vectors
