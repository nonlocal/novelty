"""

In this module we are going to code a very simple Outliers Detection system, in perticular Novelty Detection
For reference, visit http://scikit-learn.org/stable/modules/outlier_detection.html

Author: Nilesh Chaudhari(github:nonlocal)
Date : 28/Jan/18/Sun
Copyright (c) 2018-* Nilesh Chaudhari aka nonlocal.
All Rights Reserved.

"""
from __future__ import division
from sklearn.datasets import fetch_20newsgroups
from collections import Counter
import numpy as np
import re

def zoo():
    print "zoo..."

domain_categories = categories_train = ['comp.graphics', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x']
ood_categories = out_of_domain = ['alt.atheism', 'comp.os.ms-windows.misc', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

class DataModel(object):
    """docstring for DataGenerator."""

    def __init__(self, domain_categories=domain_categories, ood_categories=ood_categories):
        self.domain_categories = domain_categories
        self.ood_categories = ood_categories

    def preprocess(self, doc):
        return " ".join(re.findall(r'\w+', doc.lower()))

    def _train_data(self, domain_categories=None):
        if domain_categories is None:
            domain_categories = self.domain_categories
        corpora = fetch_20newsgroups(subset='train', categories=domain_categories)['data']
        return [self.preprocess(doc) for doc in corpora]

    def _test_data(self, domain_categories=None):
        if domain_categories is None:
            domain_categories = self.domain_categories
        corpora = fetch_20newsgroups(subset='test', categories=domain_categories)['data']
        return [self.preprocess(doc) for doc in corpora]

    def _ood_data(self, ood_categories=None):
        if ood_categories is None:
            ood_categories = self.ood_categories
        corpora = fetch_20newsgroups(categories=ood_categories)['data']
        return [self.preprocess(doc) for doc in corpora]
    def load_data(self):
        return self._train_data(), self._test_data(), self._ood_data()

data_model = DataModel()
