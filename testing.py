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

##   __testing__
from models import foo
from vsm import bar
from data import zoo
foo()
bar()
zoo()
assert type(2/3) == float, "Missing float division"
# __END__


from data import data_model
train, test, ood = data_model.load_data()

from vsm import Tfidf, CountVect, W2V

tfidf = Tfidf(train)
print tfidf.vectorize_doc(test[0]).shape
print tfidf.vectorize_docs(test[:3]).shape

countv = CountVect(train)
print countv.vectorize_doc(test[0]).shape
print countv.vectorize_docs(test[:3]).shape

w2v = W2V(train)
print w2v.vectorize_docs(test[:3]).shape

print "Everything in check so far..."
