#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 21:10:30 2020

@author: hqt98
"""
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import pickle

# tokenizer
from underthesea import word_tokenize
def tokenize(text):
    return word_tokenize(text)

# preprocessing
import string
def prep(text):
    text = text.replace('\n', ' ').strip(string.punctuation + "“”")
    translation =str.maketrans('', '', string.punctuation + '“”')
    text = text.translate(translation).lower()
    return text

class VectorSpaceIRModel():
    
    def __init__(self, corpus_path, data_path, vectorizer_path, matrix_path):
        with open(corpus_path, 'rb') as f:
            self.corpus = pickle.load(f)
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(matrix_path, 'rb') as f:
            self.matrix = pickle.load(f)
            
    def _search(self, query, topN):
        distances = linear_kernel(self.matrix, query)
        distances = distances.flatten()
        ind = np.argpartition(distances, -topN)[-topN:]
        ind = ind[np.argsort(distances[ind])]   
        return np.flip(ind)
    
    def search(self, query, topN):
        query = self.vectorizer.transform([query])
        ind = self._search(query, topN)
        return self.data.iloc[ind]
    
    def expandQuery(self, query):
        # use blind feedback
        q = self.vectorizer.transform([query])
        topTen = self.matrix[self._search(q, 10)]
        relevantVector = np.sum(topTen,axis=0)/10
        newQueryVector = q + relevantVector
        return newQueryVector
    
    def searchWithExpandedQuery(self, query, topN):
        newQueryVector = self.expandQuery(query)
        return self.data.iloc[self._search(newQueryVector, topN)]