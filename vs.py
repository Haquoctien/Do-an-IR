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

# preprocessing
import string
inchar = '#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r\x0b\x0c' + '“”'
outchar = ' '*len(inchar)
translation = str.maketrans(inchar, outchar)
def prep(text):
    text = text.replace(string.punctuation + "“”" + '\n', ' ')
    text = text.translate(translation).lower()
    return text

# class to load data and search
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
        '''
        query: csr sparse matrix, query vector
        topN: int, number of top ranking results to return
        -> indices of top ranking results in corpus
        '''
        distances = linear_kernel(self.matrix, query)
        distances = distances.flatten()
        ind = np.argpartition(distances, -topN)[-topN:]
        ind = ind[np.argsort(distances[ind])]   
        return np.flip(ind)
    
    def search(self, query, topN):
        '''
        query: string, contains query
        topN: int, number of top ranking results to return
        -> top ranking results of type pandas.DataFrame, index is doc id,
        one collumn 'text' contains doc's text
        '''
        query = self.vectorizer.transform([query])
        ind = self._search(query, topN)
        return self.data.iloc[ind]
    
    def reweightQuery(self, query):
        '''
        query: string contains query
        -> reformulated query vector of type np.array
        '''
        q = self.vectorizer.transform([query])
        topTen = self.matrix[self._search(q, 10)]
        relevantVector = np.sum(topTen,axis=0)/10
        newQueryVector = q + relevantVector
        return newQueryVector
    
    def searchWithReweightedQuery(self, query, topN):
        '''
        query: string, contains query
        topN: int, number of top ranking results to return
        -> top ranking results of type pandas.DataFrame, index is doc id,
        one collumn 'text' contains doc's text
        '''
        newQueryVector = self.reweightQuery(query)
        return self.data.iloc[self._search(newQueryVector, topN)]
    
    def expandQuery(self, query):
        '''
        '''
        q = self.vectorizer.transform([query])
        topTen = self.matrix[self._search(q, 10)]
        ind = []
        for doc in topTen:
            ind = np.append(ind,
                    np.argpartition(np.array(doc.todense())[0], -3)[-3:])
        ind = np.asanyarray(ind, dtype=int)
        relevantTerms = np.array(self.vectorizer.get_feature_names())[ind]
        expandedTerms = np.unique(np.append(
                                    self.vectorizer.tokenizer(query),relevantTerms))
        expandedQuery = ' '.join(expandedTerms)
        return expandedQuery
    
    def searchWithExpandedQuery(self, query, topN):
        '''
        '''
        new_query = self.expandQuery(query)
        new_query = self.vectorizer.transform([query])
        ind = self._search(new_query, topN)
        return self.data.iloc[ind]
        
        