#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:56:16 2020

@author: hqt98
"""

# load data
import os
root = 'news_dataset'

filepaths = [os.path.join(root,i) for i in os.listdir(root)]

corpus = dict()
for path in filepaths:
    fp = open(path,'r',errors='ignore', encoding='utf-16')
    text = fp.read()
    corpus[os.path.basename(path)[:-4]] = text

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

# vectorize corpus
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(preprocessor=prep, tokenizer=tokenize,
                             analyzer='word', max_df=0.7, norm='l2',
                             use_idf=True, smooth_idf=True, sublinear_tf=True)
matrix = vectorizer.fit_transform(corpus.values())
# convert corpus into dataframe
import pandas as pd
data = pd.DataFrame.from_dict(corpus, orient='index', columns=['text'])
# save trained vectorizer and matrix
import pickle
with open('corpus.pickle', 'wb') as f:
    pickle.dump(corpus, f)
with open('data.pickle', 'wb') as f:
    pickle.dump(data, f)
with open('vectorizer.pickle', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('matrix.pickle', 'wb') as f:
    pickle.dump(matrix, f)