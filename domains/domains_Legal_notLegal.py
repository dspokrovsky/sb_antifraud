# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 00:32:00 2016

@author: Dmitry Pokrovsky
"""

import pandas as pd
import numpy as np
import sklearn.ensemble
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
import math
from collections import Counter
from sklearn import cross_validation
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier


def entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum(count / lns * math.log(count / lns, 2) for count in p.values())

dataframe_dict = {'alexa': [],'bad':[]}
'''dictionary'''
word_dataframe = pd.read_csv('lang_dictionary.txt', names=['word'], header=None, dtype={'word': np.str}, encoding='utf-8')
word_dataframe = word_dataframe[word_dataframe['word'].map(lambda x: str(x).isalpha())]
word_dataframe = word_dataframe.applymap(lambda x: str(x).strip().lower())
word_dataframe = word_dataframe.dropna()
word_dataframe = word_dataframe.drop_duplicates()

'''read data'''
all_domains =pd.DataFrame.from_records(list(map(lambda x: (float(x[0]), x[1][:-1],) ,map( 
                    lambda x: x.split(",", 2), open('train.csv','r').readlines()))), columns=["class", "domain"])

test_domains =pd.DataFrame.from_records(list(map(lambda x: (float(0), x[0][:-1],) ,map( 
                    lambda x: x.split(",", 1), open('test.csv','r').readlines()))), columns=["class", "domain"])

'''разделяем слова обучающей выборки на группы: хорошие, плохие'''
for i in dataframe_dict :
    if i == 'alexa':
        v = all_domains[all_domains['class'] == 1]        
        dataframe_dict[i] = v   
    else:
        v = all_domains[all_domains['class'] ==0]
        dataframe_dict[i] = v
        
dataframe_dict['word'] = word_dataframe['word']

print( 'first end')

''' features '''
all_domains['length'] = [len(x) for x in all_domains['domain']]
test_domains['length'] = [len(x) for x in test_domains['domain']]


all_domains['entropy'] = [entropy(x) for x in all_domains['domain']]
test_domains['entropy'] = [entropy(x) for x in test_domains['domain']]


'''обход memory error для больших ngram'''
alexa_cv_ngram = CountVectorizer(analyzer='char', ngram_range=(3, 4))
counts_matrix = alexa_cv_ngram.fit_transform(dataframe_dict['alexa']['domain'])
alexa_counts = np.log10(counts_matrix.sum(axis=0).getA1())

dictionary_vc = CountVectorizer(analyzer='char', ngram_range=(3, 4))
counts_matrix = dictionary_vc.fit_transform(dataframe_dict['word'])
dictionary_counts = np.log10(counts_matrix.sum(axis=0).getA1())

bad_words_cv = CountVectorizer(analyzer='char', ngram_range=(3, 4))
counts_matrix = bad_words_cv.fit_transform(dataframe_dict['bad']['domain'])
bad_counts = np.log10(counts_matrix.sum(axis=0).getA1())


all_domains['bad_grams'] = bad_counts * bad_words_cv.transform(all_domains['domain']).T
all_domains['alexa_grams'] = alexa_counts * alexa_cv_ngram.transform(all_domains['domain']).T
all_domains['word_grams'] = dictionary_counts * dictionary_vc.transform(all_domains['domain']).T
all_domains['diff'] = all_domains['alexa_grams'] - all_domains['word_grams']

test_domains['bad_grams'] = bad_counts * bad_words_cv.transform(test_domains['domain']).T
test_domains['alexa_grams'] = alexa_counts * alexa_cv_ngram.transform(test_domains['domain']).T
test_domains['word_grams'] = dictionary_counts * dictionary_vc.transform(test_domains['domain']).T
test_domains['diff'] = test_domains['alexa_grams'] - test_domains['word_grams']

print( 'near PCA')
#составим словарь для ngram
data_cv = {}
''' ngam(1,2) for all domains'''
cv = CountVectorizer(analyzer='char', ngram_range=(1,2))
data_cv['X_train12'] = cv.fit_transform(all_domains['domain'])
data_cv['test_X12'] = cv.transform(test_domains['domain'])
pca = PCA(n_components=20)
data_cv['X_train_pca'] = pca.fit_transform(data_cv['X_train12'].toarray())
data_cv['test_X_pca'] = pca.transform(data_cv['test_X12'].toarray())

''' ngam(1,2) for words from language dictionary'''
cv = CountVectorizer(analyzer='char', ngram_range=(1,2))
X_train121 = cv.fit_transform(dataframe_dict['word'])
X_train12 = cv.transform(all_domains['domain'])
test_X12 = cv.transform(test_domains['domain'])
pca = PCA(n_components=10)
pca.fit_transform(X_train121.toarray())
data_cv['X_train_pca_alexa']= pca.transform(X_train12.toarray())
data_cv['test_X_pca_alexa'] = pca.transform(test_X12.toarray())

'''make feature matrix'''
X = all_domains.as_matrix(['length', 'entropy', 'alexa_grams', 'word_grams','diff', 'bad_grams'])
test_X = test_domains.as_matrix(['length', 'entropy', 'alexa_grams', 'word_grams', 'diff','bad_grams'])
y = np.array(all_domains['class'].tolist())

'''add pca features'''
X = np.hstack([X,data_cv['X_train_pca']])
test_X = np.hstack([test_X,data_cv['test_X_pca']])
X = np.hstack([X,data_cv['X_train_pca_alexa']])
test_X = np.hstack([test_X,data_cv['test_X_pca_alexa']])


print( 'Done data')

'''split data'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
#X_train = X
#y_train =y

clf = RandomForestClassifier(n_estimators=40, n_jobs=-1)
clf.fit(X_train, y_train)
z = clf.predict(X_test)
f1 = f1_score(y_test,z)
print("f1_score: ", f1)

pred = clf.predict(test_X)
test_domains['RF'] = pred

def save(pr_n):
    res = open('result'+pr_n+'.csv', 'w')
    k=0;
    for i in test_domains['domain']:
        res.write(str(test_domains[pr_n][k]))
        res.write(',')
        res.write(i)
        res.write('\n')
        k=k+1;
    
save('RF')