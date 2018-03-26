import pickle
import collections
import numpy as np
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score



## const
PKL_PATH = "court-LbParag-mini.pkl"
LEMMA_FILTER = ['NOUN', 'PROPN', 'VERB', 'ADJ', 'NUM'] # keep 'NUM' as the last, function lemma_dep_list() refers to this
nlp = spacy.load("en")



def loadDataset(pklPath=PKL_PATH):
    with open(pklPath, "rb") as pklFile:
        return np.array(pickle.load(pklFile, encoding="utf-8"))

def divideDataSet(data_set, iterateNum=1, testSize=0.2, seed=30):
    '''
    Divide the data_set into train_set and dev_set, with stratified sampling.
    '''
    pool_split = StratifiedShuffleSplit(n_splits=iterateNum, test_size=testSize, random_state=seed)
    label_array = np.array([item['label'] for item in data_set])

    for train_index, dev_index in pool_split.split(data_set, label_array):
        train_set = np.array([data_set[i] for i in train_index])
        dev_set = np.array([data_set[i] for i in dev_index])
    
    return train_set, dev_set

def chooseWordForm(token):
    return token.lemma_ if token.pos_ in LEMMA_FILTER[:-1] else token.shape_

def lemma_dep_list(doc):
    localVocab = []
    for token in doc:
        if token.pos_ in LEMMA_FILTER:
            chosen_form = chooseWordForm(token)
            localdepList = [chosen_form + "|" + child.dep_ for child in token.children if child.pos_ in LEMMA_FILTER[:-1]]
            localVocab += localdepList
            localVocab.append(chosen_form + "|self")
            
    return localVocab

def vocabBuild(dataset):
    '''
    Build vocabulary and parse dataset
    '''
    vocab = set()
    for item in dataset:
        doc = nlp("\n".join(item['text']))
        item['text'] = doc
        item['count'] = lemma_dep_list(doc)
        vocab.update(set(item['count']))
    return vocab

def addFeature(vocab, dataset, feature_set=set()):
    '''
    add feature and feature vector to dataset
    '''
    for item in dataset:
        if not 'count' in item:
            doc = nlp("\n".join(item['text']))
            item['text'] = doc
            item['count'] = lemma_dep_list(doc)
        lemma_dep_counter = collections.Counter(item['count'])

        item['feature'] = collections.defaultdict(float) 
        for pair in lemma_dep_counter.items():
            if pair[0] in vocab:
                item['feature']['cnt_' + pair[0]] = (lambda x, y: x if pair[1] > 0 else y)(pair[1] * 100 / len(item['text']), 0)        
        feature_set.update(item['feature'].keys())
    
    return feature_set

def addVector(feature_name, dataset):
    feature_to_id = dict(zip(feature_name, range(len(feature_name))))
    for item in dataset:
        item['vector'] = np.zeros(len(feature_name))
        for feature in item['feature']:
            item['vector'][feature_to_id[feature]] = item['feature'][feature]
    return dataset


## main
data_set = loadDataset()
train_set, dev_set = divideDataSet(data_set)
vocab = vocabBuild(train_set)

feature_set = addFeature(vocab, train_set)
feature_set = addFeature(vocab, dev_set, feature_set)
addVector(feature_set, train_set)
addVector(feature_set, dev_set)

vec_train = [item['vector'] for item in train_set]
class_train = [item['label'] for item in train_set]
log_model = LogisticRegression()
log_model = log_model.fit(vec_train, class_train) # tweak the parameter of log_reg

vec_dev = [item['vector'] for item in dev_set]
class_dev = [item['label'] for item in dev_set]
predClass_dev = log_model.predict(vec_dev)
print('F1', f1_score(predClass_dev, class_dev, average=None))
