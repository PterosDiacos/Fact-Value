import pickle
import collections
import numpy as np
import spacy
import time
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score


DEBUG_PRINT = True
PKL_PATH = "court-memory.pkl"
# keep 'NUM' as the last, function lemma_dep_list() refers to this
LEMMA_FILTER = ['NOUN', 'PROPN', 'VERB', 'ADJ', 'NUM'] 
nlp = spacy.load('en', disable=['ner'])


def loadDataset(pklPath=PKL_PATH):
    with open(pklPath, "rb") as pklFile:
        return np.array(pickle.load(pklFile, encoding="utf-8"))

def divideDataSet(data_set, iterateNum=1, testSize=0.2, seed=0):
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
            localdepList = [chosen_form + "|" + child.dep_ for child in token.children 
                                                           if child.pos_ in LEMMA_FILTER[:-1]]
            localdepList.append(chosen_form + "|self")
            localVocab += localdepList
                        
    return localVocab

def addCountTo(item):
    doc = nlp('\n'.join(item['text']))
    item['len'] = len(doc)
    item['count'] = lemma_dep_list(doc)
    item['text'] = None


def vocabBuild(dataset, freq_threshold=5000):
    '''
    Build vocabulary and parse dataset
    '''
    freq_counter = collections.Counter()
    for item in dataset:
        addCountTo(item)
        freq_counter.update(item['count'])

    if DEBUG_PRINT:
        print('Vocabulary building finished at {0}.'.format(time.ctime()))

    return {pair[0] for pair in freq_counter.most_common(freq_threshold)}


def featureEmbedding(vocab, dataset):
    '''
    Add feature vectors to dataset
    '''
    feature_to_id = dict(zip(vocab, range(len(vocab))))

    for item in dataset:
        if not 'count' in item: addCountTo(item)
        lemma_dep_counter = collections.Counter(item['count'])
        
        item['vector'] = np.zeros(len(vocab))
        for pair in lemma_dep_counter.items():
            if pair[0] in vocab:
                item['vector'][feature_to_id[pair[0]]] = (lambda x, y: x if pair[1] > 0 else y)(pair[1] * 100 / item['len'], 0)
    

def getVectorClassLists(dataset):
    vec_list = [item['vector'] for item in dataset]
    cls_list = [item['label'] for item in dataset]
    return vec_list, cls_list


'''
main
'''
data_set = loadDataset()
train_set, dev_set = divideDataSet(data_set, seed=10)

vocab = vocabBuild(train_set)
featureEmbedding(vocab, train_set)
featureEmbedding(vocab, dev_set)

vec_train, class_train = getVectorClassLists(train_set)
vec_dev, class_dev = getVectorClassLists(dev_set)

# model = LogisticRegression()
model = MLPClassifier(activation='relu', solver='adam', hidden_layer_sizes=(40, 50, 60), max_iter=100000)
model = model.fit(vec_train, class_train)
predClass_dev = model.predict(vec_dev)

print('F1', f1_score(predClass_dev, class_dev, average=None))
