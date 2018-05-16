import pickle
import collections
import numpy as np
import spacy
import time
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score



DEBUG_PRINT = False
PKL_PATH = 'court-mini.pkl'
# keep 'NUM' as the last, function lemma_dep_list() refers to this
LEMMA_FILTER = ['NOUN', 'PROPN', 'VERB', 'ADJ', 'NUM'] 
nlp = spacy.load('en', disable=['ner'])



def loadDataset(pklPath=PKL_PATH):
    with open(pklPath, "rb") as pklFile:
        return np.array(pickle.load(pklFile, encoding="utf-8"))

def loadDatSet(filename):
    with open(filename, 'rb') as fin:
        return pickle.load(fin, encoding='utf-8')

def saveDatSet(dataset, filename):
    with open(filename, 'wb') as fout:
        pickle.dump(dataset, fout)
                

def divideDataSet(dataset, iterateNum=1, testSize=0.2, seed=0):
    '''
    Divide the dataset into train_set and dev_set, with stratified sampling.
    '''
    pool_split = StratifiedShuffleSplit(n_splits=iterateNum, test_size=testSize, random_state=seed)
    label_array = np.array([item['label'] for item in dataset])

    for train_index, dev_index in pool_split.split(dataset, label_array):
        train_set = np.array([dataset[i] for i in train_index])
        dev_set = np.array([dataset[i] for i in dev_index])
    
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
    item['count'] = collections.Counter(lemma_dep_list(doc))
    del item['text']



def vocabBuild(dataset, freq_threshold=4000):
    '''
    Build vocabulary and parse dataset
    '''
    freq_counter = collections.Counter()
    for c, item in enumerate(dataset):
        addCountTo(item)
        freq_counter.update(item['count']) 
        
        if DEBUG_PRINT and c % 100 == 0:
            print('{0} / {1} = {2} vocabulary built at {3}.'.format(c, 
            len(dataset), 
            c / len(dataset), 
            time.ctime()))

    if DEBUG_PRINT:
        print('Vocabulary building finished at {0}.'.format(time.ctime()))

    return [pair[0] for pair in freq_counter.most_common(freq_threshold)]


def featureEmbedding(vocab, dataset):
    '''
    Add feature vectors to dataset
    '''
    feature_to_id = dict(zip(vocab, range(len(vocab))))

    for item in dataset:
        if not 'count' in item: addCountTo(item)
        
        item['vector'] = np.zeros(len(vocab))
        for pair in item['count'].items():
            if pair[0] in vocab:
                weight = np.log10(item['len']) if item['len'] > 10 else 1
                item['vector'][feature_to_id[pair[0]]] = pair[1] / weight
        del item['count']
    
    if DEBUG_PRINT:
        print('Feature embedding finished at {0}.'.format(time.ctime()))
    


def getVectorClassLists(dataset):
    vec_list = [item['vector'] for item in dataset]
    cls_list = [item['label'] for item in dataset]
    return vec_list, cls_list

def generateBatches(vec_list, cls_list, batchSize=5):
    start = 0
    vecLen = len(vec_list)
    while start < vecLen:
        yield vec_list[start:start + batchSize], cls_list[start:start + batchSize]
        start += batchSize



'''
main
'''
dataset = loadDataset()
train_set, dev_set = divideDataSet(dataset, seed=10)
vocab = vocabBuild(train_set)

featureEmbedding(vocab, train_set)
featureEmbedding(vocab, dev_set)
vec_train, class_train = getVectorClassLists(train_set)
vec_dev, class_dev = getVectorClassLists(dev_set)


model = MLPClassifier(hidden_layer_sizes=(40, 50, 60), max_iter=100000, verbose=True)
model.fit(vec_train, class_train)

# batchPool = generateBatches(vec_train, class_train)
# vec_init, class_init = next(batchPool)
# model.partial_fit(vec_init, class_init, np.unique(class_train))
# for vec_batch, class_batch in batchPool:
#     model.partial_fit(vec_batch, class_batch)


predClass_dev = model.predict(vec_dev)
print('F1', f1_score(predClass_dev, class_dev, average=None))
