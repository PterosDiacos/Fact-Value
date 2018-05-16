import spacy
import collections
import numpy as np
import re
import string


LEMMA_FILTER = ['NOUN', 'PROPN', 'VERB', 'ADJ', 'NUM'] 
nlp = spacy.load('en', disable=['ner'])



def isalpha(s, alpha_rate=0.75):
    punctMap = str.maketrans('', '', string.punctuation)
    s1 = s.translate(punctMap)
    return len(s1)/len(s) >= alpha_rate if len(s) > 0 else None

def textClean(s):
    s = re.sub(r"\s*\'\s*s", "\'s", s)
    s = re.sub(r"(?<=\w[\.,!:;?\'\"])\d*", "", s)
    s = s.replace('&AMP','&')
    return s if isalpha(s) else None

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
    doc = nlp(item['text'])
    item['len'] = len(doc)
    item['count'] = collections.Counter(lemma_dep_list(doc))
    
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
