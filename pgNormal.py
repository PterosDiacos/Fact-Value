import pickle
import string
import re
import spacy
import pandas as pd

nlp = spacy.load('en')
END_PUNCT = ['.', '?', '!', ':']

oldpklFilePath = 'courtLB.pkl'
newpklFilePath = 'court-LbParag.pkl'

labelCSVPath = 'FVLabel.csv'
label_df = pd.read_csv(labelCSVPath)
label_df = label_df.fillna('')
FACTNUM = 1
VALNUM = 2



def is_complete(s):
    doc = nlp(s)
    return len(doc[-1].text) == 1 and doc[-1].tag_ in END_PUNCT

def isalpha(s, alpha_rate=0.75):
    punctMap = str.maketrans('', '', string.punctuation)
    s1 = s.translate(punctMap)
    return len(s1)/len(s) >= alpha_rate

def line_to_sents(line, alpha_rate=0.75):
    doc = nlp(line)
    sents = []
    start_id = 0
    for i, token in enumerate(doc):
       if i == len(doc) - 1 or (len(token.text) == 1 and token.tag_ in END_PUNCT):
           s = ' '.join([tk.text for tk in doc[start_id:i + 1]])
           s = re.sub(r"\s*\'\s*s", "\'s", s)
           s = re.sub(r"(?<=\w[\.,!:;?\'\"])\d*", "", s)
           s = s.replace('&AMP','&')
           if isalpha(s, alpha_rate): sents.append(s)
           start_id = i + 1
    return sents

def isInformative(label):
    flag = 0
    for value_lb in label_df['Value'].values:
        if value_lb == '': break
        if label.find(value_lb) >= 0:
            flag = VALNUM
            break
    if not flag:
        for fact_lb in label_df['Fact'].values:
            if fact_lb == '': break
            if label.find(fact_lb) >= 0:
                flag = FACTNUM
                break
    return flag

def paragNormalize(textObj):
    '''
    Convert a text into well-sized paragraphs. 
    Return a list of textObjs that share the same label.
    '''
    commonLabel = textObj[0]
    paragList = []

    if isInformative(commonLabel):
        raw_text = textObj[1]
        workingParag = []
        pendingSent = ''
        
        for i, line in enumerate(raw_text):
            raw_text[i] = re.sub(r'\s*\[|\]\s*', '', line)
            sents = line_to_sents(line)
            if len(sents) == 0: continue
            
            if len(pendingSent) > 0:
                sents[0] = pendingSent + ' ' + sents[0]
                pendingSent = ''
        
            if i == len(raw_text) - 1:
                workingParag += sents
                paragList.append([commonLabel, workingParag])
                workingParag = []
            else:
                lastSent = sents[-1]
                if is_complete(lastSent):
                    workingParag += sents
                    paragList.append([commonLabel, workingParag])
                    workingParag = []
                else:
                    pendingSent = lastSent
                    if len(sents) > 1: workingParag += sents[:-1]

    return paragList



pklFile = open(oldpklFilePath, 'rb')
pklData = pickle.load(pklFile, encoding='utf-8')
pklFile.close()

newCorpus = []
for id, item in enumerate(pklData):
    newList = paragNormalize(item)
    if len(newList) > 0: newCorpus += newList
    if id % 50 == 0: 
        print(id, '/', len(pklData))

pklFile = open(newpklFilePath, 'wb')
pickle.dump(newCorpus, pklFile)
pklFile.close()
