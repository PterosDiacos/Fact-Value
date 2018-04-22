import zipfile as zp
import pickle
import collections
import spacy
import os
import time
import re
import string


nlp = spacy.load('en', disable=['ner'])

ZIPFILE_HOME = '/data/Dropbox/Data/Circuit_Courts/circuit-cases/'
ZIPFILE_NAME = 'sentences.zip'
PICKLE_HOME = '/home/yucao/sents/'



def loadZipFile(zipfilePath=os.path.join(ZIPFILE_HOME, ZIPFILE_NAME), yearFolderDepth=1, maxPerYear=3999):
    z = zp.ZipFile(zipfilePath, 'r')
    zmembers = filter(lambda s: s.endswith('.txt'), z.namelist())

    zmembersByYear = collections.defaultdict(list)
    for zf in zmembers:
        zmembersByYear[zf.split('/')[yearFolderDepth][-4:]].append(zf)
    
    zmembersByYear2 = collections.defaultdict(list)
    for key in zmembersByYear:
        nyear = len(zmembersByYear[key])
        if nyear > maxPerYear:
            zmembersByYear2[key + 'a'] = zmembersByYear[key][:nyear // 2]
            zmembersByYear2[key + 'b'] = zmembersByYear[key][nyear // 2:]
        else:
            zmembersByYear2[key] = zmembersByYear[key][:]
    
    del zmembersByYear
    return z, zmembersByYear2


def isalpha(s, alpha_rate=0.75):
    punctMap = str.maketrans('', '', string.punctuation)
    s1 = s.translate(punctMap)
    return len(s1)/len(s) >= alpha_rate

def tclean(s):
    s = re.sub(r"\s*\'\s*s", "\'s", s)
    s = re.sub(r"(?<=\w[\.,!:;?\'\"])\d*", "", s)
    s = s.replace('&AMP','&')
    return s if isalpha(s) else None


def spacyParse(z, zlist, year):
    print('Processing files in %s, %d in total. %s' % (year, len(zlist), time.ctime()))

    thisYear = []
    for zname in zlist:
        with z.open(zname, 'r') as fin:
            text = [line.decode('utf-8') for line in fin.readlines()]
        text = [tclean(line) for line in text]
        text = filter(lambda s: s is not None, text) 
        thisYear.append(nlp(''.join(text)))

    with open(os.path.join(PICKLE_HOME, year + '.pkl'), 'wb') as fout:
        pickle.dump(thisYear, fout)
    
    del thisYear



z, zmby = loadZipFile()
for key in zmby:
    spacyParse(z, zmby[key], key)
