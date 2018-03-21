import pickle
import spacy
import string


nlp = spacy.load('en')
punctMap = str.maketrans('', '', string.punctuation)
oldPklFilePath = 'lines_dict.pkl'
newPklFilePath = 'courtLB.pkl'


def properObj(textObj, alpha_rate=0.85):
    '''
    Check if the label contains a contentful word (noun, verb, adj), 
    and if the text contains enough alphabetic characters
    '''
    labelString = textObj[0].replace('&AMP','&')
    labelString = labelString.translate(punctMap)
    docLabel = nlp(labelString)
    posSet = {token.pos_ for token in docLabel if len(token) > 1}
    
    if len(set(['NOUN', 'VERB', 'ADJ']) & posSet) == 0:
        return 0
    else:
        textString = ' '.join(textObj[1])
        newTextString = textString.translate(punctMap)
        return len(newTextString) / len(textString) >= alpha_rate


# the original pkl file is an object of colleciton.defaultdict
pklFile = open(oldPklFilePath, 'rb')
pklData = sorted(pickle.load(pklFile, encoding='utf-8').items())
pklFile.close()


pklData = [textObj for textObj in pklData if properObj(textObj)]

# the updated pkl file is a list
pklFile = open(newPklFilePath, 'wb')
pickle.dump(pklData, pklFile)
pklFile.close()
