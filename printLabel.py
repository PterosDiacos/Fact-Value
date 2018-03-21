import pickle
import spacy


nlp = spacy.load('en')
pklFilePath = 'courtLB.pkl'
labelOverviewPath = 'labelOverview.txt'


def label_to_LBpos(textObj):
    docLabel = nlp(textObj[0])
    posList = [token.pos_ for token in docLabel]
    return ' '.join(posList)


pklFile = open(pklFilePath, 'rb')
pklData = pickle.load(pklFile, encoding='utf-8')
pklFile.close()

labelList = [item[0] + "\n" for item in pklData]
with open(labelOverviewPath, 'w', encoding='utf-8') as fout:
    fout.writelines(labelList)
