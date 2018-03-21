import pickle


pklFilePath = 'courtLB.pkl'
textOverviewPath = 'textOverview.txt'

pklFile = open(pklFilePath, 'rb')
pklData = pickle.load(pklFile, encoding='utf-8')
pklFile.close()

with open(textOverviewPath, 'w', encoding='utf-8') as fout:
    for item in pklData[:1000]:
        x = 100 if len(item[1]) > 100 else len(item[1])
        print('\n'.join(item[1][:x]) + '\n' + '*' * 10 + '\n', file=fout)
