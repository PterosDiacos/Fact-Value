import pickle
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


## constants
PKL_PATH = "court-test.pkl"


## global functions
def loadDataset(pklPath=PKL_PATH):
    with open(pklPath, "rb") as pklFile:
        return np.array(pickle.load(pklFile, encoding="utf-8"))

def divideDataSet(data_set, iterateNum=1, testSize=0.2, seed=10):
    '''
    Divide the data_set into train_set and dev_set, with stratified sampling.
    '''
    pool_split = StratifiedShuffleSplit(n_splits=iterateNum, test_size=testSize, random_state=seed)
    label_array = np.array([item['label'] for item in data_set])

    for train_index, dev_index in pool_split.split(data_set, label_array):
        train_set = np.array([data_set[i] for i in train_index])
        dev_set = np.array([data_set[i] for i in dev_index])
    
    return train_set, dev_set

def saveDataset(pklPath, dataset):
    pklFile = open(pklPath, "wb")
    pickle.dump(dataset, pklFile)
    pklFile.close()

## main
data_set = loadDataset()
train_dev_set, test_set = divideDataSet(data_set, testSize=0.4, seed=10)
# saveDataset('court-train-dev.pkl', train_dev_set)
saveDataset('court-t40.pkl', test_set)
