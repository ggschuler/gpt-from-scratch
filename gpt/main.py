from data.data import Data
from data.split import Split

data = Data()
shakespeare = data.download()
tks = data.tokenize()
print(tks.shape)
splitter = Split(0.9, 0.1)
train_tks, val_tks = splitter.split_it(tks)
print(train_tks.shape)