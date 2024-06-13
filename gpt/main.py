from data.data import Data
from data.split import Split
from data.data_loader import DataLoader

data = Data()
shakespeare = data.download()
tks = data.tokenize()
splitter = Split(0.9, 0.1)
train_tks, val_tks = splitter.split_it(tks)
loader = DataLoader(train_tks, val_tks)
loader.get_batch('train')