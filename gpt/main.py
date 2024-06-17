from data.data import Data
from data.data_loader import DataLoader

data = Data()
vocab_size = data.vocab_size
shakespeare = data.download()
train_tks, val_tks = data.split(.9, .1)

loader = DataLoader(train_tks, val_tks, vocab_size)
loader.train()