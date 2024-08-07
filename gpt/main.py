from data.data import Data
from data.data_loader import DataLoader
from models.trainer import Trainer
import os

os.environ['MASTER_ADDR'] 
os.environ['MASTER_PORT']


data = Data()
shakespeare = data.download()
train_tks, val_tks = data.split(.9, .1)

loader = DataLoader(train_tks, val_tks, data.vocab_size)
trainer = Trainer(loader)

trainer.train(data.int_to_str, 0, 2)