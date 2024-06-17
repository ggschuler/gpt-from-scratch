import torch
from models.bigram import BigramLanguageModel

class DataLoader:
    def __init__(self, train_data, val_data, vocab_size):
        self.batch_size = 4
        self.block_size = 8
        self.train_data = train_data
        self.val_data = val_data
        self.vocab_size = vocab_size
        
    def get_batch(self, split):
        data = self.train_data if split=='train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x,y
    
    def train(self):
        xb, yb = self.get_batch('train')
        bigram = BigramLanguageModel(self.vocab_size)


