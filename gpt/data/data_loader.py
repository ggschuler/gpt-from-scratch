import torch

class DataLoader:
    def __init__(self, train_data, val_data):
        self.batch_size = 4
        self.block_size = 8
        self.train_data = train_data
        self.val_data = val_data
        

    def get_batch(self, split):
        data = self.train_data if split=='train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        print(x)
        print(y)
