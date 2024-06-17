import wget
import os
import torch

class Data:
    def __init__(self) -> None:
        self.url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt' 
        self.chars = 0
        self.text = 0
        self.vocab_size = 0
        self.path = os.path.join('gpt', 'data')
        self.str_to_int = 0
        self.int_to_str = 0

    def download(self):
        if not os.path.exists('input.txt'):
            filename = wget.download(self.url)
        else:
            filename = 'input.txt'
        with open(filename, 'r', encoding='utf-8') as f:
            self.text = f.read()
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
            
    def tokenize(self):
        self.str_to_int = { ch:i for i, ch in enumerate(self.chars)}
        self.int_to_str = { i:ch for i, ch in enumerate(self.chars)}
        encode = lambda s: [self.str_to_int[c] for c in s]
        decode = lambda l: ''.join([self.int_to_str[i] for i in l])
        data = torch.tensor(encode(self.text), dtype=torch.long)
        return data
    
    def split(self, t_pct, v_pct):
        tks = self.tokenize()
        n = int(t_pct*len(tks))
        train_data = tks[:n]
        val_data = tks[n:]
        return train_data, val_data