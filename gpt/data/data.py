import wget
import os
import torch
import tempfile

class Data:
    def __init__(self) -> None:
        self.url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt' 
        self.chars = 0
        self.text = 0
        self.path = os.path.join('gpt', 'data')

    def download(self):
        if not os.path.exists('input.txt'):
            filename = wget.download(self.url)
        else:
            filename = 'input.txt'
        with open(filename, 'r', encoding='utf-8') as f:
            self.text = f.read()
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        print(self.vocab_size)
            
    def tokenize(self):
        str_to_int = { ch:i for i, ch in enumerate(self.chars)}
        int_to_str = { i:ch for i, ch in enumerate(self.chars)}
        encode = lambda s: [str_to_int[c] for c in s]
        decode = lambda l: ''.join([int_to_str[i] for i in l])
        data = torch.tensor(encode(self.text), dtype=torch.long)
        return data