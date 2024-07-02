import os
import torch
import requests

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
        input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
        if not os.path.exists(input_file_path):
            data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w') as f:
            f.write(requests.get(data_url).text)

        with open(input_file_path, 'r') as f:
            self.text = f.read()
        print(f"length of dataset in characters: {len(self.text):,}")
        
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