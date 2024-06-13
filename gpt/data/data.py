import wget

class Downloader:
    def __init__(self) -> None:
        self.url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt' 
        self.chars = 0
        self.path = 'data/data.txt'
        filename = wget.download(self.url, out=self.path)
        with open(self.path, 'r', encoding='utf-8') as f:
            text = f.read()
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        print('\nlength of dataset in characters: ', len(text))
        print('length of char set: ', self.vocab_size)