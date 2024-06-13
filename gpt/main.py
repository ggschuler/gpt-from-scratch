from data.data import Downloader
from data.char_tokenizer import Tokenizer

data = Downloader()
tokens = Tokenizer(data.chars)
tokens.tokenize()