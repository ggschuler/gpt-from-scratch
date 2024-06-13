class Tokenizer:
    def __init__(self, chars) -> None:
        self.chars = chars

    def tokenize(self):
        str_to_int = { ch:i for i, ch in enumerate(self.chars)}
        int_to_str = { i:ch for i, ch in enumerate(self.chars)}
        encode = lambda s: [str_to_int[c] for c in s]
        decode = lambda l: ''.join([int_to_str[i] for i in l])
        print(encode('hi there!'))
        print(decode(encode('hi there!')))