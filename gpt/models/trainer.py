from models.bigram import BigramLanguageModel
import torch

class Trainer:
    def __init__(self, loader):    
        self.batch_size = 32
        self.lr = 1e-3
        self.num_epochs = 10000
        self.loader = loader

    def train(self, dec):
        model = BigramLanguageModel(self.loader.vocab_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        for epoch in range(self.num_epochs):
            xb, yb = self.loader.get_batch('train')
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print(loss.item())
        out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()
        decoded = [''.join([dec[i] for i in out])]
        print(decoded)
