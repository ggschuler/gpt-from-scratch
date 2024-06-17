from models.bigram import BigramLanguageModel
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Trainer:
    def __init__(self, loader):    
        self.batch_size = 32
        self.lr = 1e-3
        self.num_epochs = 10000
        self.eval_iters = 200
        self.eval_interval = 300
        self.loader = loader
    
    @torch.no_grad
    def estimate_loss(self, model):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.loader.get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    def train(self, dec):
        print(device)
        model = BigramLanguageModel(self.loader.vocab_size)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        for epoch in range(self.num_epochs):
            if epoch % self.eval_interval == 0:
                losses = self.estimate_loss(model)
                print(losses['train'])
                print(f'step {epoch}: train loss {losses["train"]}, val loss {losses["val"]}.')

            xb, yb = self.loader.get_batch('train')
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        
        context = model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=500)[0].tolist()
        decoded = [''.join([dec[i] for i in context])]
        print(decoded)
