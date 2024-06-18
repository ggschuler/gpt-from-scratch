from models.bigram import BigramLanguageModel
from models.bigram_attention import BigramLanguageModelAttention
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Trainer:
    def __init__(self, loader):
        self.loader = loader    
        self.batch_size = self.loader.batch_size
        self.block_size = self.loader.block_size
        self.lr = 3e-4
        self.n_heads = 6
        self.n_layer = 6
        self.dropout_rate = 0.2
        self.num_epochs = 5000
        self.eval_iters = 200
        self.eval_interval = 500
        self.n_embeddings = 384
    
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
        #model = BigramLanguageModel(self.loader.vocab_size)
        model = BigramLanguageModelAttention(vocab_size=self.loader.vocab_size, 
                                             n_embeddings=self.n_embeddings, 
                                             block_size=self.loader.block_size,
                                             n_heads=self.n_heads,
                                             n_layer=self.n_layer,
                                             dropout_rate=self.dropout_rate)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        for epoch in range(self.num_epochs):
            if epoch % self.eval_interval == 0:
                losses = self.estimate_loss(model)
                print(f'step {epoch}: train loss {losses["train"]}, val loss {losses["val"]}.')

            xb, yb = self.loader.get_batch('train')
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        
        context = model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=500)[0].tolist()
        decoded = ''.join([dec[i] for i in context])
        print(decoded)
