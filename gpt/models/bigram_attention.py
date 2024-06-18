import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AttentionHead(nn.Module):
    def __init__(self, head_size, n_embeddings, block_size):
        super().__init__()
        self.key = nn.Linear(n_embeddings, head_size, bias=False)
        self.query = nn.Linear(n_embeddings, head_size, bias=False)
        self.value = nn.Linear(n_embeddings, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        weights = q @ k.transpose(-2,-1) * C**-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=1)
        out = weights @ v
        return out

class MultiHeadAtttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embeddings, block_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size, n_embeddings, block_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)

class FeedForward(nn.Module):
    def __init__(self, n_embeddings):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embeddings, n_embeddings),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class BigramLanguageModelAttention(nn.Module):
    def __init__(self, vocab_size, n_embeddings, block_size):
        self.block_size = block_size
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embeddings)
        self.position_embedding_table = nn.Embedding(block_size, n_embeddings)
        self.linear = nn.Linear(n_embeddings, vocab_size)
        self.self_attention_head = AttentionHead(head_size=n_embeddings, 
                                                 n_embeddings=n_embeddings, 
                                                 block_size=block_size)
        self.multiheads = MultiHeadAtttention(4, n_embeddings//4, n_embeddings, block_size)
        self.ffw = FeedForward(n_embeddings)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx) # B T C
        pos_embeddings = self.position_embedding_table(torch.arange(T, device=device)) # positional embeddings do not change the behavior of a bigram model.
        x = token_embeddings + pos_embeddings
        #x = self.self_attention_head(x)
        x = self.multiheads(x)
        x = self.ffw(x)
        logits = self.linear(x) #batch X time X channels
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cropped = idx[:, -self.block_size:]
            logits, loss = self(idx_cropped)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx