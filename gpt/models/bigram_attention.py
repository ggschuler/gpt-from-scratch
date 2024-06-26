import torch
import torch.nn as nn
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AttentionHead(nn.Module):
    def __init__(self, head_size, n_embeddings, block_size, dropout_rate):
        super().__init__()
        self.key = nn.Linear(n_embeddings, head_size, bias=False)
        self.query = nn.Linear(n_embeddings, head_size, bias=False)
        self.value = nn.Linear(n_embeddings, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        weights = q @ k.transpose(-2,-1) * C**-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=1)
        weights = self.dropout(weights)
        out = weights @ v
        return out

class MultiHeadAtttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embeddings, block_size, dropout_rate):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size, n_embeddings, block_size, dropout_rate) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embeddings, n_embeddings)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embeddings, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embeddings, 4 * n_embeddings),
            nn.ReLU(),
            nn.Linear(4 * n_embeddings, n_embeddings),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    def __init__(self, n_embeddings, n_heads, block_size, dropout_rate):
        super().__init__()
        head_size = n_embeddings // n_heads
        self.self_attention = MultiHeadAtttention(n_heads, head_size, n_embeddings, block_size, dropout_rate)
        self.ffwd = FeedForward(n_embeddings, dropout_rate)
        self.layernorm1 = nn.LayerNorm(n_embeddings)
        self.layernorm2 = nn.LayerNorm(n_embeddings)

    def forward(self, x):
        x = x + self.self_attention(self.layernorm1(x))
        x = x + self.ffwd(self.layernorm2(x))
        return x

class BigramLanguageModelAttention(nn.Module):
    def __init__(self, vocab_size, n_embeddings, block_size, n_heads, n_layer, dropout_rate):
        self.block_size = block_size
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embeddings)
        self.position_embedding_table = nn.Embedding(block_size, n_embeddings)
        self.blocks = nn.Sequential(*[Block(n_embeddings, n_heads=n_heads, block_size=block_size, dropout_rate=dropout_rate) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(n_embeddings)
        self.linear = nn.Linear(n_embeddings, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx) # B T C
        pos_embeddings = self.position_embedding_table(torch.arange(T, device=device)) # positional embeddings do not change the behavior of a bigram model.
        x = token_embeddings + pos_embeddings
        x = self.blocks(x)
        x = self.layer_norm(x)
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