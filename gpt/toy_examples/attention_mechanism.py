import torch
import torch.nn as nn
import torch.nn.functional as F
B, T, C = 4, 8, 32
x = torch.randn(B,T,C)

head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)
q = query(x)
raw_weights = q @ k.transpose(-2, -1) # B,T,16 @ B, 16,T = B, T, T
tril = torch.tril(torch.ones(T, T)) # lower triangular portion of T x T matrix, providing non-future connections for each token
masked_weights = raw_weights.masked_fill(tril == 0, float('-inf'))
softmaxed_weights = F.softmax(masked_weights, dim=1)

v = value(x)
out = softmaxed_weights @ v


#print(raw_weights[0])
#print(masked_weights[0])
#print(softmaxed_weights[0])
print(out.shape)