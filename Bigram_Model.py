import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64 # parallel processing batches
block_size = 256 # maximum content length
max_iters = 5000
eval_interval = 500
lr = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

with open('tiny-shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))

vocab_size = len(chars)

encodings = {ch:i for i,ch in enumerate(chars)} # map character to integer
decodings = {i:ch for i,ch in enumerate(chars)} # map integer to character

encoder = lambda s: [encodings[c] for c in s] # takes a string, returns a list of integers
decoder = lambda l: ''.join([decodings[i] for i in l]) # takes a list of integers, returns a string

data = torch.tensor(encoder(text), dtype=torch.long)

train = data[:int(0.9*len(data))]
val = data[int(0.9*len(data)):]

def get_batch(split):
    data = train if split == 'train' else val
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x, y

@torch.no_grad()
def estimate_loss():
    out_loss = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out_loss[split] = losses.mean()
    model.train()
    return out_loss

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        # Attention scores or "Affinities"
        weights = q @ k.transpose(-2, -1) * C**(-0.5)  # (B, T, C) @ (B, C, T) ==> (B, T, T)

        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        weights = F.softmax(weights, dim=-1) # (B, T, T)
        weights = self.dropout(weights)

        # weighted aggregationof values
        v = self.value(x) # (B, T, C)
        out = weights @ v

        return out    
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class MultiHeadAttention(nn.Module):
    # multiple self-attention heads running in parallel
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class Block(nn.Module):
    # transformer block: communication followed by computation
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the no. of heads we would like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # multi-headed self-attention: 4 heads of 8-dimensional self-attention
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) # row-wise normalized: per token (mean = 0, std = 1)
        self.ln2 = nn.LayerNorm(n_embd) # row-wise normalized: per token (mean = 0, std = 1)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # adding residual connection: forking and adding attention
        x = x + self.ffwd(self.ln2(x)) # adding residual connection: forking and adding feed forward neural net that has processed info jump from attention
        return x
    
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # vocab_size X 32 dimensional embeddings => 65*32: gives token embeddings
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # each token from 0 to blocksize+1 will also get positonal vector embedding
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # linear model head
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_embd = self.token_embedding_table(idx) # {batch, time, embedded channel}
        pos_embd = self.position_embedding_table(torch.arange(T, device=device).unsqueeze(0).expand(B, T)) # {time, channel}
        x = tok_embd + pos_embd # {batch, time, embedded channel}
        x = self.blocks(x) # {batch, time, embedded channel}
        x = self.ln_f(x) # {batch, time, embedded channel}
        logits = self.lm_head(x) # contains both {how tokens are being represented} + {where the token is positioned}
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is {B, T} array of indices in current context
        for _ in range(max_new_tokens):
            # cropping idx to last block_size tokens
            cond_idx = idx[:, -block_size:]
            # getting predictions
            logits, loss = self(cond_idx) 
            # focusing on only last time-step
            logits = logits[:, -1, :] # {B, C}
            # applying softmax to get probabilities
            prob = F.softmax(logits, dim=1) # {B, C}
            # sample from distribution
            idx_next = torch.multinomial(prob, num_samples=1) # {B, 1}
            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # {B, T+1}
        return idx
    
model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # optimizer for gradient descent

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decoder(m.generate(context, max_new_tokens=500)[0].tolist()))