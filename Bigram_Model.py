import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32 # parallel processing batches
block_size = 8 # maximum content length
max_iters = 3000
eval_interval = 300
lr = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

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

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # vocab_size X vocab_size => 65*65
    
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # {batch, time, channel}

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
            logits, loss = self(idx) # getting predictions
            # focusing on only last time-step
            logits = logits[:, -1, :] # {B, C}
            # applying softmax to get probabilities
            prob = F.softmax(logits, dim=1) # {B, C}
            # sample from distribution
            idx_next = torch.multinomial(prob, num_samples=1) # {B, 1}
            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # {B, T+1}
        return idx
    
model = BigramLanguageModel(vocab_size)
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