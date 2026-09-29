#STEP-1
import torch

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
print("".join(chars))
print("vocab size:", vocab_size)

#step-2
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

print(encode("hii there"))
print(decode(encode("hii there")))

#step-3
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:50])

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

#step-4
torch.manual_seed(1337)
block_size = 4
batch_size = 4

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y

xb, yb = get_batch("train")
print("inputs:\n", xb)
print("targets:\n", yb)

for t in range(block_size):
    context = xb[0, :t + 1]
    target = yb[0, t]
    print(f"{decode(context.tolist())!r} -> {decode([target.item()])!r}")

# Step-5
import torch.nn as nn
from torch.nn import functional as F

device = "mps" if torch.backends.mps.is_available() else "cpu"

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel(vocab_size).to(device)
xb, yb = get_batch("train")
logits, loss = model(xb.to(device), yb.to(device))
print(logits.shape, loss.item())

start = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(start, 100)[0].tolist()))


#step-6
batch_size = 32
max_iters = 10000
eval_interval = 1000
eval_iters = 200

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X.to(device), Y.to(device))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for step in range(max_iters):
    if step % eval_interval == 0 or step == max_iters - 1:
        l = estimate_loss()
        print(f"step {step}: train loss {l['train']:.4f}, val loss {l['val']:.4f}")

    xb, yb = get_batch("train")
    logits, loss = model(xb.to(device), yb.to(device))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(decode(model.generate(start, 300)[0].tolist()))
