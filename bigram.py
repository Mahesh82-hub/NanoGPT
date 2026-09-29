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
