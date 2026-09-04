# sampling from model

import torch
import torch.nn.functional as F
import tiktoken

from train_gpt2 import device, model


# @TODO: Naive way of loading model, device should be loaded from a util or config and model should be ideally loaded from saved checkpoint

# create a generator different than the main global one and set the seed to 42
sample_rng = torch.Generator(device=device)
sample_rng.manual_seed(42)

num_return_sequences = 5

max_length = 30

model.eval()

# prefix tokens
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model")
tokens = torch.tensor(tokens, dtype=torch.long) #(8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)

x = tokens.to(device)

# generate, B = 5 and T = 8

while x.size(1) < max_length:
    # forwarding the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size) 
        # get the probabilities - for each batch
        probs = F.softmax(logits, dim=-1) 
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from top-k probabilities
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)


for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
