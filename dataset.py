import os
import numpy as np
import torch

from train_gpt2 import master_process 


def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:

    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}, "split should be either train or val"

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s] # get the train vs val
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards] # get the actual relative file paths
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")

        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(shards[self.current_shard])
        # state
        self.current_position = self.B * self.T * self.process_rank

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = buf[:-1].view(B, T) # inputs
        y = buf[1:].view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes  # we're not doing B * T + 1 here because in training the consumption was only till B * T, the +1 was used for target hence there is no overlap
        # if loading next batch would be out of bounds, reset
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank  # 1 epoch completed (almost)

        return x, y

# Naive dataloader for single file input
# class DataLoaderLite:

#     def __init__(self, B, T, process_rank, num_processes, split):
#         self.B = B
#         self.T = T
#         self.process_rank = process_rank
#         self.num_processes = num_processes
#         assert split in {'train', 'val'}, "split should be either train or val"

#         # at init load tokens from disk and store them in memory
#         with open('input.txt', 'r') as f:
#             text = f.read()
#         enc = tiktoken.get_encoding('gpt2')
#         tokens = enc.encode(text)
#         self.tokens = torch.tensor(tokens)
#         print(f"loaded {len(self.tokens)} tokens")
#         print(f"1 epoch = {len(self.tokens) // (B*T)} steps")

#         # state
#         self.current_position = self.B * self.T * self.process_rank

#     def next_batch(self):
#         B, T = self.B, self.T
#         buf = self.tokens[self.current_position : self.current_position+B*T+1]
#         x = buf[:-1].view(B, T) # inputs
#         y = buf[1:].view(B, T) # targets
#         # advance the position in the tensor
#         self.current_position += B * T * self.num_processes  # we're not doing B * T + 1 here because in training the consumption was only till B * T, the +1 was used for target hence there is no overlap
#         # if loading next batch would be out of bounds, reset
#         if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
#             self.current_position = B * T * self.process_rank  # 1 epoch completed (almost)
#         return x, y
