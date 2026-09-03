import torch
from torch.nn import functional as F
import torch.distributed as dist


from datasets import load_dataset

# ds = load_dataset("Rowan/hellaswag", split="train[:4]") 
ds = load_dataset("Rowan/hellaswag", split="validation")

def render(data, tokenizer):
    # this is one question and then 4 options with label
    ques = tokenizer.encode(data["ctx"])
    options = data["endings"]
    label = int(data["label"])

    options_encoded = []
    max_seq = 0

    for opt in options:
        opt_encoded = tokenizer.encode(" " + opt) # we need a space between question and answer 
        if len(opt_encoded) > max_seq:
            max_seq = len(opt_encoded)
        options_encoded.append(opt_encoded)

    max_seq += len(ques)

    tokens = []
    targets = []
 
    for i, opt in enumerate(options_encoded):
        padding_len = max_seq - (len(ques) + len(opt))
        seq = ques + opt + [0] * padding_len

        # create targets / target
        mask_curr = [-1] * (len(ques) - 1) + opt + [-1] * (padding_len + 1)
 
        tokens.append(seq)
        targets.append(mask_curr)

    return tokens, targets, label
        
block_size = 1024 # @TODO: assume this will come from some global config which has set the block size for the model - again not needed when RoPE is implemented

def eval_hellaswag(model, tokenizer, device, ddp, ddp_rank, ddp_world_size):
    #@NOTE: the model to be passed here should be raw in case ddp is not true   

    label_correct = []  # this will hold either 0 or 1 based on whether the opt_pred == label
    label_correct_avg = []  # this will hold either 0 or 1 based on whether the opt_pred_avg == label

    if ddp_rank == 0:
        print(f"Total sets - {len(ds)//ddp_world_size}")

    model.eval()

    for i, item in enumerate(ds):

        if i % ddp_world_size != ddp_rank:
            continue # only process the sets which are multiple of your alloted rank

        tokens, targets, label = render(item, tokenizer) # (4, T_full)
        assert len(tokens[0]) <= block_size, f"seq length can't be more than supported block size: {block_size}"  # again not needed when RoPE is implemented
        
        # wrap to tensors and move to device
        tokens, targets = torch.tensor(tokens, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
        tokens, targets = tokens.to(device), targets.to(device)
        
        with torch.no_grad():
            logits, _ = model(tokens) # this loss is not usefull because it will give a CE over all the batches

        # logits shape (4, T, C), targets (4, T)
        losses = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1, # this will assign 0 loss to tokens where target is -1
            reduction='none',
        ).view(logits.size(0), -1) # (4, T)   

        sum_loss = losses.sum(dim=1) # sum the loss for each option, -1 tokens have already 0 loss
        counts = (targets != -1).sum(dim=1)   # (4,)
        avg_loss = sum_loss / counts

        opt_pred, opt_pred_avg = sum_loss.argmin(), avg_loss.argmin()

        label_correct.append(int(opt_pred.item()==label))
        label_correct_avg.append(int(opt_pred_avg.item()==label))

        if i % 500 == 0 and i != 0 and ddp_rank == 0:
            print(f"{i} texts evaluated")

    # print(f"label correct - {label_correct}")
    # print(f"label correct avg - {label_correct_avg}")

    stats = torch.tensor([len(label_correct), sum(label_correct), sum(label_correct_avg)], dtype=torch.long, device=device)

    if ddp: # if master process
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    if ddp_rank == 0:
        total, correct, correct_avg = stats.tolist()
        accuracy = correct / total
        accuracy_avg = correct_avg / total

        print(f"accuracy of hellaswag eval over tokens: {accuracy*100} and on per token: {accuracy_avg*100}")

    return accuracy, accuracy_avg


# test script
def main():
    from train_gpt2 import GPT, GPTConfig
    import tiktoken

    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    
    enc = tiktoken.get_encoding('gpt2')
    ddp_rank = 0  # for mps check
    ddp_world_size = 1 # for mps check

    # model = GPT.from_pretrained("gpt2")
    # model = model.to(device)
    # eval_hellaswag(model, enc, device, ddp=False, ddp_rank=ddp_rank, ddp_world_size=ddp_world_size) # evaluate on GPT-2 124M model
    # print(f"Hellaswag evaluation completed for gpt2")

    print(f"Hellaswag evaluation on initialized tinygpt model")
    model = GPT(GPTConfig(block_size=block_size, vocab_size=50304))
    model = model.to(device)

    accuracy, accuracy_avg = eval_hellaswag(model, enc, device, ddp=False, ddp_rank=ddp_rank, ddp_world_size=ddp_world_size) # evaluate on our initialized model
    print(f"Hellaswag evaluation completed for gpt2 self")

if __name__ == "__main__":
    main()


