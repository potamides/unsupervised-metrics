import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from collections import defaultdict

def padding(arr, pad_token, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, :lens[i]] = 1
    return padded, mask

def collate_idf(arr, tokenize, numericalize):
    tokens = [["[CLS]"] + tokenize(a) + ["[SEP]"] for a in arr]
    arr = [numericalize(a) for a in tokens]
    idf_dict = defaultdict(lambda: 1.)
    idf_weights = [[idf_dict[i] for i in a] for a in arr]
    pad_token = numericalize(["[PAD]"])[0]
    padded, mask = padding(arr, pad_token, dtype=torch.long)
    padded_idf, _ = padding(idf_weights, pad_token, dtype=torch.float)

    return padded, padded_idf, mask, tokens

def embed(all_sens, bs, model, tokenizer, device):
    all_embeddings = list()
    padded_sens, padded_idf, mask, tokens = collate_idf(all_sens, tokenizer.tokenize,
            tokenizer.convert_tokens_to_ids)
    data = TensorDataset(padded_sens, mask)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=bs)

    model.eval()
    with torch.no_grad():
        for (batch_padded_sens, batch_mask) in dataloader:
            batch_padded_sens.to(device)
            batch_mask.to(device)
            all_embeddings.append(model(batch_padded_sens, batch_mask)["last_hidden_state"].cpu())
    return torch.cat(all_embeddings), padded_idf, tokens, mask.unsqueeze(-1)
