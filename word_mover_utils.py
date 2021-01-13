import numpy as np
import torch
import string
import logging
from pyemd import emd
from collections import defaultdict

def padding(arr, pad_token, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, :lens[i]] = 1
    return padded, lens, mask

def bert_encode(model, x, attention_mask):
    model.eval()
    with torch.no_grad():
        return model(input_ids=x, attention_mask=attention_mask)["hidden_states"]

def collate_idf(arr, tokenize, numericalize, idf_dict, device):
    tokens = [["[CLS]"]+tokenize(a)+["[SEP]"] for a in arr]
    arr = [numericalize(a) for a in tokens]

    idf_weights = [[idf_dict[i] for i in a] for a in arr]

    pad_token = numericalize(["[PAD]"])[0]

    padded, lens, mask = padding(arr, pad_token, dtype=torch.long)
    padded_idf, _, _ = padding(idf_weights, pad_token, dtype=torch.float)

    padded = padded.to(device=device)
    mask = mask.to(device=device)
    lens = lens.to(device=device)
    return padded, padded_idf, lens, mask, tokens

def get_bert_embedding(all_sens, model, tokenizer, idf_dict, device):
    padded_sens, padded_idf, lens, mask, tokens = collate_idf(all_sens,
                                                      tokenizer.tokenize, tokenizer.convert_tokens_to_ids,
                                                      idf_dict,
                                                      device)
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_sens), len(all_sens)):
            batch_embedding = bert_encode(model, padded_sens[i:i+len(all_sens)],
                                          attention_mask=mask[i:i+len(all_sens)])
            batch_embedding = torch.stack(batch_embedding)
            embeddings.append(batch_embedding)
            del batch_embedding

    total_embedding = torch.cat(embeddings, dim=-3)
    return total_embedding, lens, mask, padded_idf, tokens

def pairwise_distances(x, y=None):
    x_norm = (x**2).sum(1).view(-1, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    y_t = torch.transpose(y, 0, 1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)

def slide_window(input_, w=3, o=2):
    if input_.size - w + 1 <= 0:
        w = input_.size
    sh = (input_.size - w + 1, w)
    st = input_.strides * 2
    view = np.lib.stride_tricks.as_strided(input_, strides = st, shape = sh)[0::o]
    return view.copy().tolist()

def _safe_divide(numerator, denominator):
    return numerator / (denominator + 1e-30)

def load_ngram(ids, embedding, idf, n, o, device):
    new_a = []
    new_idf = []

    slide_wins = slide_window(np.array(ids), w=n, o=o)
    for slide_win in slide_wins:
        new_idf.append(idf[slide_win].sum().item())
        scale = _safe_divide(idf[slide_win], idf[slide_win].sum(0)).unsqueeze(-1).to(device)
        tmp =  (scale * embedding[slide_win]).sum(0)
        new_a.append(tmp)
    new_a = torch.stack(new_a, 0).to(device)
    return new_a, new_idf


def embedd_batch(model, tokenizer, batch, n_gram, device):
    idf_dict_src = defaultdict(lambda: 1.)
    embedding, _, _, idf, tokens = get_bert_embedding(batch, model, tokenizer, idf_dict_src, device)
    embedding = embedding[-1]

    embedding_ngrams, idf_ngrams = list(), list()
    for i in range(len(batch)):
        ids = [k for k, w in enumerate(tokens[i]) if w not in set(string.punctuation) and '##' not in w]
        embedding_ngram, idf_ngram = load_ngram(ids, embedding[i], idf[i], n_gram, 1, device)
        embedding_ngrams.append(embedding_ngram)
        idf_ngrams.append(idf_ngram)

    return embedding_ngrams, idf_ngrams

def compute_score(src_embedding_ngrams, src_idf_ngrams, tgt_embedding_ngrams, tgt_idf_ngrams):
    embeddings = torch.cat([src_embedding_ngrams, tgt_embedding_ngrams], 0)
    embeddings.div_(torch.norm(embeddings, dim=-1).unsqueeze(-1) + 1e-30)
    distance_matrix = pairwise_distances(embeddings, embeddings)

    c1 = np.zeros(len(src_idf_ngrams) + len(tgt_idf_ngrams))
    c2 = np.zeros_like(c1)

    c1[:len(src_idf_ngrams)] = src_idf_ngrams
    c2[-len(tgt_idf_ngrams):] = tgt_idf_ngrams

    score = 1 - emd(_safe_divide(c1, np.sum(c1)),
                _safe_divide(c2, np.sum(c2)),
                distance_matrix.double().cpu().numpy())

    return score

def word_mover_align(model, tokenizer, source_data, target_data, n_gram, batch_size, device, candidates=None):
    logging.info("Embedding source sentences with mBERT.")
    src_embedding_ngrams, src_idf_ngrams = list(), list()
    for src_batch_start in range(0, len(source_data), batch_size):
        batch_src = source_data[src_batch_start:src_batch_start+batch_size]
        batch_src_embedding_ngrams, batch_src_idf_ngrams = embedd_batch(model, tokenizer, batch_src, n_gram, device)
        src_embedding_ngrams.extend(batch_src_embedding_ngrams)
        src_idf_ngrams.extend(batch_src_idf_ngrams)

    logging.info("Embedding target sentences with mBERT.")
    tgt_embedding_ngrams, tgt_idf_ngrams = list(), list()
    for tgt_batch_start in range(0, len(target_data), batch_size):
        batch_tgt = target_data[tgt_batch_start:tgt_batch_start+batch_size]
        batch_tgt_embedding_ngrams, batch_tgt_idf_ngrams = embedd_batch(model, tokenizer, batch_tgt, n_gram, device)
        tgt_embedding_ngrams.extend(batch_tgt_embedding_ngrams)
        tgt_idf_ngrams.extend(batch_tgt_idf_ngrams)

    logging.info("Computing word mover scores.")
    pairs = list()
    for src_index in range(len(source_data)):
        best_score = 0
        best_tgt_index = -1
        # use only the nearest neighbors, when they are provided
        for tgt_index in range(len(target_data)) if candidates is None else candidates[src_index]:
            batch_src_embedding_ngrams = src_embedding_ngrams[src_index]
            batch_src_idf_ngrams = src_idf_ngrams[src_index]
            batch_tgt_embedding_ngrams = tgt_embedding_ngrams[tgt_index]
            batch_tgt_idf_ngrams = tgt_idf_ngrams[tgt_index]
            score = compute_score(batch_src_embedding_ngrams, batch_src_idf_ngrams,
                    batch_tgt_embedding_ngrams, batch_tgt_idf_ngrams)
            if score > best_score:
                best_score = score
                best_tgt_index = tgt_index

        pairs.append((source_data[src_index], target_data[best_tgt_index]))

    return pairs
