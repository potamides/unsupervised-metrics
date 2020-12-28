from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import string
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
        _, _, x_encoded_layers, _ = model(input_ids = x, token_type_ids = None, attention_mask = attention_mask)
    return x_encoded_layers

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

def compute_scores_for_batch(model, tokenizer, batch_src, batch_tgt, n_gram,  device):
    idf_dict_src = defaultdict(lambda: 1.)
    idf_dict_tgt = defaultdict(lambda: 1.)
    src_embedding, _, _, src_idf, src_tokens = get_bert_embedding(batch_src, model, tokenizer, idf_dict_src, device)
    tgt_embedding, _, _, tgt_idf, tgt_tokens = get_bert_embedding(batch_tgt, model, tokenizer, idf_dict_tgt, device)
    src_embedding = src_embedding[-1]
    tgt_embedding = tgt_embedding[-1]

    src_embedding_ngrams, src_idf_ngrams = list(), list()
    for i in range(len(batch_src)):
        src_ids = [k for k, w in enumerate(src_tokens[i]) if w not in set(string.punctuation) and '##' not in w]
        src_embedding_ngram, src_idf_ngram = load_ngram(src_ids, src_embedding[i], src_idf[i], n_gram, 1, device)
        src_embedding_ngrams.append(src_embedding_ngram)
        src_idf_ngrams.append(src_idf_ngram)

    tgt_embedding_ngrams, tgt_idf_ngrams = list(), list()
    for j in range(len(batch_tgt)):
        tgt_ids = [k for k, w in enumerate(tgt_tokens[j]) if w not in set(string.punctuation) and '##' not in w]
        tgt_embedding_ngram, tgt_idf_ngram = load_ngram(tgt_ids, tgt_embedding[j], tgt_idf[j], n_gram, 1, device)
        tgt_embedding_ngrams.append(tgt_embedding_ngram)
        tgt_idf_ngrams.append(tgt_idf_ngram)

    score_matrix = torch.zeros(len(batch_src), len(batch_tgt))
    for i in range(len(batch_src)):
        for j in range(len(batch_tgt)):
            embeddings = torch.cat([src_embedding_ngrams[i], tgt_embedding_ngrams[j]], 0)
            embeddings.div_(torch.norm(embeddings, dim=-1).unsqueeze(-1) + 1e-30)
            distance_matrix = pairwise_distances(embeddings, embeddings)

            c1 = np.zeros(len(src_idf_ngrams[i]) + len(tgt_idf_ngrams[j]))
            c2 = np.zeros_like(c1)

            c1[:len(src_idf_ngrams[i])] = src_idf_ngrams[i]
            c2[-len(tgt_idf_ngrams[j]):] = tgt_idf_ngrams[j]

            score = 1 - emd(_safe_divide(c1, np.sum(c1)),
                        _safe_divide(c2, np.sum(c2)),
                        distance_matrix.double().cpu().numpy())

            score_matrix[i][j] = score

    return score_matrix

def word_mover_align(model, tokenizer, source_data, target_data, n_gram, batch_size, device):

    pairs = []
    for src_batch_start in range(0, len(source_data), batch_size):
        batch_src = source_data[src_batch_start:src_batch_start+batch_size]
        best_matches = list()
        for tgt_batch_start in range(0, len(target_data), batch_size):
            batch_tgt = target_data[tgt_batch_start:tgt_batch_start+batch_size]

            score_matrix = compute_scores_for_batch(model, tokenizer, batch_src, batch_tgt, n_gram, device)
            for i in range(len(batch_src)):
                score, index = torch.max(score_matrix[i], 0)
                try:
                    if score > best_matches[i][0]:
                        best_matches[i] = (score, tgt_batch_start + index)
                except IndexError:
                        best_matches.append((score, tgt_batch_start + index))

        pairs.extend([(batch_src[src], target_data[tgt]) for src, (_, tgt) in enumerate(best_matches)])

    return pairs
