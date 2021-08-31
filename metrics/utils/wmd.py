import numpy as np
import torch
from torch.nn.functional import cosine_similarity
import string
from pyemd import emd

def pairwise_distances(x, y=None):
    x_norm = (x**2).sum(1).view(-1, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    y_t = torch.transpose(y, 0, 1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)

def slide_window(input_, w=3, o=1):
    if input_.size - w + 1 <= 0:
        w = input_.size
    sh = (input_.size - w + 1, w)
    st = input_.strides * 2
    view = np.lib.stride_tricks.as_strided(input_, strides = st, shape = sh)[0::o]
    return view.copy().tolist()

def _safe_divide(numerator, denominator):
    return numerator / (denominator + 1e-30)

def load_ngram(tokens, embedding, idf, n_gram, suffix_filter=True):
    new_a = []
    new_idf = []
    ids = [k for k, w in enumerate(tokens) if not suffix_filter or (w not in set(string.punctuation) and '##' not in w)]

    slide_wins = slide_window(np.array(ids), w=n_gram)
    for slide_win in slide_wins:
        new_idf.append(idf[slide_win].sum().item())
        scale = _safe_divide(idf[slide_win], idf[slide_win].sum(0)).unsqueeze(-1)
        tmp =  (scale * embedding[slide_win]).sum(0)
        new_a.append(tmp)
    new_a = torch.stack(new_a, 0)
    return new_a, new_idf

def compute_score(src_embedding_ngrams, src_idf_ngrams, tgt_embedding_ngrams, tgt_idf_ngrams, use_cosine=False):
    embeddings = torch.cat([src_embedding_ngrams, tgt_embedding_ngrams], 0)
    embeddings.div_(torch.norm(embeddings, dim=-1).unsqueeze(-1) + 1e-30)
    if use_cosine:
        distance_matrix = cosine_similarity(embeddings[:, None, :], embeddings[None, :, :], dim=2)
    else:
        distance_matrix = pairwise_distances(embeddings, embeddings)

    c1 = np.zeros(len(src_idf_ngrams) + len(tgt_idf_ngrams))
    c2 = np.zeros_like(c1)

    c1[:len(src_idf_ngrams)] = src_idf_ngrams
    c2[-len(tgt_idf_ngrams):] = tgt_idf_ngrams

    return -emd(_safe_divide(c1, np.sum(c1)), _safe_divide(c2, np.sum(c2)), distance_matrix.double().numpy())

def word_mover_align(source_data, target_data, n_gram, candidates=None, use_cosine=False, suffix_filter=True):
    src_embedding_ngrams, src_idf_ngrams = list(), list()
    for embedding, idf, tokens in zip(*source_data):
        embedding_ngrams, idf_ngrams = load_ngram(tokens, embedding, idf, n_gram, suffix_filter)
        src_embedding_ngrams.append(embedding_ngrams)
        src_idf_ngrams.append(idf_ngrams)

    tgt_embedding_ngrams, tgt_idf_ngrams = list(), list()
    for embedding, idf, tokens in zip(*target_data):
        embedding_ngrams, idf_ngrams = load_ngram(tokens, embedding, idf, n_gram, suffix_filter)
        tgt_embedding_ngrams.append(embedding_ngrams)
        tgt_idf_ngrams.append(idf_ngrams)

    pairs, scores = list(), list()
    for src_index in range(len(src_embedding_ngrams)):
        best_score = float("-inf")
        best_tgt_index = -1
        # use only the nearest neighbors, when they are provided
        for tgt_index in range(len(tgt_embedding_ngrams)) if candidates is None else candidates[src_index]:
            batch_src_embedding_ngrams = src_embedding_ngrams[src_index]
            batch_src_idf_ngrams = src_idf_ngrams[src_index]
            batch_tgt_embedding_ngrams = tgt_embedding_ngrams[tgt_index]
            batch_tgt_idf_ngrams = tgt_idf_ngrams[tgt_index]
            score = compute_score(batch_src_embedding_ngrams, batch_src_idf_ngrams,
                    batch_tgt_embedding_ngrams, batch_tgt_idf_ngrams, use_cosine)
            if score > best_score:
                best_score = score
                best_tgt_index = tgt_index

        pairs.append((src_index, best_tgt_index))
        scores.append(best_score)

    return pairs, scores

def word_mover_score(source_data, target_data, n_gram, use_cosine=False, suffix_filter=True):
    src_embedding_ngrams, src_idf_ngrams = list(), list()
    for embedding, idf, tokens in zip(*source_data):
        embedding_ngrams, idf_ngrams = load_ngram(tokens, embedding, idf, n_gram, suffix_filter)
        src_embedding_ngrams.append(embedding_ngrams)
        src_idf_ngrams.append(idf_ngrams)

    tgt_embedding_ngrams, tgt_idf_ngrams = list(), list()
    for embedding, idf, tokens in zip(*target_data):
        embedding_ngrams, idf_ngrams = load_ngram(tokens, embedding, idf, n_gram, suffix_filter)
        tgt_embedding_ngrams.append(embedding_ngrams)
        tgt_idf_ngrams.append(idf_ngrams)

    scores = list()
    for data in zip(src_embedding_ngrams, src_idf_ngrams, tgt_embedding_ngrams, tgt_idf_ngrams):
        scores.append(compute_score(*data, use_cosine))

    return scores
