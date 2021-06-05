from faiss import IndexFlatL2, IndexFlatIP, index_cpu_to_all_gpus, normalize_L2 
import numpy as np

# Adopted from https://github.com/pytorch/fairseq/blob/master/examples/criss/mining/mine.py
def knn_sharded(source_data, target_data, k, batch_size, device, use_cosine=False):
    if use_cosine:
        normalize_L2(source_data)
        normalize_L2(target_data)
    sims = []
    inds = []
    dim = source_data.shape[-1]
    xfrom = 0

    for x_batch in np.array_split(source_data, np.ceil(len(source_data) / batch_size)):
        yfrom = 0
        bsims, binds = [], []
        for y_batch in np.array_split(target_data, np.ceil(len(target_data) / batch_size)):
            neighbor_size = min(k, y_batch.shape[0])
            idx = IndexFlatIP(dim) if use_cosine else IndexFlatL2(dim)
            if device != 'cpu':
                idx = index_cpu_to_all_gpus(idx)
            idx.add(y_batch)
            bsim, bind = idx.search(x_batch, neighbor_size)

            bsims.append(bsim)
            binds.append(bind + yfrom)
            yfrom += y_batch.shape[0]
            del idx
            del y_batch
        bsims = np.concatenate(bsims, axis=1)
        binds = np.concatenate(binds, axis=1)
        aux = np.argsort(-bsims, axis=1)
        sim_batch = np.zeros((x_batch.shape[0], k), dtype=np.float32)
        ind_batch = np.zeros((x_batch.shape[0], k), dtype=np.int64)
        for i in range(x_batch.shape[0]):
            for j in range(k):
                sim_batch[i, j] = bsims[i, aux[i, j]]
                ind_batch[i, j] = binds[i, aux[i, j]]
        sims.append(sim_batch)
        inds.append(ind_batch)
        xfrom += x_batch.shape[0]
        del x_batch
    sim = np.concatenate(sims, axis=0)
    ind = np.concatenate(inds, axis=0)
    return sim, ind

def score_candidates(sim_mat, candidate_inds, fwd_mean, bwd_mean):
    scores = np.zeros(candidate_inds.shape)
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            k = int(candidate_inds[i, j])
            scores[i, j] = sim_mat[i, j] / ((fwd_mean[i] + bwd_mean[k]) / 2)
    return scores

def ratio_margin_align(source_data, target_data, k, batch_size, device):
    src2tgt_sim, src2tgt_ind = knn_sharded(source_data.numpy(), target_data.numpy(), k, batch_size, device, True)
    tgt2src_sim, _ = knn_sharded(target_data.numpy(), source_data.numpy(), k, batch_size, device)

    src2tgt_mean = src2tgt_sim.mean(axis=1)
    tgt2src_mean = tgt2src_sim.mean(axis=1)
    fwd_scores = score_candidates(src2tgt_sim, src2tgt_ind, src2tgt_mean, tgt2src_mean)
    fwd_best = src2tgt_ind[np.arange(src2tgt_sim.shape[0]), fwd_scores.argmax(axis=1)]

    return fwd_best, fwd_scores.max(axis=1)

def wcd_align(source_data, target_data, k, batch_size, device):
    squared_scores, indeces = knn_sharded(source_data.numpy(), target_data.numpy(), k, batch_size, device)
    return indeces, np.sqrt(squared_scores)

def cosine_align(source_data, target_data, k, batch_size, device):
    scores, indeces = knn_sharded(source_data.numpy(), target_data.numpy(), k, batch_size, device, True)
    return indeces, scores
