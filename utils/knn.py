import faiss
import numpy as np

def knn_sharded(source_data, target_data, k, batch_size, device):
    sims = []
    inds = []
    dim = source_data.shape[-1]
    xfrom = 0

    for x_batch in np.array_split(source_data, np.ceil(len(source_data) / batch_size)):
        yfrom = 0
        bsims, binds = [], []
        for y_batch in np.array_split(target_data, np.ceil(len(target_data) / batch_size)):
            neighbor_size = min(k, y_batch.shape[0])
            idx = faiss.IndexFlatIP(dim)
            if device != 'cpu':
                idx = faiss.index_cpu_to_all_gpus(idx)
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

def find_nearest_neighbors(source_data, target_data, k, batch_size, device):
    _, indeces = knn_sharded(source_data.cpu().numpy(), target_data.cpu().numpy(), k, batch_size, device)
    return indeces
