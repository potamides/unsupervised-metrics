import faiss
import logging
import numpy as np
from os import environ, path
from pathlib import Path
from LASER.source.lib.text_processing import Token, BPEfastApply
from LASER.source.embed import SentenceEncoder, EncodeFile

environ["LASER"] = (Path(__file__).parent / "LASER").resolve()

def load_batch(emb_file, dim):
    embeddings = np.fromfile(emb_file, dtype=np.float32)
    num_rows = int(embeddings.shape[0] / dim)
    embeddings = embeddings.reshape((num_rows, dim))
    faiss.normalize_L2(embeddings)
    return embeddings

def embed_batch(content, tmpdir=(Path(__file__).parent / "data").resolve(), lang="en"):
    model_dir = path.join(environ.get("LASER"), "models")
    encoder_path = path.join(model_dir, "bilstm.93langs.2018-12-26.pt")
    bpe_codes_path = path.join(model_dir, "93langs.fcodes")
    print(f' - Encoder: loading {encoder_path}')
    encoder = SentenceEncoder(encoder_path,
                              max_sentences=None,
                              max_tokens=12000,
                              sort_kind='mergesort',
                              cpu=True)
    ifname = path.join(tmpdir, "content.txt")
    bpe_fname = path.join(tmpdir, 'bpe')
    bpe_oname = path.join(tmpdir, 'out.raw')
    with ifname.open("w") as f:
        f.write(content)
    if lang != '--':
        tok_fname = path.join(tmpdir, "tok")
        Token(ifname,
              tok_fname,
              lang=lang,
              romanize=True if lang == 'el' else False,
              lower_case=True,
              gzip=False,
              verbose=True,
              over_write=False)
        ifname = tok_fname
    BPEfastApply(ifname,
                 bpe_fname,
                 bpe_codes_path,
                 verbose=True, over_write=False)
    ifname = bpe_fname
    EncodeFile(encoder,
               ifname,
               bpe_oname,
               verbose=True,
               over_write=False,
               buffer_size=10000)
    dim = 1024
    embedding = np.fromfile(bpe_oname, dtype=np.float32, count=-1)
    embedding.resize(embedding.shape[0] // dim, dim)
    return embedding

def knn_sharded(x_batches_f, y_batches_f, dim, k, device):
    sims = []
    inds = []
    xfrom = 0
    xto = 0
    for x_batch_f in x_batches_f:
        yfrom = 0
        yto = 0
        x_batch = load_batch(x_batch_f, dim)
        xto = xfrom + x_batch.shape[0]
        bsims, binds = [], []
        for y_batch_f in y_batches_f:
            y_batch = load_batch(y_batch_f, dim)
            neighbor_size = min(k, y_batch.shape[0])
            yto = yfrom + y_batch.shape[0]
            logging.info("{}-{}  ->  {}-{}".format(xfrom, xto, yfrom, yto))
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
