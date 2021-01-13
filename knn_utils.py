import faiss
import logging
import numpy as np
from os import environ, path, remove
from pathlib import Path
from contextlib import suppress
# required for LASER to work
environ["LASER"] = str((Path(__file__).parent / "LASER").resolve())
from LASER.source.lib.text_processing import Token, BPEfastApply
from LASER.source.embed import SentenceEncoder, EncodeFile

dim = 1024

def embed(content, tmpdir=(Path(__file__).parent / "data").resolve(), lang="en"):
    model_dir = path.join(environ.get("LASER"), "models")
    encoder_path = path.join(model_dir, "bilstm.93langs.2018-12-26.pt")
    bpe_codes_path = path.join(model_dir, "93langs.fcodes")
    logging.info(f' - Encoder: loading {encoder_path}')
    encoder = SentenceEncoder(encoder_path,
                              max_sentences=None,
                              max_tokens=12000,
                              sort_kind='mergesort',
                              cpu=True)
    ifname = path.join(tmpdir, "content.txt")
    tok_fname = path.join(tmpdir, "tok")
    bpe_fname = path.join(tmpdir, 'bpe')
    bpe_oname = path.join(tmpdir, 'out.raw')

    with open(ifname, "wb") as f:
        f.write('\n'.join(content).encode())
    with suppress(FileNotFoundError):
        remove(tok_fname)
        remove(bpe_fname)
        remove(bpe_oname)
    if lang != '--':
        Token(ifname,
              tok_fname,
              lang=lang,
              romanize=True if lang == 'el' else False,
              lower_case=True,
              gzip=False)
        ifname = tok_fname
    BPEfastApply(ifname,
                 bpe_fname,
                 bpe_codes_path)
    ifname = bpe_fname
    EncodeFile(encoder,
               ifname,
               bpe_oname,
               buffer_size=10000)
    # TODO: implement lazy loading for larger datasets
    embedding = np.fromfile(bpe_oname, dtype=np.float32, count=-1)
    embedding.resize(embedding.shape[0] // dim, dim)
    return embedding

def knn_sharded(source_data, target_data, source_lang, target_lang, k, batch_size, device):
    sims = []
    inds = []
    xfrom = 0
    xto = 0
    logging.info("Embedding source sentences with LASER.")
    source_embeddings = embed(source_data, lang=source_lang)
    logging.info("Embedding target sentences with LASER.")
    target_embeddings = embed(target_data, lang=target_lang)

    logging.info("Finding the {} nearest neighbors of {} {} sentences in {} {} sentences.".format(
        k, len(source_data), source_lang, len(target_data), target_lang))
    for x_batch in np.array_split(source_embeddings, np.ceil(len(source_embeddings) / batch_size)):
        yfrom = 0
        yto = 0
        xto = xfrom + x_batch.shape[0]
        bsims, binds = [], []
        for y_batch in np.array_split(target_embeddings, np.ceil(len(target_embeddings) / batch_size)):
            neighbor_size = min(k, y_batch.shape[0])
            yto = yfrom + y_batch.shape[0]
            logging.info(f"Comparing: {xfrom}-{xto}  ->  {yfrom}-{yto}")
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

def find_nearest_neighbors(source_data, target_data, source_lang, target_lang, k, batch_size, device):
    _, indeces = knn_sharded(source_data, target_data, source_lang, target_lang, k, batch_size, device)
    return indeces
