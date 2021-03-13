import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from collections import defaultdict
from sys import path, argv
from os.path import join, dirname, abspath, isfile
from shutil import copyfileobj
from fasttext import tokenize
from urllib.request import urlretrieve
from gzip import open as gopen
from tempfile import NamedTemporaryFile as TempFile
path.append(abspath(join(dirname(__file__), 'vecmap')))
from .vecmap.map_embeddings import main as vecmap

fasttext_url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/"
datadir = str(abspath(join(dirname(__file__), '../data')))

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

def bert_embed(all_sens, batch_size, model, tokenizer, device):
    padded_sens, padded_idf, mask, tokens = collate_idf(all_sens, tokenizer.tokenize,
            tokenizer.convert_tokens_to_ids)
    data = TensorDataset(padded_sens, mask)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    all_embeddings = torch.zeros((len(all_sens), mask.shape[1], model.config.hidden_size))

    model.eval()
    with torch.no_grad():
        for batch_index, (batch_padded_sens, batch_mask) in enumerate(dataloader):
            pos = batch_index * batch_size
            batch_padded_sens = batch_padded_sens.to(device)
            batch_mask = batch_mask.to(device)
            all_embeddings[pos:pos + len(batch_mask)] = model(batch_padded_sens, batch_mask)["last_hidden_state"].cpu()
    return all_embeddings, padded_idf, tokens, mask.unsqueeze(-1)

def map_multilingual_embeddings(src_lang, tgt_lang, batch_size, device):
    src_emb = get_embeddings_file(src_lang)
    tgt_emb = get_embeddings_file(tgt_lang)
    src_dict, tgt_dict = defaultdict(lambda: torch.zeros(300)), defaultdict(lambda: torch.zeros(300))
    global argv

    # forgive me lord, for i have sinned
    with TempFile(dir=datadir, buffering=0) as src_map, TempFile(dir=datadir, buffering=0) as tgt_map:
        arguments = ['--batch_size', str(batch_size), '--unsupervised', src_emb, tgt_emb, src_map.name, tgt_map.name]
        if "cuda" in device:
            arguments.insert(0, '--cuda')
        argv.extend(arguments)
        vecmap()
        argv = argv[:-len(arguments)]

        for dict_, map_file in ((src_dict, src_map), (tgt_dict, tgt_map)):
            for line in map_file.readlines()[1:]:
                tokens = line.decode().rstrip().split(' ')
                dict_[tokens[0]] = torch.tensor(list(map(float, tokens[1:])))

    return src_dict, tgt_dict

def get_embeddings_file(lang_id):
    filename = f"cc.{lang_id}.300.vec"
    gz_filename = filename + ".gz"

    if isfile(join(datadir, filename)):
        return join(datadir, filename)

    urlretrieve(join(fasttext_url, gz_filename), join(datadir, gz_filename))

    with gopen(join(datadir, gz_filename), 'rb') as f:
        with open(join(datadir, filename), 'wb') as f_out:
            copyfileobj(f, f_out)

    return join(datadir, filename)

def vecmap_embed(all_sents, lang_dict):
    tokens, idf_weights, embeddings = list(), list(), list()
    for sent in all_sents:
        tokens.append([word for word in tokenize(sent)])
        idf_weights.append([1] * len(tokens[-1]))
        embeddings.append(torch.stack([lang_dict[word] for word in tokens[-1]]))

    idf_weights, mask = padding(idf_weights, 0, dtype=torch.float)
    embeddings = pad_sequence(embeddings, batch_first=True)

    return embeddings, idf_weights, tokens, mask.unsqueeze(-1)
