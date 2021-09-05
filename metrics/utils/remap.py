import numpy as np
import torch
from itertools import chain
from subprocess import check_output, DEVNULL
from tempfile import NamedTemporaryFile as TempFile
from simalign import SentenceAligner
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from .env import DATADIR

def convert_sent_to_input(sents, tokenizer, max_seq_length):
    input_ids = []
    mask = []
    for sent in sents:
        ids = tokenizer.convert_tokens_to_ids(sent)
        mask.append([1] * (len(ids) + 2) + [0] * (max_seq_length - len(ids)))
        input_ids.append([101] + ids + [102] + [0] * (max_seq_length - len(ids)))
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

def convert_words_to_bpe(sent_pairs, tokenizer):
    bpe_para, bpe_table = [], []

    for (src_sent, tgt_sent) in sent_pairs:
        src_bpe_table, tgt_bpe_table = [], []
        src_sent_bpe, tgt_sent_bpe = [], []

        for word in src_sent:
            token = tokenizer.tokenize(word)
            word2bpe_map = []
            for i in range(len(token)):
                word2bpe_map.append(len(src_sent_bpe)+i)
            src_sent_bpe.extend(token)
            src_bpe_table.append(word2bpe_map)

        for word in tgt_sent:
            token = tokenizer.tokenize(word)
            word2bpe_map = []
            for i in range(len(token)):
                word2bpe_map.append(len(tgt_sent_bpe)+i)
            tgt_sent_bpe.extend(token)
            tgt_bpe_table.append(word2bpe_map)

        bpe_para.append([src_sent_bpe, tgt_sent_bpe])
        bpe_table.append([src_bpe_table, tgt_bpe_table])

    return bpe_para, bpe_table


def get_aligned_features_avgbpe(sent_pairs, align_pairs, model,
        tokenizer, batch_size, device, layer=12, max_seq_length=175):
    bpe_para, bpe_table = convert_words_to_bpe(sent_pairs, tokenizer)

    # filter long/empty sentences
    fltr_src_bpe, fltr_tgt_bpe, fltr_align_pairs, fltr_bpe_table, align_cnt = [], [], [], [], 0
    for cnt, (src, tgt) in enumerate(bpe_para):
        if len(src) <= max_seq_length and len(tgt) <= max_seq_length and len(src) > 0 and len(tgt) > 0:
            fltr_src_bpe.append(src)
            fltr_tgt_bpe.append(tgt)
            fltr_align_pairs.append(align_pairs[cnt])
            fltr_bpe_table.append(bpe_table[cnt])
            align_cnt += len(align_pairs[cnt])

    src_input, src_mask = convert_sent_to_input(fltr_src_bpe, tokenizer, max_seq_length)
    tgt_input, tgt_mask = convert_sent_to_input(fltr_tgt_bpe, tokenizer, max_seq_length)

    src_data = TensorDataset(src_input, src_mask)
    src_sampler = SequentialSampler(src_data)
    src_dataloader = DataLoader(src_data, sampler=src_sampler, batch_size=batch_size)

    tgt_data = TensorDataset(tgt_input, tgt_mask)
    tgt_sampler = SequentialSampler(tgt_data)
    tgt_dataloader = DataLoader(tgt_data, sampler=tgt_sampler, batch_size=batch_size)

    src_embed = []
    tgt_embed = []

    model.eval()
    with torch.no_grad():
        for batch in src_dataloader:
            input_ids, input_mask = batch
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)

            hidden_state = model(input_ids, attention_mask=input_mask)["hidden_states"][layer]
            src_embed.append(hidden_state[:,1:].cpu().numpy()) # remove CLS

    with torch.no_grad():
        for batch in tgt_dataloader:
            input_ids, input_mask = batch
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)

            hidden_state = model(input_ids, attention_mask=input_mask)["hidden_states"][layer]
            tgt_embed.append(hidden_state[:,1:].cpu().numpy())

    src_embed = np.concatenate(src_embed)
    tgt_embed = np.concatenate(tgt_embed)

    feature_size = src_embed.shape[2]
    cnt, src_matrix, tgt_matrix = 0, np.zeros((align_cnt, feature_size)), np.zeros((align_cnt, feature_size))
    for i, pairs in enumerate(fltr_align_pairs):
        for a in pairs:
            if len(fltr_bpe_table[i][0][a[0]]) > 0 and len(fltr_bpe_table[i][1][a[1]]) > 0: # token alignment (0,0)
                src_word_avg_embed = np.zeros((1, feature_size))

                for j in fltr_bpe_table[i][0][a[0]]:
                    src_word_avg_embed += src_embed[i][j,:]
                src_matrix[cnt,:] = src_word_avg_embed / len(fltr_bpe_table[i][0][a[0]])

                tgt_word_avg_embed = np.zeros((1, feature_size))
                for j in fltr_bpe_table[i][1][a[1]]:
                    tgt_word_avg_embed += tgt_embed[i][j,:]

                tgt_matrix[cnt,:] = tgt_word_avg_embed / len(fltr_bpe_table[i][1][a[1]])
                cnt += 1

    return src_matrix, tgt_matrix

def fast_align(sent_pairs, tokenizer, size, max_seq_length=100):
    tokenized_pairs = list()
    for source_sent, target_sent in sent_pairs:
        sent1 = tokenizer.basic_tokenizer.tokenize(source_sent)
        sent2 = tokenizer.basic_tokenizer.tokenize(target_sent)

        if 0 < len(sent1) <= max_seq_length and 0 < len(sent2) <= max_seq_length:
            tokenized_pairs.append((sent1, sent2))

        if len(tokenized_pairs) >= size:
            break

    with TempFile(dir=DATADIR, buffering=0) as fwd_file, TempFile(dir=DATADIR, buffering=0) as bwd_file:
        for file_, data, flags in ((fwd_file, tokenized_pairs, "-dov"), (bwd_file, tokenized_pairs, "-dovr")):
            file_.write("\n".join([f'{" ".join(src)} ||| {" ".join(tgt)}'.lower() for src, tgt in data]).encode())
            asym_aligned = check_output(["fast_align", "-i", file_.name, flags], stderr=DEVNULL)
            file_.seek(0)
            file_.truncate()
            file_.write(asym_aligned)

        sym_aligned = check_output(["atools", "-i", fwd_file.name, "-j", bwd_file.name, "-c", "grow-diag-final-and"])

    sym_aligned = [[tuple(map(int, pair.split(b"-"))) for pair in pairs.split()] for pairs in sym_aligned.splitlines()]
    return tokenized_pairs, sym_aligned

def awesome_align(sentpairs, model, tokenizer, size, device, projection=None, max_seq_length=100):
    tokenized_pairs, alignments = list(), list()
    for src, tgt in sentpairs:
        sent_src, sent_tgt = tokenizer.basic_tokenizer.tokenize(src), tokenizer.basic_tokenizer.tokenize(tgt)
        if 0 < len(sent_src) <= max_seq_length and 0 < len(sent_tgt) <= max_seq_length:
            token_src = [tokenizer.tokenize(word) for word in sent_src]
            token_tgt = [tokenizer.tokenize(word) for word in sent_tgt]
            wid_src = [tokenizer.convert_tokens_to_ids(x) for x in token_src]
            wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
            ids_src = tokenizer.prepare_for_model(list(chain(*wid_src)), return_tensors='pt', truncation=True)['input_ids']
            ids_tgt = tokenizer.prepare_for_model(list(chain(*wid_tgt)), return_tensors='pt', truncation=True)['input_ids']
            sub2word_map_src = []
            for i, word_list in enumerate(token_src):
                sub2word_map_src.extend([i] * len(word_list))
            sub2word_map_tgt = []
            for i, word_list in enumerate(token_tgt):
                sub2word_map_tgt.extend([i] * len(word_list))

            # alignment
            align_layer = 8
            threshold = 1e-3
            model.eval()
            with torch.no_grad():
                out_src = model(ids_src.unsqueeze(0).to(device))["hidden_states"][align_layer]
                out_tgt = model(ids_tgt.unsqueeze(0).to(device))["hidden_states"][align_layer]

                if projection is not None:
                    projection = projection.to(device)
                    if projection.ndim == 2: # CLP
                        out_src = torch.matmul(out_src, projection)
                    else: # UMD
                        out_src = out_src - (out_src * projection).sum(2, keepdim=True) * \
                                projection.repeat(out_src.shape[0], out_src.shape[1], 1)

                dot_prod = torch.matmul(out_src[0, 1:-1].cpu(), out_tgt[0, 1:-1].transpose(-1, -2).cpu())

                softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
                softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

                softmax_inter = (softmax_srctgt > threshold)*(softmax_tgtsrc > threshold)

            align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
            align_words = set()
            for i, j in align_subwords:
                align_words.add((sub2word_map_src[i], sub2word_map_tgt[j]))

            tokenized_pairs.append((sent_src, sent_tgt))
            alignments.append(list(align_words))

            if len(tokenized_pairs) >= size:
                break

    return tokenized_pairs, alignments

def sim_align(sent_pairs, tokenizer, size, device, max_seq_length=100):
    tokenized_pairs, alignments = list(), list()
    aligner = SentenceAligner(matching_methods="i", token_type="word", device=device)
    for source_sent, target_sent in sent_pairs:
        sent1 = tokenizer.basic_tokenizer.tokenize(source_sent)
        sent2 = tokenizer.basic_tokenizer.tokenize(target_sent)

        if 0 < len(sent1) <= max_seq_length and 0 < len(sent2) <= max_seq_length:
            tokenized_pairs.append((sent1, sent2))
            alignments.append(aligner.get_word_aligns(sent1, sent2)["itermax"])

        if len(tokenized_pairs) >= size:
            break

    return tokenized_pairs, alignments

def clp(x, z, orthogonal=True):
    if orthogonal:
        u, _, vt = np.linalg.svd(z.T.dot(x))
        w = vt.T.dot(u.T)
    else:
        x_pseudoinv = np.linalg.inv(x.T.dot(x)).dot(x.T)
        w = x_pseudoinv.dot(z)
    return torch.Tensor(w)

def umd(x, z):
    *_, v = np.linalg.svd(x - z)
    v_b = v[0]
    return torch.Tensor(v_b)
