from ..utils.wmd import word_mover_align, word_mover_score
from ..utils.knn import wcd_align, ratio_margin_align, cosine_align
from ..utils.nmt import train, translate
from ..utils.perplexity import lm_perplexity
from ..utils.dataset import DATADIR
from os.path import isfile, join
from json import dumps
from math import ceil
from numpy import arange, array
from nltk.metrics.distance import edit_distance
from ..common import CommonScore
from re import findall
import logging
import torch

class XMoverAlign(CommonScore):
    def __init__(self, device, k, n_gram, knn_batch_size, use_cosine, align_batch_size):
        self.device = device
        self.k = k
        self.n_gram = n_gram
        self.knn_batch_size = knn_batch_size
        self.use_cosine = use_cosine
        self.align_batch_size = align_batch_size

    def _mean_pool_embed(self, source_sents, target_sents):
        source_sent_embeddings = torch.empty(len(source_sents), 768)
        target_sent_embeddings = torch.empty(len(target_sents), 768)
        idx = 0
        while idx < max(len(source_sents), len(target_sents)):
            src_embeddings, _, _, src_mask, tgt_embeddings, _, _, tgt_mask = self._embed(
                source_sents[idx:idx + self.align_batch_size], target_sents[idx:idx + self.align_batch_size])
            source_sent_embeddings[idx:idx + len(src_embeddings)] = torch.sum(src_embeddings * src_mask, 1) / torch.sum(src_mask, 1)
            target_sent_embeddings[idx:idx + len(tgt_embeddings)] = torch.sum(tgt_embeddings * tgt_mask, 1) / torch.sum(tgt_mask, 1)
            idx += self.align_batch_size

        return source_sent_embeddings, target_sent_embeddings

    def _memory_efficient_word_mover_align(self, source_sents, target_sents, candidates):
        pairs, scores, idx, k = list(), list(), 0, candidates.shape[1]
        batch_size = ceil(self.align_batch_size / k)
        while idx < len(source_sents):
            src_embeddings, src_idf, src_tokens, _, tgt_embeddings, tgt_idf, tgt_tokens, _ = self._embed(
                source_sents[idx:idx + batch_size],
                [target_sents[candidate] for candidate in candidates[idx:idx + batch_size].flatten()])
            batch_pairs, batch_scores = word_mover_align((src_embeddings, src_idf, src_tokens),
                (tgt_embeddings, tgt_idf, tgt_tokens), self.n_gram,
                arange(len(src_embeddings) * k).reshape(len(src_embeddings), k))
            pairs.extend([(src + idx, candidates[idx:idx + batch_size].flatten()[tgt]) for src, tgt in batch_pairs])
            scores.extend(batch_scores)
            idx += batch_size
        return pairs, scores

    def align(self, source_sents, target_sents):
        candidates = None
        logging.info("Obtaining sentence embeddings.")
        source_sent_embeddings, target_sent_embeddings = self._mean_pool_embed(source_sents, target_sents)
        logging.info("Searching for nearest neighbors.")
        if self.use_cosine:
            candidates, _ = cosine_align(source_sent_embeddings, target_sent_embeddings, self.k,
                    self.knn_batch_size, self.device)
        else:
            candidates, _ = wcd_align(source_sent_embeddings, target_sent_embeddings, self.k,
                    self.knn_batch_size, self.device)

        logging.info("Filter best nearest neighbors with Word Mover's Distance.")
        pairs, scores = self._memory_efficient_word_mover_align(source_sents, target_sents, candidates)
        sent_pairs = [(source_sents[src_idx], target_sents[tgt_idx]) for src_idx, tgt_idx in pairs]
        return sent_pairs, scores

    def score(self, source_sents, target_sents, same_language=False):
        src_embeddings, src_idf, src_tokens, _, tgt_embeddings, tgt_idf, tgt_tokens, _ = self._embed(source_sents,
                target_sents, same_language)
        scores = word_mover_score((src_embeddings, src_idf, src_tokens), (tgt_embeddings, tgt_idf, tgt_tokens),
                self.n_gram)
        return scores

class XMoverLMAlign(XMoverAlign):
    """
    Extends XMoverScore based sentence aligner with an additional language model.
    """

    def __init__(self, device, k, n_gram, knn_batch_size, align_batch_size, use_cosine, use_lm, weights):
        super().__init__(device, k, n_gram, knn_batch_size, use_cosine, align_batch_size)
        self.device = device
        self.use_lm = use_lm
        self.weights = weights

    #Override
    def score(self, source_sents, target_sents):
        """
        Compute WMD scores and combine results with perplexity of GPT2 language
        model. This only makes sense when the hyptheses are in English.
        """
        wmd_scores = super().score(source_sents, target_sents)
        if self.use_lm:
            lm_scores = lm_perplexity(target_sents, self.device)
            return (self.weights[0] * array(wmd_scores) + self.weights[1] * array(lm_scores)).tolist()
        else:
            return wmd_scores

class XMoverNMTAlign(XMoverAlign):
    """
    Able to mine data to train an NMT model, which is then combined with the score.
    """

    def __init__(self, device, k, n_gram, knn_batch_size, train_size, align_batch_size, src_lang, tgt_lang,
            mt_model_name, translate_batch_size, ratio, use_cosine, mine_batch_size):
        super().__init__(device, k, n_gram, knn_batch_size, use_cosine, align_batch_size)
        self.train_size = train_size
        self.knn_batch_size = knn_batch_size
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.mt_model_name = mt_model_name
        self.translate_batch_size = translate_batch_size
        self.ratio = ratio
        self.mt_model = None
        self.mt_tokenizer = None
        self.use_cosine = use_cosine
        self.mine_batch_size = mine_batch_size

    #Override
    def score(self, source_sents, target_sents):
        scores = super().score(source_sents, target_sents)
        if self.mt_model is None or self.mt_tokenizer is None:
            return scores
        else:
            mt_scores = super().score(self.translate(source_sents), target_sents, True)
            return [(1 - self.ratio) * score + self.ratio * mt_score for score, mt_score in zip(scores, mt_scores)]

    def train(self, source_sents, target_sents, suffix="data", overwrite=True, k=None):
        file_path, batch, batch_size = join(DATADIR, f"mined-{suffix}.json"), 0, self.mine_batch_size
        pairs, scores = list(), list()
        if not isfile(file_path) or overwrite:
            while batch < len(source_sents):
                logging.info("Obtaining sentence embeddings.")
                batch_src, batch_tgt = source_sents[batch:batch + batch_size], target_sents[batch:batch + batch_size]
                source_sent_embeddings, target_sent_embeddings = self._mean_pool_embed(batch_src, batch_tgt)
                if self.use_cosine:
                    logging.info("Mining pseudo parallel data with Ratio Margin function.")
                    batch_pairs, batch_scores = ratio_margin_align(source_sent_embeddings, target_sent_embeddings,
                            self.k if k is None else k, self.knn_batch_size, self.device)
                else:
                    logging.info("Mining pseudo parallel data using Word Centroid Distance.")
                    candidates, _ = wcd_align(source_sent_embeddings, target_sent_embeddings, self.k if k is None else k,
                            self.knn_batch_size, self.device)
                    logging.info("Computing exact Word Mover's Distances for candidates.")
                    batch_pairs, batch_scores = self._memory_efficient_word_mover_align(batch_src, batch_tgt, candidates)
                del source_sent_embeddings, target_sent_embeddings
                pairs.extend([(src + batch, tgt + batch) for src, tgt in batch_pairs]), scores.extend(batch_scores)
                batch += batch_size
            with open(file_path, "wb") as f:
                idx = 0
                for _, (src, tgt) in sorted(zip(scores, pairs), key=lambda tup: tup[0], reverse=True):
                    src_sent, tgt_sent = source_sents[src], target_sents[tgt]
                    if (
                        edit_distance(src_sent, tgt_sent) / max(len(src_sent), len(tgt_sent)) > 0.5
                        and set(findall("[0-9]+", src_sent)) == set(findall("[0-9]+",tgt_sent))
                    ):
                        line = { "translation": { self.src_lang: src_sent, self.tgt_lang: tgt_sent} }
                        f.write(dumps(line, ensure_ascii=False).encode() + b"\n")
                        idx += 1
                    if idx >= self.train_size:
                        break

        logging.info("Training MT model with pseudo parallel data.")
        self.mt_model, self.mt_tokenizer = train(self.mt_model_name, self.src_lang, self.tgt_lang, file_path,
                overwrite, suffix)
        self.mt_model.to(self.device)

    def translate(self, sentences):
        logging.info("Translating sentences into target language.")
        return translate(self.mt_model, self.mt_tokenizer, sentences, self.translate_batch_size, self.device)

class XMoverNMTLMAlign(XMoverNMTAlign):
    """
    Combine NMT and LM XMoverScore extensions.
    """

    def __init__(self, device, k, n_gram, knn_batch_size, train_size, align_batch_size, src_lang, tgt_lang,
            mt_model_name, translate_batch_size, ratio, use_cosine, mine_batch_size, use_lm, weights):
        super().__init__(device, k, n_gram, knn_batch_size, train_size, align_batch_size, src_lang, tgt_lang,
                mt_model_name, translate_batch_size, ratio, use_cosine, mine_batch_size)
        self.device = device
        self.use_lm = use_lm
        self.weights = weights

    #Override
    def score(self, source_sents, target_sents):
        """
        Compute WMD scores on hypotheses and pseudo translations and combine
        results with perplexity of GPT2 language model. This only makes sense
        when the hyptheses are in English.
        """
        nmt_scores = super().score(source_sents, target_sents)
        if self.use_lm:
            lm_scores = lm_perplexity(target_sents, self.device)
            return (self.weights[0] * array(nmt_scores) + self.weights[1] * array(lm_scores)).tolist()
        else:
            return nmt_scores
