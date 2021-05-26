from transformers import BertModel, BertTokenizer, BertConfig
from utils.wmd import word_mover_align, word_mover_score
from utils.knn import wcd_align, ratio_margin_align, cosine_align
from utils.embed import bert_embed, vecmap_embed, map_multilingual_embeddings
from utils.remap import word_align, get_aligned_features_avgbpe, clp, umd
from utils.nmt import train, translate
from torch.cuda import is_available as cuda_is_available
from torch.nn.functional import cosine_similarity, mse_loss, l1_loss
from os.path import isfile, join, dirname, abspath
from json import dumps
from math import ceil
from numpy import corrcoef, argsort, arange
from itertools import islice
from nltk.metrics.distance import edit_distance
from abc import ABC, abstractmethod
import logging
import torch

class Common(ABC):
    @abstractmethod
    def _embed():
        pass

    @abstractmethod
    def align():
        pass

    @abstractmethod
    def score():
        pass

    def precision(self, source_sents, ref_sents):
        """
        Computes Precision @ 1 scores.
        """
        pairs, _ = self.align(source_sents, ref_sents)
        return sum([reference == predicted for reference, (_, predicted) in zip(ref_sents, pairs)]) / len(ref_sents)

    def correlation(self, source_sents, system_sents, ref_scores):
        """
        Computes Pearson and Spearman correlation coefficients.
        """
        scores = self.score(source_sents, system_sents)
        ref_ranks, ranks = argsort(ref_scores).argsort(), argsort(scores).argsort()
        return corrcoef(ref_scores, scores)[0,1], corrcoef(ref_ranks, ranks)[0,1]

    def error(self, source_sents, system_sents, ref_scores):
        """
        Computes Mean Absolute Error and Root Mean Squared Error.
        """
        scores = self.score(source_sents, system_sents)
        rmse = mse_loss(torch.FloatTensor(ref_scores), torch.FloatTensor(scores)).sqrt()
        mae = l1_loss(torch.FloatTensor(ref_scores), torch.FloatTensor(scores))
        return rmse, mae

class XMoverAligner(Common):
    def __init__(self, device, k, n_gram, knn_batch_size, use_cosine, align_batch_size):
        self.device = device
        self.k = k
        self.n_gram = n_gram
        self.knn_batch_size = knn_batch_size
        self.use_cosine = use_cosine
        self.align_batch_size = align_batch_size

    def _mean_pool_embed(self, source_sents, target_sents):
        source_sent_embeddings, target_sent_embeddings, idx  = list(), list(), 0
        while idx < max(len(source_sents), len(target_sents)):
            src_embeddings, _, _, src_mask, tgt_embeddings, _, _, tgt_mask = self._embed(
                source_sents[idx:idx + self.align_batch_size], target_sents[idx:idx + self.align_batch_size])
            if len(src_embeddings) > 0:
                source_sent_embeddings.append(torch.sum(src_embeddings * src_mask, 1) / torch.sum(src_mask, 1))
            if len(tgt_embeddings) > 0:
                target_sent_embeddings.append(torch.sum(tgt_embeddings * tgt_mask, 1) / torch.sum(tgt_mask, 1))
            idx += self.align_batch_size

        return source_sent_embeddings, target_sent_embeddings

    def _memory_efficient_word_mover_align(self, source_sents, target_sents, candidates):
        pairs, scores, idx, k = list(), list(), 0, candidates.shape[1]
        while idx < len(source_sents):
            src_embeddings, src_idf, src_tokens, _, tgt_embeddings, tgt_idf, tgt_tokens, _ = self._embed(
                source_sents[idx:idx + self.align_batch_size],
                [target_sents[candidate] for candidate in candidates[idx:idx + self.align_batch_size].flatten()])
            batch_pairs, batch_scores = word_mover_align((src_embeddings, src_idf, src_tokens),
                (tgt_embeddings, tgt_idf, tgt_tokens), self.n_gram,
                arange(len(src_embeddings) * k).reshape(len(src_embeddings), k))
            pairs.extend([(src + idx, candidates[idx:idx + self.align_batch_size].flatten()[tgt]) for src, tgt in batch_pairs])
            scores.extend(batch_scores)
            idx += self.align_batch_size
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

class RatioMarginAligner(Common):
    def __init__(self, device, k, knn_batch_size):
        self.device = device
        self.k = k
        self.knn_batch_size = knn_batch_size

    def align(self, source_sents, target_sents):
        src_embeddings, _, _, src_mask, tgt_embeddings, _, _, tgt_mask = self._embed(
                source_sents, target_sents)

        logging.info("Computing scores with Ratio Margin algorithm.")
        source_sent_embeddings = torch.sum(src_embeddings * src_mask, 1) / torch.sum(src_mask, 1)
        target_sent_embeddings = torch.sum(tgt_embeddings * tgt_mask, 1) / torch.sum(tgt_mask, 1)
        indeces, scores = ratio_margin_align(source_sent_embeddings, target_sent_embeddings, self.k,
                self.knn_batch_size, self.device)

        sent_pairs = [(source_sents[src_idx], target_sents[tgt_idx]) for src_idx, tgt_idx in enumerate(indeces)]
        return sent_pairs, scores

    def score(self, source_sents, target_sents):
        src_embeddings, _, _, src_mask, tgt_embeddings, _, _, tgt_mask = self._embed(source_sents, target_sents)
        source_sent_embeddings = torch.sum(src_embeddings * src_mask, 1) / torch.sum(src_mask, 1)
        target_sent_embeddings = torch.sum(tgt_embeddings * tgt_mask, 1) / torch.sum(tgt_mask, 1)
        scores = cosine_similarity(source_sent_embeddings, target_sent_embeddings)
        return scores

class XMoverNMTAligner(XMoverAligner):
    """
    Extends XMoverScore based sentence aligner with an additional language model.
    """

    def __init__(self, device, k, n_gram, knn_batch_size, datadir, train_size,
        align_batch_size, src_lang, tgt_lang, mt_model_name, translate_batch_size, ratio, use_cosine):
        super().__init__(device, k, n_gram, knn_batch_size, use_cosine, align_batch_size)
        self.datadir = datadir
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

    #Override
    def score(self, source_sents, target_sents):
        scores = super().score(source_sents, target_sents)
        if self.mt_model is None or self.mt_tokenizer is None:
            return scores
        else:
            mt_scores = super().score(self.translate(source_sents), target_sents, True)
            return [(1 - self.ratio) * score + self.ratio * mt_score for score, mt_score in zip(scores, mt_scores)]

    def train(self, source_sents, target_sents, overwrite=True, k=1):
        file_path, pairs, scores = join(self.datadir, f"{self.src_lang}-{self.tgt_lang}-mined.json"), list(), list()
        if not isfile(file_path) or overwrite:
            logging.info("Obtaining sentence embeddings.")
            source_sent_embeddings, target_sent_embeddings = self._mean_pool_embed(source_sents, target_sents)
            pairs, scores = list(), list()
            if self.use_cosine:
                logging.info("Mining pseudo parallel data with Ratio Margin function.")
                pairs, scores = ratio_margin_align(torch.cat(source_sent_embeddings), torch.cat(target_sent_embeddings),
                        self.k, self.knn_batch_size, self.device)
            else:
                logging.info("Mining pseudo parallel data using Word Centroid Distance.")
                candidates = wcd_align(torch.cat(source_sent_embeddings), torch.cat(target_sent_embeddings), k,
                    self.knn_batch_size, self.device)[0]
                logging.info("Computing exact Word Mover's Distances for candidates.")
                pairs, scores = self._memory_efficient_word_mover_align(source_sents, target_sents, candidates)
        with open(file_path, "wb") as f:
            idx = 0
            for _, (src, tgt) in sorted(zip(scores, pairs), key=lambda tup: tup[0], reverse=True):
                src_sent, tgt_sent = source_sents[src], target_sents[tgt]
                if edit_distance(src_sent, tgt_sent) / max(len(src_sent), len(tgt_sent)) > 0.5:
                    line = { "translation": { self.src_lang: src_sent, self.tgt_lang: tgt_sent} }
                    f.write(dumps(line, ensure_ascii=False).encode() + b"\n")
                    idx += 1
                if idx >= self.train_size:
                    break
        logging.info("Training MT model with pseudo parallel data.")
        self.mt_model, self.mt_tokenizer = train(self.mt_model_name, self.src_lang, self.tgt_lang, file_path,
                overwrite, self.datadir)
        self.mt_model.to(self.device)

    def translate(self, sentences):
        logging.info("Translating sentences into target language.")
        return translate(self.mt_model, self.mt_tokenizer, sentences, self.translate_batch_size, self.device)

class BertEmbedder(Common):
    def __init__(self, model_name, mapping, device, do_lower_case, remap_size, embed_batch_size):
        config = BertConfig.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
        self.model = BertModel.from_pretrained(model_name, config=config)
        self.model.to(device)
        self.device = device
        self.mapping = mapping
        self.remap_size = remap_size
        self.embed_batch_size = embed_batch_size
        self.projection = None

    def _embed(self, source_sents, target_sents, same_language=False):
        logging.info("Embedding source sentences with mBERT.")
        src_embeddings, src_idf, src_tokens, src_mask = bert_embed(source_sents, self.embed_batch_size, self.model,
                self.tokenizer, self.device)
        logging.info("Embedding target sentences with mBERT.")
        tgt_embeddings, tgt_idf, tgt_tokens, tgt_mask = bert_embed(target_sents, self.embed_batch_size, self.model,
                self.tokenizer, self.device)
        
        if self.projection is not None and not same_language:
            if self.mapping == 'CLP':
                logging.info("Remap cross-lingual alignments with CLP")
                src_embeddings = torch.matmul(src_embeddings, self.projection)
            else:
                logging.info("Remap cross-lingual alignments with UMD")
                src_embeddings = src_embeddings - (src_embeddings * self.projection).sum(2, keepdim=True) * \
                        self.projection.repeat(src_embeddings.shape[0], src_embeddings.shape[1], 1)        

        return src_embeddings, src_idf, src_tokens, src_mask, tgt_embeddings, tgt_idf, tgt_tokens, tgt_mask

    def remap(self, source_sents, target_sents):
        logging.info(f'Computing projection tensor for {"CLP" if self.mapping == "CLP" else "UMD"} remapping method.')

        sent_pairs, scores = self.align(source_sents, target_sents)
        sorted_sent_pairs = list()
        for _, (src_sent, tgt_sent) in sorted(zip(scores, sent_pairs), key=lambda tup: tup[0], reverse=True):
            if edit_distance(src_sent, tgt_sent) / max(len(src_sent), len(tgt_sent)) > 0.5:
                sorted_sent_pairs.append((src_sent, tgt_sent))

        tokenized_pairs, align_pairs = word_align(sorted_sent_pairs, self.tokenizer, self.remap_size)
        src_matrix, tgt_matrix = get_aligned_features_avgbpe(tokenized_pairs, align_pairs,
                self.model, self.tokenizer, self.embed_batch_size, self.device)

        logging.info(f"Using {len(src_matrix)} aligned word pairs to compute projection tensor.")
        if self.mapping == "CLP":
            self.projection = clp(src_matrix, tgt_matrix)
        else:
            self.projection = umd(src_matrix, tgt_matrix)

class VecMapEmbedder(Common):
    def __init__(self, device, src_lang, tgt_lang, batch_size):
        self.device = device
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.batch_size = batch_size
        self.src_dict = None
        self.tgt_dict = None

    def _embed(self, source_sents, target_sents, same_language=False):
        if self.src_dict is None or self.tgt_dict is None:
            logging.info("Obtaining cross-lingual word embedding mappings from fasttext embeddings.")
            self.src_dict, self.tgt_dict = map_multilingual_embeddings(self.src_lang, self.tgt_lang,
                self.batch_size, self.device)
        src_embeddings, src_idf, src_tokens, src_mask = vecmap_embed(source_sents,
                *((self.tgt_dict, self.tgt_lang) if same_language else (self.src_dict, self.src_lang)))
        tgt_embeddings, tgt_idf, tgt_tokens, tgt_mask = vecmap_embed(target_sents, self.tgt_dict, self.tgt_lang)

        return src_embeddings, src_idf, src_tokens, src_mask, tgt_embeddings, tgt_idf, tgt_tokens, tgt_mask

class XMoverBertAligner(XMoverAligner, BertEmbedder):
    def __init__(
        self,
        model_name="bert-base-multilingual-cased",
        mapping="UMD",
        device="cuda" if cuda_is_available() else "cpu",
        do_lower_case=False,
        use_cosine = False,
        k = 20,
        n_gram = 1,
        remap_size = 3000,
        embed_batch_size = 128,
        knn_batch_size = 1000000,
        align_batch_size = 5000
    ):
        logging.info("Using device \"%s\" for computations.", device)
        XMoverAligner.__init__(self, device, k, n_gram, knn_batch_size, use_cosine, align_batch_size)
        BertEmbedder.__init__(self, model_name, mapping, device, do_lower_case, remap_size, embed_batch_size)

class RatioMarginBertAligner(RatioMarginAligner, BertEmbedder):
    def __init__(
        self,
        model_name="bert-base-multilingual-cased",
        mapping="UMD",
        device="cuda" if cuda_is_available() else "cpu",
        do_lower_case=False,
        k = 20,
        remap_size = 3000,
        embed_batch_size = 128,
        knn_batch_size = 1000000
    ):
        RatioMarginAligner.__init__(self, device, k, knn_batch_size)
        BertEmbedder.__init__(self, model_name, mapping, device, do_lower_case, remap_size, embed_batch_size)

class XMoverVecMapAligner(XMoverAligner, VecMapEmbedder):
    def __init__(
        self,
        device="cuda" if cuda_is_available() else "cpu",
        use_cosine = False,
        k = 20,
        n_gram = 1,
        knn_batch_size = 1000000,
        src_lang = "de",
        tgt_lang = "en",
        batch_size = 5000,
        align_batch_size = 5000
    ):
        logging.info("Using device \"%s\" for computations.", device)
        XMoverAligner.__init__(self, device, k, n_gram, knn_batch_size, use_cosine, align_batch_size)
        VecMapEmbedder.__init__(self, device, src_lang, tgt_lang, batch_size)

class XMoverNMTBertAligner(XMoverNMTAligner, BertEmbedder):
    def __init__(
        self,
        device="cuda" if cuda_is_available() else "cpu",
        use_cosine = False,
        k = 20,
        n_gram = 1,
        knn_batch_size = 1000000,
        datadir = str(abspath(join(dirname(__file__), 'data'))),
        train_size = 500000,
        align_batch_size = 5000,
        src_lang = "de",
        tgt_lang = "en",
        model_name="bert-base-multilingual-cased",
        mt_model_name="facebook/mbart-large-cc25",
        mapping="UMD",
        do_lower_case=False,
        remap_size = 3000,
        embed_batch_size = 128,
        translate_batch_size = 16,
        ratio = 0.5
    ):
        logging.info("Using device \"%s\" for computations.", device)
        XMoverNMTAligner.__init__(self, device, k, n_gram, knn_batch_size, datadir,
            train_size, align_batch_size, src_lang, tgt_lang, mt_model_name, translate_batch_size, ratio, use_cosine)
        BertEmbedder.__init__(self, model_name, mapping, device, do_lower_case, remap_size, embed_batch_size)
