from .utils.embed import vecmap_embed, map_multilingual_embeddings
from .utils.knn import ratio_margin_align
from torch.nn.functional import cosine_similarity
from .common import CommonScore
from torch.cuda import is_available as cuda_is_available
import logging, torch

class VecMapScore(CommonScore):
    def __init__(
        self,
        device="cuda" if cuda_is_available() else "cpu",
        src_lang="en",
        tgt_lang="de",
        batch_size=5000,
        knn_batch_size = 1000000,
        k = 5
    ):
        self.device = device
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.batch_size = batch_size
        self.knn_batch_size = knn_batch_size
        self.k = k
        self.src_dict = None
        self.tgt_dict = None

    def _embed(self, source_sents, target_sents):
        if self.src_dict is None or self.tgt_dict is None:
            logging.info("Obtaining cross-lingual word embedding mappings from fasttext embeddings.")
            self.src_dict, self.tgt_dict = map_multilingual_embeddings(self.src_lang, self.tgt_lang,
                self.batch_size, self.device)

        src_embeddings, *_, src_mask = vecmap_embed(source_sents, self.src_dict, self.src_lang)
        tgt_embeddings, *_, tgt_mask = vecmap_embed(target_sents, self.tgt_dict, self.tgt_lang)
        source_sent_embeddings = torch.sum(src_embeddings * src_mask, 1) / torch.sum(src_mask, 1)
        target_sent_embeddings = torch.sum(tgt_embeddings * tgt_mask, 1) / torch.sum(tgt_mask, 1)

        return source_sent_embeddings, target_sent_embeddings

    def align(self, source_sents, target_sents):
        source_embeddings, target_embeddings = self._embed(source_sents, target_sents)
        indeces, scores = ratio_margin_align(source_embeddings, target_embeddings, self.k,
                self.knn_batch_size, self.device)

        sent_pairs = [(source_sents[src_idx], target_sents[tgt_idx]) for src_idx, tgt_idx in indeces]
        return sent_pairs, scores

    def score(self, source_sents, target_sents):
        source_embeddings, target_embeddings = self._embed(source_sents, target_sents)
        return cosine_similarity(source_embeddings, target_embeddings)
