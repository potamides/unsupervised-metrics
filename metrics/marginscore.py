from .common import CommonScore
from .xmoverscore import BertEmbed
from utils.knn import ratio_margin_align
from torch.nn.functional import cosine_similarity
from torch.cuda import is_available as cuda_is_available
from os.path import join, dirname, abspath
from torch import sum as tsum
import logging

class RatioMarginAlign(CommonScore):
    def __init__(self, device, k, knn_batch_size):
        self.device = device
        self.k = k
        self.knn_batch_size = knn_batch_size

    def align(self, source_sents, target_sents):
        src_embeddings, _, _, src_mask, tgt_embeddings, _, _, tgt_mask = self._embed(
                source_sents, target_sents)

        logging.info("Computing scores with Ratio Margin algorithm.")
        source_sent_embeddings = tsum(src_embeddings * src_mask, 1) / tsum(src_mask, 1)
        target_sent_embeddings = tsum(tgt_embeddings * tgt_mask, 1) / tsum(tgt_mask, 1)
        indeces, scores = ratio_margin_align(source_sent_embeddings, target_sent_embeddings, self.k,
                self.knn_batch_size, self.device)

        sent_pairs = [(source_sents[src_idx], target_sents[tgt_idx]) for src_idx, tgt_idx in enumerate(indeces)]
        return sent_pairs, scores

    def score(self, source_sents, target_sents):
        src_embeddings, _, _, src_mask, tgt_embeddings, _, _, tgt_mask = self._embed(source_sents, target_sents)
        source_sent_embeddings = tsum(src_embeddings * src_mask, 1) / tsum(src_mask, 1)
        target_sent_embeddings = tsum(tgt_embeddings * tgt_mask, 1) / tsum(tgt_mask, 1)
        scores = cosine_similarity(source_sent_embeddings, target_sent_embeddings)
        return scores

class RatioMarginBertAlignScore(RatioMarginAlign, BertEmbed):
    def __init__(
        self,
        model_name="bert-base-multilingual-cased",
        mapping="UMD",
        device="cuda" if cuda_is_available() else "cpu",
        datadir = str(abspath(join(dirname(__file__), '../data'))),
        do_lower_case=False,
        alignment = "awesome",
        k = 20,
        remap_size = 2000,
        embed_batch_size = 128,
        knn_batch_size = 1000000
    ):
        RatioMarginAlign.__init__(self, device, k, knn_batch_size)
        BertEmbed.__init__(self, model_name, mapping, device, do_lower_case, remap_size, embed_batch_size,
                alignment, datadir)
