from transformers import BertModel, BertTokenizer, BertConfig
from utils.wmd import word_mover_align, word_mover_score
from utils.knn import find_nearest_neighbors
from utils.embed import embed
from utils.remap import word_align, get_aligned_features_avgbpe, clp, umd
from torch.cuda import is_available as cuda_is_available
from numpy import corrcoef
import logging
import torch

class XMoverAligner:
    def __init__(
        self,
        model_name="bert-base-multilingual-cased",
        mapping="CLP",
        device="cuda" if cuda_is_available() else "cpu",
        do_lower_case=False,
        use_knn = True,
        k = 20,
        n_gram = 1,
        embed_batch_size = 128,
        knn_batch_size = 1000000,
    ):
        logging.info("Using device \"%s\" for computations.", device)
        config = BertConfig.from_pretrained(model_name)

        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
        self.model = BertModel.from_pretrained(model_name, config=config)
        self.model.to(device)
        self.device = device
        self.mapping = mapping
        self.remap_size = 10000 if self.mapping == "CLP" else 2000
        self.use_knn = use_knn
        self.k = k
        self.n_gram = n_gram
        self.embed_batch_size = embed_batch_size
        self.knn_batch_size = knn_batch_size
        self.projection = None

    def _embed(self, source_sents, target_sents):
        logging.info("Embedding source sentences with mBERT.")
        src_embeddings, src_idf, src_tokens, src_mask = embed(source_sents, self.embed_batch_size, self.model,
                self.tokenizer, self.device)
        logging.info("Embedding target sentences with mBERT.")
        tgt_embeddings, tgt_idf, tgt_tokens, tgt_mask = embed(target_sents, self.embed_batch_size, self.model,
                self.tokenizer, self.device)
        
        if self.projection is not None:
            if self.mapping == 'CLP':
                logging.info("Remap cross-lingual alignments with CLP")
                src_embeddings = torch.matmul(src_embeddings, self.projection)
            else:
                logging.info("Remap cross-lingual alignments with UMD")
                src_embeddings = src_embeddings - (src_embeddings * self.projection).sum(2, keepdim=True) * \
                        self.projection.repeat(src_embeddings.shape[0], src_embeddings.shape[1], 1)        

        return src_embeddings, src_idf, src_tokens, src_mask, tgt_embeddings, tgt_idf, tgt_tokens, tgt_mask


    def align(self, source_sents, target_sents):
        src_embeddings, src_idf, src_tokens, src_mask, tgt_embeddings, tgt_idf, tgt_tokens, tgt_mask = self._embed(
                source_sents, target_sents)

        candidates = None
        if self.use_knn:
            logging.info("Finding nearest neighbors with KNN algorithm.")
            source_sent_embeddings = torch.sum(src_embeddings * src_mask, 1) / torch.sum(src_mask, 1)
            target_sent_embeddings = torch.sum(tgt_embeddings * tgt_mask, 1) / torch.sum(tgt_mask, 1)
            candidates = find_nearest_neighbors(source_sent_embeddings, target_sent_embeddings, self.k,
                    self.knn_batch_size, self.device)

        logging.info("Computing word mover scores.")
        pairs, scores = word_mover_align((src_embeddings, src_idf, src_tokens), (tgt_embeddings, tgt_idf, tgt_tokens),
                self.n_gram, candidates)
        sent_pairs = [(source_sents[src_idx], target_sents[tgt_idx]) for src_idx, tgt_idx in pairs]
        return sent_pairs, scores

    def score(self, source_sents, target_sents):
        src_embeddings, src_idf, src_tokens, _, tgt_embeddings, tgt_idf, tgt_tokens, _ = self._embed( source_sents,
                target_sents)
        scores = word_mover_score((src_embeddings, src_idf, src_tokens), (tgt_embeddings, tgt_idf, tgt_tokens),
                self.n_gram)
        return scores

    def remap(self, source_sents, target_sents):
        logging.info(f'Computing projection tensor for {"CLP" if self.mapping == "CLP" else "UMD"} remapping method.')

        sent_pairs, scores = self.align(source_sents, target_sents)
        sorted_sent_pairs = list()
        for _, (src_sent, tgt_sent) in sorted(zip(scores, sent_pairs), key=lambda tup: tup[0], reverse=True):
            sorted_sent_pairs.append((src_sent, tgt_sent))

        tokenized_pairs, align_pairs = word_align(sorted_sent_pairs, self.tokenizer, self.remap_size)
        src_matrix, tgt_matrix = get_aligned_features_avgbpe(tokenized_pairs, align_pairs,
                self.model, self.tokenizer, self.embed_batch_size, self.device)

        logging.info(f"Using {len(src_matrix)} aligned word pairs to compute projection tensor.")
        if self.mapping == "CLP":
            self.projection = clp(src_matrix, tgt_matrix)
        else:
            self.projection = umd(src_matrix, tgt_matrix)

    def precision(self, source_sents, ref_sents):
        pairs, _ = self.align(source_sents, ref_sents)
        return sum([reference == predicted for reference, (_, predicted) in zip(ref_sents, pairs)]) / len(ref_sents)

    def correlation(self, source_sents, system_sents, ref_scores):
        scores = self.score(source_sents, system_sents)
        return corrcoef(ref_scores, scores)[0,1]
