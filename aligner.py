from transformers import BertModel, BertTokenizer, BertConfig
from utils.wmd import word_mover_align
from utils.knn import find_nearest_neighbors
from utils.embed import embed
from utils.remap import word_align, get_aligned_features_avgbpe, clp, umd
from torch.cuda import is_available as cuda_is_available
from random import sample
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
        self.mapping = mapping
        self.device = device
        self.use_knn = use_knn
        self.k = k
        self.n_gram = n_gram
        self.embed_batch_size = embed_batch_size
        self.knn_batch_size = knn_batch_size
        self.projection = None

    def align(self, source_sents, target_sents, return_indeces=False):
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

        return pairs if return_indeces else sent_pairs, scores

    def remap(self, source_sents, target_sents):
        logging.info(f'Computing projection tensor for {"CLP" if self.mapping == "CLP" else "UMD"} remapping method.')

        sent_pairs, scores = self.align(source_sents, target_sents)
        sorted_sent_pairs = list()
        for _, (src_sent, tgt_sent) in sorted(zip(scores, sent_pairs), key=lambda tup: tup[0], reverse=True):
            sorted_sent_pairs.append((src_sent, tgt_sent))

        size = 30000 if self.mapping == "CLP" else 2000
        tokenized_pairs, align_pairs = word_align(sorted_sent_pairs, self.tokenizer, size)
        src_matrix, tgt_matrix = get_aligned_features_avgbpe(tokenized_pairs, align_pairs,
                self.model, self.tokenizer, self.embed_batch_size, self.device)

        logging.info(f"Using {len(src_matrix)} aligned word pairs to compute projection tensor.")
        if self.mapping == "CLP":
            self.projection = clp(src_matrix, tgt_matrix)
        else:
            self.projection = umd(src_matrix, tgt_matrix)

    def precision(self, ref_source_sents, ref_target_sents):
        shuffled_target_sents = sample(ref_target_sents, len(ref_target_sents))
        pairs, _ = self.align(ref_source_sents, shuffled_target_sents, True)

        return sum([ref == shuffled_target_sents[out] for ref, (_, out) in
            zip(ref_target_sents, pairs)]) / len(ref_source_sents)
        
