from transformers import BertModel, BertTokenizer, BertConfig
from utils.wmd import word_mover_align
from utils.knn import find_nearest_neighbors
from utils.embed import embed
from torch.cuda import is_available as cuda_is_available
from random import sample
import logging
import torch

class XMoverAligner:

    def __init__(
        self,
        model_name='bert-base-multilingual-cased',
        do_lower_case=False,
        device="cuda" if cuda_is_available() else "cpu",
        k = 20,
        n_gram = 1,
        embed_batch_size = 128,
        knn_batch_size = 1000000,
        use_knn = True
    ):
        logging.info("Using device \"%s\" for computations.", device)
        config = BertConfig.from_pretrained(model_name)

        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
        self.model = BertModel.from_pretrained(model_name, config=config)
        self.model.to(device)
        self.device = device
        self.k = k
        self.n_gram = n_gram
        self.embed_batch_size = embed_batch_size
        self.knn_batch_size = knn_batch_size
        self.use_knn = use_knn

    def align_sents(self, source_sents, target_sents):
        logging.info("Embedding source sentences with mBERT.")
        source_data = embed(source_sents, self.embed_batch_size, self.model, self.tokenizer, self.device)
        logging.info("Embedding target sentences with mBERT.")
        target_data = embed(target_sents, self.embed_batch_size, self.model, self.tokenizer, self.device)
        candidates = None
        if self.use_knn:
            logging.info("Finding nearest neighbors with KNN algorithm.")
            source_sent_embeddings = torch.sum(source_data[0] * source_data[3], 1) / torch.sum(source_data[3], 1)
            target_sent_embeddings = torch.sum(target_data[0] * target_data[3], 1) / torch.sum(target_data[3], 1)
            candidates = find_nearest_neighbors(source_sent_embeddings, target_sent_embeddings, self.k,
                    self.knn_batch_size, self.device)
        logging.info("Computing word mover scores.")
        return word_mover_align(source_data[:3], target_data[:3], self.n_gram, self.device, candidates) 

    def accuracy_on_sents(self, ref_source_sents, ref_target_sents):
        shuffled_target_sents = sample(ref_target_sents, len(ref_target_sents))
        pairs = self.align_sents(ref_source_sents, shuffled_target_sents)

        return sum([ref == shuffled_target_sents[out] for ref, (_, out) in zip(ref_target_sents, pairs)]) / len(ref_source_sents)
        
