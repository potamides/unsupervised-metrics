from transformers import BertModel, BertTokenizer, BertConfig
from word_mover_utils import word_mover_align
from knn_utils import find_nearest_neighbors
from torch.cuda import is_available as cuda_is_available
from random import sample
import logging

class XMoverAligner:

    def __init__(
        self,
        model_name='bert-base-multilingual-cased',
        do_lower_case=False,
        device="cuda" if cuda_is_available() else "cpu",
        k = 5,
        n_gram = 1,
        word_mover_batch_size = 128,
        nearest_neighbor_batch_size = 1000000,
        use_knn = True
    ):
        logging.info("Using device \"%s\" for computations.", device)
        config = BertConfig.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)

        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
        self.model = BertModel.from_pretrained(model_name, config=config)
        self.model.to(device)
        self.device = device
        self.k = k
        self.n_gram = n_gram
        self.word_mover_batch_size = word_mover_batch_size
        self.nearest_neighbor_batch_size = nearest_neighbor_batch_size
        self.use_knn = use_knn

    def align_data(self, source_data, target_data, source_lang, target_lang):
        candidates = find_nearest_neighbors(source_data, target_data, source_lang, target_lang,
                self.k, self.nearest_neighbor_batch_size, self.device) if self.use_knn else None
        return word_mover_align(self.model, self.tokenizer, source_data, target_data,
                self.n_gram, self.word_mover_batch_size, self.device, candidates)

    def accuracy_on_data(self, ref_source_data, ref_target_data, source_lang, target_lang):
        pairs = self.align_data(ref_source_data, sample(ref_target_data,
            len(ref_target_data)), source_lang, target_lang)

        return sum([ref == out for ref, (_, out) in zip(ref_target_data,
            pairs)]) / len(ref_source_data)
        
