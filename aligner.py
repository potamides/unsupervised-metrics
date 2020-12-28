from transformers import BertModel, BertTokenizer, BertConfig
from align_utils import word_mover_align
class XMoverAligner:

    def __init__(
        self,
        model_name=None,
        do_lower_case=False,
        device='cuda:0',
        n_gram = 2,
        batch_size = 256
    ):
        config = BertConfig.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)

        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
        self.model = BertModel.from_pretrained(model_name, config=config)
        self.model.to(device)

    def align_data(self, source_data, target_data):
        return word_mover_align(self.model, self.tokenizer, source_data, target_data,
                self.n_gram, self.batch_size, self.device)
