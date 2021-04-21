from os.path import join, dirname, abspath
from transformers import MarianConfig, MarianMTModel, MarianTokenizer, Trainer, TrainingArguments
from tokenizers import SentencePieceBPETokenizer
import torch

# TODO: update transformers to use latest seq2seq training methods
# see: https://github.com/huggingface/transformers/blob/master/examples/seq2seq/run_translation.py
# see: https://github.com/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb
# see: https://huggingface.co/transformers/custom_datasets.html
# see: https://huggingface.co/docs/datasets/loading_datasets.html
datadir = str(abspath(join(dirname(__file__), '../data')))

class Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, pairs):
        self.tokenizer = tokenizer
        self.source = tokenizer([pair[0] for pair in pairs], return_tensors="pt", truncation=True, padding=True)
        with tokenizer.as_target_tokenizer():
            self.target = tokenizer([pair[1] for pair in pairs], return_tensors="pt", truncation=True, padding=True)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.source.items()}
        item['labels'] = self.target[idx]
        return item

    def __len__(self):
        return len(self.source)

def obtain_tokenizer(paths, src_lang, tgt_lang, overwrite=False, cachedir=datadir):
    cachedir = join(cachedir, f"{src_lang}-{tgt_lang}")
    if not overwrite:
        try:
            return MarianTokenizer.from_pretrained(cachedir)
        except EnvironmentError:
            pass
    tokenizer = SentencePieceBPETokenizer()
    tokenizer.train(files=paths, vocab_size=50265, special_tokens=[
        "<s>", "<pad>", "</s>", "<unk>", "<mask>", "<eop>", "<eod>"])
    tokenizer.save_model(cachedir)
    return MarianTokenizer.from_pretrained(cachedir)

def obtain_model(train_data, paths, src_lang, tgt_lang, overwrite=False, cachedir=datadir):
    cachedir = join(cachedir, f"{src_lang}-{tgt_lang}")
    if not overwrite:
        try:
            return MarianMTModel.from_pretrained(cachedir)
        except EnvironmentError:
            pass
    model = MarianMTModel(config=MarianConfig())
    tokenizer = obtain_tokenizer(paths, src_lang, tgt_lang, overwrite, cachedir)
    dataset = Dataset(tokenizer, train_data)

    training_args = TrainingArguments(
        output_dir=cachedir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_gpu_train_batch_size=64,
        save_steps=10000,
        save_total_limit=2,
        prediction_loss_only=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(cachedir)

    return model
