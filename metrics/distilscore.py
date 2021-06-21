from sentence_transformers import SentenceTransformer, models, losses
from sentence_transformers.evaluation import TranslationEvaluator, SequentialEvaluator
from sentence_transformers.datasets import ParallelSentencesDataset
from torch.utils.data import DataLoader
from torch.cuda import is_available as cuda_is_available
from torch.nn.functional import cosine_similarity
from .common import CommonScore
from .utils.knn import ratio_margin_align
from .utils.dataset import DATADIR
from os.path import join, isfile
from nltk.metrics.distance import edit_distance
from pathlib import Path

import logging
import numpy as np

class DistilScore(CommonScore):
    def __init__(
        self,
        teacher_model_name="bert-base-nli-stsb-mean-tokens",
        student_model_name="xlm-roberta-base",
        source_language="en",
        target_language="de",
        device="cuda" if cuda_is_available() else "cpu",
        max_seq_length=128,                 # Student model max. lengths for inputs (number of word pieces)
        train_batch_size=64,                # Batch size for training
        inference_batch_size=64,            # Batch size at inference
        max_sentences_per_trainfile=200000, # Maximum number of  parallel sentences for training
        train_max_sentence_length=250,      # Maximum length (characters) for parallel training sentences
        num_epochs=5,                       # Train for x epochs
        num_warmup_steps=10000,             # Warumup steps
        num_evaluation_steps=1000,          # Evaluate performance after every xxxx steps
        knn_batch_size = 1000000,
        mine_batch_size = 5000000,
        train_size = 500000,
        k = 5,
        suffix = None
    ):
        self.teacher_model_name = teacher_model_name
        self.train_batch_size = train_batch_size
        self.inference_batch_size = inference_batch_size
        self.max_sentences_per_trainfile = max_sentences_per_trainfile
        self.train_max_sentence_length = train_max_sentence_length
        self.num_epochs = num_epochs
        self.num_warmup_steps = num_warmup_steps
        self.num_evaluation_steps = num_evaluation_steps
        self.device = device
        self.knn_batch_size = knn_batch_size
        self.mine_batch_size = mine_batch_size
        self.train_size = train_size
        self.k = k
        self.cache_dir = join(DATADIR, "distillation",
                f"{source_language}-{target_language}-{teacher_model_name}-{student_model_name}")
        self.suffix = suffix

        logging.info("Creating model from scratch")
        word_embedding_model = models.Transformer(student_model_name, max_seq_length=max_seq_length)
        # Apply mean pooling to get one fixed sized sentence vector
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

    @property
    def path(self):
        path = self.cache_folder + f"-{self.suffix}" if self.suffix is not None else ""
        Path(path).mkdir(parents=True, exist_ok=True)
        return path

    def _embed(self, source_sents, target_sents):
        return self.model.encode(source_sents), self.model.encode(target_sents)

    def align(self, source_sents, target_sents):
        source_embeddings, target_embeddings = self._embed(source_sents, target_sents)
        indeces, scores = ratio_margin_align(source_embeddings, target_embeddings, self.k,
                self.knn_batch_size, self.device)

        sent_pairs = [(source_sents[src_idx], target_sents[tgt_idx]) for src_idx, tgt_idx in indeces]
        return sent_pairs, scores

    def score(self, source_sents, target_sents):
        source_embeddings, target_embeddings = self._embed(source_sents, target_sents)
        return cosine_similarity(source_embeddings, target_embeddings)

    def mine(self, source_sents, target_sents, overwrite=True):
        logging.info("Mining pseudo parallel data.")
        file_path = join(self.path, "mined-sentence-pairs.txt")
        pairs, scores, batch, batch_size = list(), list(), 0, self.mine_batch_size
        if not isfile(file_path) or overwrite:
            while batch < len(source_sents):
                logging.info("Obtaining sentence embeddings.")
                batch_src, batch_tgt = source_sents[batch:batch + batch_size], target_sents[batch:batch + batch_size]
                source_embeddings, target_embeddings = self._embed(batch_src, batch_tgt)
                logging.info("Mining pseudo parallel data with Ratio Margin function.")
                batch_pairs, batch_scores = ratio_margin_align(source_embeddings, target_embeddings, self.k,
                        self.knn_batch_size, self.device)
                del source_embeddings, target_embeddings
                pairs.extend([(src + batch, tgt + batch) for src, tgt in batch_pairs]), scores.extend(batch_scores)
                batch += batch_size
            with open(file_path, "wb") as f:
                idx = 0
                for _, (src, tgt) in sorted(zip(scores, pairs), key=lambda tup: tup[0], reverse=True):
                    src_sent, tgt_sent = source_sents[src], target_sents[tgt]
                    if edit_distance(src_sent, tgt_sent) / max(len(src_sent), len(tgt_sent)) > 0.5:
                        f.write(f"{src_sent}\t{tgt_sent}\n".encode())
                        idx += 1
                    if idx >= self.train_size:
                        break
        return file_path

    def train(self, source_sents, target_sents, dev_source_sents=None, dev_target_sents=None, unaligned=True, overwrite=True):
        if not isfile(join(self.path, 'config.json')) or overwrite:
            logging.info("Loading teacher model and training data.")
            teacher_model = SentenceTransformer(self.teacher_model_name, device=self.device)

            train_data = ParallelSentencesDataset(student_model=self.model, teacher_model=teacher_model,
                    batch_size=self.inference_batch_size, use_embedding_cache=True)
            if unaligned:
                train_data.load_data(self.mine(source_sents, target_sents, overwrite=overwrite),
                        max_sentences=self.max_sentences_per_trainfile, max_sentence_length=self.train_max_sentence_length)
            else:
                train_data.add_dataset(zip(source_sents, target_sents), max_sentences=self.max_sentences_per_trainfile,
                        max_sentence_length=self.train_max_sentence_length)

            train_dataloader = DataLoader(train_data, shuffle=True, batch_size=self.train_batch_size)
            train_loss = losses.MSELoss(model=self.model)

            dev_trans_acc = None
            if dev_source_sents is not None and dev_target_sents is not None:
                # TranslationEvaluator computes the embeddings for all parallel sentences. It then checks if the
                # embedding of source[i] is the closest to target[i] out of all available target sentences
                dev_trans_acc = TranslationEvaluator(dev_source_sents, dev_target_sents, write_csv=False,
                        batch_size=self.inference_batch_size)

            # Train the model
            logging.info("Fine-tuning student model.")
            self.model.fit(train_objectives=[(train_dataloader, train_loss)],
                evaluator=None if dev_trans_acc is None else SequentialEvaluator([dev_trans_acc], main_score_function=np.mean),
                epochs=self.num_epochs,
                warmup_steps=self.num_warmup_steps,
                evaluation_steps=self.num_evaluation_steps,
                optimizer_params= {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False}
            )
            self.model.save(self.path)
        else:
            self.model = SentenceTransformer(self.path, device=self.device)
