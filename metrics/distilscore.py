from sentence_transformers import SentenceTransformer, models, losses
from sentence_transformers.evaluation import TranslationEvaluator, SequentialEvaluator
from sentence_transformers.datasets import ParallelSentencesDataset
from torch.utils.data import DataLoader
from torch.cuda import is_available as cuda_is_available
from torch.nn.functional import cosine_similarity
from torch import from_numpy
from .common import CommonScore
from .utils.knn import ratio_margin_align
from .utils.dataset import DATADIR, LangDetect
from .utils.nmt import language2mBART
from os.path import join, isfile, basename
from nltk.metrics.distance import edit_distance
from pathlib import Path
from math import ceil

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
        student_is_pretrained=False,
        train_batch_size=64,                # Batch size for training
        inference_batch_size=64,            # Batch size at inference
        num_epochs=10,                      # Train for x epochs
        knn_batch_size = 1000000,
        mine_batch_size = 5000000,
        train_size = 200000,
        k = 5,
        suffix = None
    ):
        assert "en" in [source_language, target_language], "One language has to be English!"
        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name
        self.target_language = target_language
        self.train_batch_size = train_batch_size
        self.inference_batch_size = inference_batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.knn_batch_size = knn_batch_size
        self.mine_batch_size = mine_batch_size
        self.train_size = train_size
        self.k = k
        self.cache_dir = join(DATADIR, "distillation",
            f"{'-'.join(sorted([source_language, target_language]))}-{basename(teacher_model_name)}-{basename(student_model_name)}")
        self.suffix = suffix
        if student_is_pretrained:
            self.model = SentenceTransformer(student_model_name, device=self.device)
        else:
            self.model = self.load_student(student_model_name)

    def load_student(self, model_name):
        logging.info("Creating model from scratch")
        word_embedding_model = models.Transformer(model_name)
        # Apply mean pooling to get one fixed sized sentence vector
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=self.device)

        # mBART also has a decoder but we are only interested in the encoder output. To make sure that
        # sentence_transformers use the encoder output we monkey patch the forward method. Don't do this at home kids.
        if "mbart" in model_name:
            mbart, detector = word_embedding_model.auto_model, LangDetect()
            mbart.forward = lambda **kv: type(mbart).forward(mbart, **kv)[-1:]
            
            def tokenize(text):
                model.tokenizer.src_lang = language2mBART[detector.detect(text)]
                return word_embedding_model.tokenize(text)

            self.model.tokenize = tokenize

        return model

    @property
    def path(self):
        path = self.cache_dir + f"-{self.suffix}" if self.suffix is not None else ""
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
        return cosine_similarity(from_numpy(source_embeddings), from_numpy(target_embeddings))

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
                batch_pairs, batch_scores = ratio_margin_align(from_numpy(source_embeddings),
                        from_numpy(target_embeddings), self.k, self.knn_batch_size, self.device)
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
            # Train a new model to avoid overfitting
            new_model = self.load_student(self.student_model_name)
            logging.info("Loading teacher model and training data.")
            teacher_model = SentenceTransformer(self.teacher_model_name, device=self.device)
            train_data = ParallelSentencesDataset(student_model=new_model, teacher_model=teacher_model,
                    batch_size=self.inference_batch_size, use_embedding_cache=True)

            if self.target_language == "en": # since teacher embeds source sentences make sure they are in english
                source_sents, target_sents = target_sents, source_sents

            if unaligned:
                train_data.load_data(self.mine(source_sents, target_sents, overwrite=overwrite),
                        max_sentences=self.train_size, max_sentence_length=None)
            else:
                train_data.add_dataset(zip(source_sents, target_sents), max_sentences=self.train_size,
                        max_sentence_length=None)

            train_dataloader = DataLoader(train_data, shuffle=True, batch_size=self.train_batch_size)
            train_loss = losses.MSELoss(model=new_model)

            dev_trans_acc = None
            if dev_source_sents is not None and dev_target_sents is not None:
                # TranslationEvaluator computes the embeddings for all parallel sentences. It then checks if the
                # embedding of source[i] is the closest to target[i] out of all available target sentences
                dev_trans_acc = TranslationEvaluator(dev_source_sents, dev_target_sents, write_csv=False,
                        batch_size=self.inference_batch_size)

            # Train the model
            logging.info("Fine-tuning student model.")
            warmup_steps = ceil(len(train_dataloader) * self.num_epochs * 0.1)  # 10% of train data for warm-up
            new_model.fit(train_objectives=[(train_dataloader, train_loss)],
                evaluator=None if dev_trans_acc is None else SequentialEvaluator([dev_trans_acc], main_score_function=np.mean),
                epochs=self.num_epochs,
                warmup_steps=warmup_steps,
                optimizer_params= {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False}
            )
            new_model.save(self.path)

        self.model = SentenceTransformer(self.path, device=self.device)
