from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import models, losses
from torch.utils.data import DataLoader
from os.path import join, isfile, basename
from torch.cuda import is_available as cuda_is_available
from torch.nn.functional import cosine_similarity
from torch import from_numpy
from math import ceil
from .utils.knn import ratio_margin_align
from .common import CommonScore
from .utils.dataset import DATADIR
from nltk.metrics.distance import edit_distance
from pathlib import Path
import logging

class CSEScore(CommonScore):
    def __init__(
        self,
        model_name="xlm-roberta-base",
        source_language="en",
        target_language="de",
        device="cuda" if cuda_is_available() else "cpu",
        train_batch_size=128,
        max_seq_length=32,
        num_epochs=1,
        knn_batch_size = 1000000,
        mine_batch_size = 5000000,
        train_size = 100000,
        k = 5,
        suffix = None
    ):
        self.model_name = model_name
        self.train_batch_size = train_batch_size
        self.max_seq_length = max_seq_length
        self.num_epochs = num_epochs
        self.device = device
        self.knn_batch_size = knn_batch_size
        self.mine_batch_size = mine_batch_size
        self.train_size = train_size
        self.k = k
        self.cache_dir = join(DATADIR, "SimCSE",
            f"{'-'.join(sorted([source_language, target_language]))}-{basename(model_name)}")
        self.suffix = suffix
        self.model = self.load_model(model_name)

    def load_model(self, model_name):
        word_embedding_model = models.Transformer(model_name, max_seq_length=self.max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        return SentenceTransformer(modules=[word_embedding_model, pooling_model])

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

    def mine(self, source_sents, target_sents, mine_size, overwrite=True):
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
                check_duplicates_set = set()
                for _, (src, tgt) in sorted(zip(scores, pairs), key=lambda tup: tup[0], reverse=True):
                    src_sent, tgt_sent = source_sents[src], target_sents[tgt]
                    if tgt_sent not in check_duplicates_set and edit_distance(src_sent, tgt_sent) / max(len(src_sent), len(tgt_sent)) > 0.5:
                        check_duplicates_set.add(tgt_sent)
                        f.write(f"{src_sent}\t{tgt_sent}\n".encode())
                        idx += 1
                    if idx >= mine_size:
                        break

        with open(file_path, "rb") as f:
            sents = list()
            for line in f:
                sents.append(line.strip().split("\t"))
            return sents

    def train(self, source_sents, target_sents, mine_size=0, top_percent=0.02, overwrite=True):
        if not isfile(join(self.path, 'config.json')) or overwrite:
            # Train a new model
            new_model = self.load_model(self.model_name)

            # Convert train sentences to sentence pairs
            mono_size = (self.train_size - mine_size) / 2
            mine_mono_size = mine_size / top_percent
            source_train_data = [InputExample(texts=[s, s]) for s in source_sents[:mono_size]]
            target_train_data = [InputExample(texts=[s, s]) for s in target_sents[:mono_size]]
            mined_train_data = [] if mine_size == 0 else [InputExample(texts=[s, t]) for s, t in self.mine(
                    source_sents[mono_size:mine_mono_size+mono_size],
                    target_sents[mono_size:mine_mono_size+mono_size],
                    mine_size,
                    overwrite=overwrite)
                ]
            train_data = source_train_data + target_train_data + mined_train_data

            # DataLoader to batch your data
            train_dataloader = DataLoader(train_data, batch_size=self.train_batch_size, shuffle=True)

            # Use the denoising auto-encoder loss
            train_loss = losses.MultipleNegativesRankingLoss(new_model)

            # Call the fit method
            warmup_steps = ceil(len(train_dataloader) * self.num_epochs * 0.1)  # 10% of train data for warm-up
            new_model.fit(train_objectives=[(train_dataloader, train_loss)],
                epochs=self.num_epochs,
                warmup_steps=warmup_steps,
                optimizer_params={'lr': 5e-5}
            )

        self.model = SentenceTransformer(self.path, device=self.device)
