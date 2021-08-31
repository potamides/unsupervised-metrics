from sentence_transformers import SentenceTransformer, InputExample, models, util
from torch.utils.data import DataLoader
from os.path import join, isfile, basename
from torch.cuda import device_count, is_available as cuda_is_available
from torch.nn import CrossEntropyLoss, Module, DataParallel
from torch.nn.functional import cosine_similarity
from math import ceil
from .utils.knn import ratio_margin_align
from .common import CommonScore
from .utils.dataset import DATADIR
from .utils.wmd import word_mover_score
from nltk.metrics.distance import edit_distance
from pathlib import Path
import logging
import torch

class AdditiveMarginSoftmaxLoss(Module):
    """
    Contrastive learning loss function used by LaBSE and SimCSE.
    """
    def __init__(self, model, scale = 20.0, margin = 0.0, symmetric = True, similarity_fct = util.cos_sim):
        super().__init__()
        self.model = model
        self.scale = scale
        self.margin = margin
        self.symmetric = symmetric
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = CrossEntropyLoss()

    def additive_margin_softmax_loss(self, embeddings_a, embeddings_b):
        scores = self.similarity_fct(embeddings_a, embeddings_b)
        scores.diagonal().subtract_(self.margin)
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
        return self.cross_entropy_loss(self.scale * scores, labels)

    def forward(self, sentence_features, _):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        assert len(reps) == 2, "Inputs should be source texts and translations"
        embeddings_a = reps[0]
        embeddings_b = reps[1]

        if self.symmetric:
            return self.additive_margin_softmax_loss(embeddings_a, embeddings_b) + self.additive_margin_softmax_loss(embeddings_b, embeddings_a)
        else:
            return self.additive_margin_softmax_loss(embeddings_a, embeddings_b)

    def get_config_dict(self):
        return {'scale': self.scale, 'margin': self.margin, 'symmetric': self.symmetric, 'similarity_fct': self.similarity_fct.__name__}

class ContrastScore(CommonScore):
    def __init__(
        self,
        model_name="xlm-roberta-base",
        source_language="en",
        target_language="de",
        device="cuda" if cuda_is_available() else "cpu",
        parallelize= False,
        train_batch_size=256,
        max_seq_length=None,
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
        self.parallelize = parallelize
        self.knn_batch_size = knn_batch_size
        self.mine_batch_size = mine_batch_size
        self.train_size = train_size
        self.k = k
        self.cache_dir = join(DATADIR, "contrastive-learning",
            f"{'-'.join(sorted([source_language, target_language]))}-{basename(model_name)}")
        self.suffix = suffix
        self.model = self.load_model(model_name)

    def load_model(self, model_name):
        word_embedding_model = models.Transformer(model_name, max_seq_length=self.max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        return SentenceTransformer(modules=[word_embedding_model, pooling_model], device=self.device)

    @property
    def path(self):
        path = self.cache_dir + f"-{self.suffix}" if self.suffix is not None else ""
        Path(path).mkdir(parents=True, exist_ok=True)
        return path

    def _embed(self, source_sents, target_sents):
        return (
            self.model.encode(source_sents, convert_to_tensor=True).cpu(),
            self.model.encode(target_sents, convert_to_tensor=True).cpu())

    def align(self, source_sents, target_sents):
        source_embeddings, target_embeddings = self._embed(source_sents, target_sents)
        indeces, scores = ratio_margin_align(source_embeddings, target_embeddings, self.k,
                self.knn_batch_size, self.device)

        sent_pairs = [(source_sents[src_idx], target_sents[tgt_idx]) for src_idx, tgt_idx in indeces]
        return sent_pairs, scores

    def score(self, source_sents, target_sents):
        source_embeddings, target_embeddings = self._embed(source_sents, target_sents)
        return cosine_similarity(source_embeddings, target_embeddings)

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
                batch_pairs, batch_scores = ratio_margin_align(source_embeddings, target_embeddings, self.k,
                        self.knn_batch_size, self.device)
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
                sents.append(line.decode().strip().split("\t"))
            return sents

    def train(self, source_sents, target_sents, overwrite=True):
        if not isfile(join(self.path, 'config.json')) or overwrite:
            # Convert train sentences to sentence pairs
            train_data = [InputExample(texts=[s, t]) for s, t in self.mine(source_sents, target_sents, self.train_size,
                    overwrite=overwrite)]

            # DataLoader to batch your data
            train_dataloader = DataLoader(train_data, batch_size=self.train_batch_size, shuffle=True)

            # Train a new model
            del self.model
            new_model = self.load_model(self.model_name)

            # Use contrastive learning loss
            if self.parallelize and device_count() > 1:
               logging.info(f"Training on {device_count()} GPUs.")
               train_loss = AdditiveMarginSoftmaxLoss(DataParallel(new_model))
            else:
               train_loss = AdditiveMarginSoftmaxLoss(new_model)

            # Call the fit method
            warmup_steps = ceil(len(train_dataloader) * self.num_epochs * 0.1)  # 10% of train data for warm-up
            new_model.fit(train_objectives=[(train_dataloader, train_loss)],
                epochs=self.num_epochs,
                warmup_steps=warmup_steps,
                optimizer_params={'lr': 5e-5}
            )
            new_model.save(self.path)

        self.model = SentenceTransformer(self.path, device=self.device)

class XLMoverScore(ContrastScore):
    """
    n_gram        - n-gram size of word mover's distance
    suffix_filter - filter embeddings of word suffixes (original XLMoverScore
        does this, but it doesn't make sense for SentencePiece-based Models)
    """
    def __init__(self, n_gram=1, suffix_filter=False, **kwargs):
        super().__init__(**kwargs)
        self.n_gram = n_gram
        self.suffix_filter = suffix_filter

    #Override
    def score(self, source_sents, target_sents):
        embedding_model = self.model[0]
        tokenizer = self.model.tokenizer

        src_inputs = tokenizer(source_sents, padding=True, truncation=True, return_tensors="pt")
        src_idf = src_inputs['attention_mask'].float()
        src_tokens = [[tokenizer.cls_token, *tokenizer.tokenize(sent), tokenizer.sep_token] for sent in source_sents]
        src_embeddings = embedding_model(**src_inputs)['last_hidden_state']

        tgt_inputs = tokenizer(target_sents, padding=True, truncation=True, return_tensors="pt").values()
        tgt_idf = tgt_inputs['attention_mask'].float()
        tgt_tokens = [[tokenizer.cls_token, *tokenizer.tokenize(sent), tokenizer.sep_token] for sent in target_sents]
        tgt_embeddings = embedding_model(**tgt_inputs)['last_hidden_state']

        return word_mover_score((src_embeddings, src_idf, src_tokens), (tgt_embeddings, tgt_idf, tgt_tokens),
                self.n_gram, True, self.suffix_filter)
