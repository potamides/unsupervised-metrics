from transformers import BertModel, BertTokenizer, BertConfig
from ..utils.embed import bert_embed, vecmap_embed, map_multilingual_embeddings
from ..utils.remap import fast_align, awesome_align, sim_align, get_aligned_features_avgbpe, clp, umd
from ..utils.env import DATADIR
from ..common import CommonScore
from os.path import isfile, join
from nltk.metrics.distance import edit_distance
from numpy import load
from io import BytesIO
from functools import cached_property
from urllib.request import urlopen
from urllib.error import URLError
import logging
import torch

class BertEmbed(CommonScore):
    def __init__(self, model_name, monolingual_model_name, mapping, device, do_lower_case, embed_batch_size):
        self.model_name = model_name
        self.monolingual_model_name = monolingual_model_name if monolingual_model_name else model_name
        self.do_lower_case = do_lower_case
        self.device = device
        self.mapping = mapping
        self.embed_batch_size = embed_batch_size
        self.projection = None

    @cached_property
    def model(self):
        config = BertConfig.from_pretrained(self.model_name, output_hidden_states=True)
        return BertModel.from_pretrained(self.model_name, config=config).to(self.device)

    @cached_property
    def monolingual_model(self):
        if self.monolingual_model_name != self.model_name:
            config = BertConfig.from_pretrained(self.monolingual_model_name, output_hidden_states=True)
            return BertModel.from_pretrained(self.monolingual_model_name, config=config).to(self.device)
        else:
            return self.model

    @cached_property
    def tokenizer(self):
        return BertTokenizer.from_pretrained(self.model_name, do_lower_case=self.do_lower_case)

    @cached_property
    def monolingual_tokenzier(self):
        if self.monolingual_model_name != self.model_name:
            return BertTokenizer.from_pretrained(self.monolingual_model_name, do_lower_case=self.do_lower_case)
        else:
            return self.tokenizer

    def _embed(self, source_sents, target_sents, same_language=False):
        model, tokenizer = (self.monolingual_model, self.monolingual_tokenzier) if same_language else (self.model, self.tokenizer)
        src_embeddings, src_idf, src_tokens, src_mask = bert_embed(source_sents, self.embed_batch_size, model,
                tokenizer, self.device)
        tgt_embeddings, tgt_idf, tgt_tokens, tgt_mask = bert_embed(target_sents, self.embed_batch_size, model,
                tokenizer, self.device)

        if self.projection is not None and not same_language:
            if self.mapping == 'CLP':
                src_embeddings = torch.matmul(src_embeddings, self.projection)
            else:
                src_embeddings = src_embeddings - (src_embeddings * self.projection).sum(2, keepdim=True) * \
                        self.projection.repeat(src_embeddings.shape[0], src_embeddings.shape[1], 1)

        return src_embeddings, src_idf, src_tokens, src_mask, tgt_embeddings, tgt_idf, tgt_tokens, tgt_mask

class BertRemap(BertEmbed):
    def __init__(self, model_name, monolingual_model_name, mapping, device, do_lower_case, remap_size, embed_batch_size, alignment):
        super().__init__(model_name, monolingual_model_name, mapping, device, do_lower_case, embed_batch_size)
        self.remap_size = remap_size
        self.alignment = alignment

    def remap(self, source_sents, target_sents, suffix="tensor", aligned=False, overwrite=True, new_mapping=None):
        file_path, mapping = join(DATADIR, f"projection-{suffix}.pt"), new_mapping or self.mapping
        if not isfile(file_path) or overwrite:
            logging.info(f'Computing projection tensor for {mapping} remapping method.')
            sorted_sent_pairs = list()
            if aligned:
                sorted_sent_pairs.extend(zip(source_sents, target_sents))
            else:
                sent_pairs, scores = self.align(source_sents, target_sents)
                for _, (src_sent, tgt_sent) in sorted(zip(scores, sent_pairs), key=lambda tup: tup[0], reverse=True):
                    if edit_distance(src_sent, tgt_sent) / max(len(src_sent), len(tgt_sent)) > 0.5:
                        sorted_sent_pairs.append((src_sent, tgt_sent))
            if self.alignment == "fast":
                tokenized_pairs, align_pairs = fast_align(sorted_sent_pairs, self.tokenizer, self.remap_size)
            elif self.alignment == "sim":
                tokenized_pairs, align_pairs = sim_align(sorted_sent_pairs, self.tokenizer, self.remap_size, self.device)
            else: # awesome
                tokenized_pairs, align_pairs = awesome_align(sorted_sent_pairs, self.model, self.tokenizer,
                        self.remap_size, self.device)
                if self.alignment.endswith("remap"): # awesome-remap
                    src_matrix, tgt_matrix = get_aligned_features_avgbpe(tokenized_pairs, align_pairs,
                            self.model, self.tokenizer, self.embed_batch_size, self.device, 8)
                    tokenized_pairs, align_pairs = awesome_align(sorted_sent_pairs, self.model, self.tokenizer,
                            self.remap_size, self.device,
                            clp(src_matrix, tgt_matrix) if mapping == "CLP" else umd(src_matrix, tgt_matrix))
            src_matrix, tgt_matrix = get_aligned_features_avgbpe(tokenized_pairs, align_pairs,
                    self.model, self.tokenizer, self.embed_batch_size, self.device)

            logging.info(f"Using {len(src_matrix)} aligned word pairs to compute projection tensor.")
            if mapping == "CLP":
                self.projection = clp(src_matrix, tgt_matrix)
            else:
                self.projection = umd(src_matrix, tgt_matrix)
            torch.save(self.projection, file_path)
        else:
            logging.info(f'Loading {mapping} projection tensor from disk.')
            self.projection = torch.load(file_path)
        if new_mapping:
            self.mapping = new_mapping

class BertRemapPretrained(BertEmbed):
    """
    Obtains pretrained remapping matrices from original XMoverScore repository.
    """

    commit = "73ef48058f8e47e0d99434b7c75a9ceb6f253d94"
    path = "mapping/layer-12/{}.{}-{}.2k.12.{}"
    url = f"https://github.com/potamides/ACL20-Reference-Free-MT-Evaluation/raw/{commit}/{path}"

    def remap(self, source_lang, target_lang):
        for corpus in ["europarl-v7", "flores-v1", "un-v1", "wikimedia-v20210402", "wikimatrix-v1", "multi-cc-aligned-v1.1"]:
            try:
                if self.mapping == "CLP":
                    download = urlopen(self.url.format(corpus, source_lang, target_lang, "BAM")).read()
                    self.projection = torch.tensor(load(BytesIO(download)), dtype=torch.float32)
                else:
                    download = urlopen(self.url.format(corpus, source_lang, target_lang, "GBDD")).read()
                    self.projection = torch.tensor(load(BytesIO(download))[0], dtype=torch.float32)
                break
            except URLError as e:
                if e.status == 404:
                    pass
        else:
            raise ValueError("Language direction does not exist!")

class VecMapEmbed(CommonScore):
    def __init__(self, device, src_lang, tgt_lang, batch_size):
        self.device = device
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.batch_size = batch_size
        self.src_dict = None
        self.tgt_dict = None

    def _embed(self, source_sents, target_sents, same_language=False):
        if self.src_dict is None or self.tgt_dict is None:
            logging.info("Obtaining cross-lingual word embedding mappings from fasttext embeddings.")
            self.src_dict, self.tgt_dict = map_multilingual_embeddings(self.src_lang, self.tgt_lang,
                self.batch_size, self.device)
        src_embeddings, src_idf, src_tokens, src_mask = vecmap_embed(source_sents,
                *((self.tgt_dict, self.tgt_lang) if same_language else (self.src_dict, self.src_lang)))
        tgt_embeddings, tgt_idf, tgt_tokens, tgt_mask = vecmap_embed(target_sents, self.tgt_dict, self.tgt_lang)

        return src_embeddings, src_idf, src_tokens, src_mask, tgt_embeddings, tgt_idf, tgt_tokens, tgt_mask
