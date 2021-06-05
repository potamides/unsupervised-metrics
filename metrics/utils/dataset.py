#!/usr/bin/env python
from csv import reader, QUOTE_NONE
from itertools import islice
from os import getenv
from os.path import isfile, join
from gzip import open as gopen
from tarfile import open as topen
from urllib.request import urlretrieve
from urllib.error import URLError
from pathlib import Path
from io import TextIOWrapper
from mosestokenizer import MosesTokenizer, MosesDetokenizer
from truecase import get_true_case
import logging

DATADIR = getenv("XMOVER_HOME", join(getenv("XDG_CACHE_HOME", join(Path.home(), ".cache")), "xmoverscore"))
Path(DATADIR).mkdir(parents=True, exist_ok=True)

class DatasetLoader():
    def __init__(self, source_language, target_language, min_monolingual_sent_len=3, max_monolingual_sent_len=80):
        self.source_lang = source_language
        self.target_lang = target_language
        self.min_monolingual_sent_len = min_monolingual_sent_len
        self.max_monolingual_sent_len = max_monolingual_sent_len

    @property
    def monolingual_data(self):
        return {
            "filenames": (
                f"news.2020.{self.source_lang}.shuffled.deduped.gz",
                f"news.2020.{self.target_lang}.shuffled.deduped.gz"
            ),
            "urls": (
                f"http://data.statmt.org/news-crawl/{self.source_lang}",
                f"http://data.statmt.org/news-crawl/{self.target_lang}"
            ),
            "samples": (40000, 10000000),
        }
    @property
    def parallel_data(self):
        return {
            "filenames": (
                # brute force try both directions, since order doesn't matter
                f"news-commentary-v15.{self.source_lang}-{self.target_lang}.tsv.gz",
                f"news-commentary-v15.{self.target_lang}-{self.source_lang}.tsv.gz"
            ),
            "urls": (
                "http://data.statmt.org/news-commentary/v15/training",
                "http://data.statmt.org/news-commentary/v15/training"
            ),
            "samples": (10000, 40000),
        }
    @property
    def news_eval_data(self):
        return {
            "filename": "DAseg-wmt-newstest2016.tar.gz",
            "url": "http://www.computing.dcu.ie/~ygraham",
            "samples": 560,
            "members": (
                f"DAseg-wmt-newstest2016/DAseg.newstest2016.source.{self.source_lang}-{self.target_lang}",
                f"DAseg-wmt-newstest2016/DAseg.newstest2016.mt-system.{self.source_lang}-{self.target_lang}",
                f"DAseg-wmt-newstest2016/DAseg.newstest2016.human.{self.source_lang}-{self.target_lang}",
            )
        }
    @property
    def wmt_eval_data(self):
        return {
            "filename": f"testset_{self.source_lang}-{self.target_lang}.tsv",
            "url": "https://github.com/AIPHES/ACL20-Reference-Free-MT-Evaluation/raw/master/WMT17/testset",
            "samples": 560,
        }
    @property
    def mlqe_eval_data(self): 
        return {
            "filename": f"{self.source_lang}-{self.target_lang}-test.tar.gz",
            "url": "https://github.com/sheffieldnlp/mlqe-pe/raw/master/data/direct-assessments/test",
            "member": f"{self.source_lang}-{self.target_lang}/test20.{self.source_lang}{self.target_lang}.df.short.tsv",
            "samples": 1000,
        }

    def download(_, dataset):
        logging.info(f"Checking datasets.")
        if "filename" in dataset and "url" in dataset:
            identifiers = ((dataset["filename"], dataset["url"]),)
        else:
            identifiers = zip(dataset["filenames"], dataset["urls"])
        for filename, url in identifiers:
            if not isfile(join(DATADIR, filename)):
                try:
                    urlretrieve(join(url, filename), join(DATADIR, filename))
                    logging.info(f"Downloaded {filename} dataset.")
                except URLError:
                    pass

    def load(self, name):
        if name in ["parallel", "parallel-align"]:
            self.download(self.parallel_data)
            parallel_source, parallel_target = list(), list()
            index = 0 if isfile(join(DATADIR, self.parallel_data["filenames"][0])) else 1
            with gopen(join(DATADIR, self.parallel_data["filenames"][index]), 'rt') as tsvfile:
                samples = self.parallel_data["samples"][1 if name.endswith("align") else 0]
                for src, tgt in islice(reader(tsvfile, delimiter="\t", quoting=QUOTE_NONE), samples):
                    if src.strip() and tgt.strip():
                        parallel_source.append(src if index == 0 else tgt)
                        parallel_target.append(tgt if index == 0 else src)
            return parallel_source, parallel_target

        elif name in ["monolingual-align", "monolingual-train"]:
            self.download(self.monolingual_data)
            mono_source, mono_target= list(), list()
            mpath, mfiles = DATADIR, self.monolingual_data["filenames"]
            with gopen(join(mpath, mfiles[0]), "rt") as f, gopen(join(mpath, mfiles[1]), "rt") as g, \
                MosesTokenizer(self.source_lang) as src_tokenize, MosesTokenizer(self.target_lang) as tgt_tokenize:        
                collected_src_samples, collected_tgt_samples = 0, 0
                for src in f:
                    if self.min_monolingual_sent_len <= len(src_tokenize(src)) <= self.max_monolingual_sent_len:
                        mono_source.append(src.strip())
                        collected_src_samples += 1
                        if collected_src_samples >= self.monolingual_data["samples"][1 if name.endswith("train") else 0]:
                            break
                for tgt in g:
                    if self.min_monolingual_sent_len <= len(tgt_tokenize(tgt)) < self.max_monolingual_sent_len:
                        mono_target.append(tgt.strip())
                        collected_tgt_samples += 1
                        if collected_tgt_samples >= self.monolingual_data["samples"][1 if name.endswith("train") else 0]:
                            break
            return mono_source, mono_target

        elif name in ["scored", "scored-mlqe", "scored-wmt"]:
            eval_source, eval_system, eval_scores = list(), list(), list()
            if name.endswith("mlqe"):
                self.download(self.mlqe_eval_data)
                samples, member = self.mlqe_eval_data["samples"], self.mlqe_eval_data["member"]
                with topen(join(DATADIR, self.mlqe_eval_data["filename"]), 'r:gz') as tf:
                    tsvdata = reader(TextIOWrapper(tf.extractfile(member)), delimiter="\t", quoting=QUOTE_NONE)
                    for _, src, mt, *_, score, _ in islice(tsvdata, 1, samples + 1):
                        eval_source.append(src.strip())
                        eval_system.append(mt.strip())
                        eval_scores.append(float(score))
            elif name.endswith("wmt"):
                self.download(self.wmt_eval_data)
                with open(join(DATADIR, self.wmt_eval_data["filename"]), newline='') as f:
                    tsvdata = reader(f, delimiter="\t")
                    with MosesDetokenizer(self.source_lang) as src_detokenize, \
                    MosesDetokenizer(self.target_lang) as tgt_detokenize:
                        for _, src, mt, _, score, _ in islice(tsvdata, 1, self.wmt_eval_data["samples"] + 1):
                            eval_source.append(src_detokenize(src.split()))
                            eval_system.append(get_true_case(tgt_detokenize(mt.split())))
                            eval_scores.append(float(score))
            else:
                self.download(self.news_eval_data)
                samples, members = self.news_eval_data["samples"], self.news_eval_data["members"]
                with topen(join(DATADIR, self.news_eval_data["filename"]), 'r:gz') as tf:
                    for src, mt, score in zip(*map(lambda x: islice(tf.extractfile(x), samples), members)):
                        eval_source.append(src.decode().strip())
                        eval_system.append(mt.decode().strip())
                        eval_scores.append(float(score.decode()))
            return eval_source, eval_system, eval_scores
        else:
            raise ValueError(f"{name} is not a valid type!")