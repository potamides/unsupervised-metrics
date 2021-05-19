#!/usr/bin/env python
from aligner import RatioMarginBertAligner, XMoverBertAligner, XMoverVecMapAligner, XMoverNMTBertAligner
from csv import reader, QUOTE_NONE
from itertools import islice
from os.path import isfile, join
from gzip import open as gopen
from tarfile import open as topen
from urllib.request import urlretrieve
from urllib.error import URLError
from pathlib import Path
from nltk import tokenize
from io import TextIOWrapper
import logging

source_lang, target_lang = "en", "de"
iterations = 5
max_monolingual_sent_len = 80

monolingual_data = {
    "filenames": (f"news.2007.{source_lang}.shuffled.deduped.gz", f"news.2007.{target_lang}.shuffled.deduped.gz"),
    "urls": (f"http://data.statmt.org/news-crawl/{source_lang}", f"http://data.statmt.org/news-crawl/{target_lang}"),
    "samples": 20000,
    "path": str(Path(__file__).parent / "data")
}
parallel_data = {
    "filenames": (
        # brute force try both directions, since order doesn't matter
        f"news-commentary-v15.{source_lang}-{target_lang}.tsv.gz",
        f"news-commentary-v15.{target_lang}-{source_lang}.tsv.gz"
    ),
    "urls": ("http://data.statmt.org/news-commentary/v15/training", ),
    "samples": 3000,
    "path": str(Path(__file__).parent / "data")
}
news_eval_data = {
    "filename": "DAseg-wmt-newstest2016.tar.gz",
    "url": "http://www.computing.dcu.ie/~ygraham",
    "samples": 560,
    "path": str(Path(__file__).parent / "data"),
    "members": (
        f"DAseg-wmt-newstest2016/DAseg.newstest2016.source.{source_lang}-{target_lang}",
        f"DAseg-wmt-newstest2016/DAseg.newstest2016.mt-system.{source_lang}-{target_lang}",
        f"DAseg-wmt-newstest2016/DAseg.newstest2016.human.{source_lang}-{target_lang}",
    )
}
mlqe_eval_data = {
    "filename": f"{source_lang}-{target_lang}-test.tar.gz",
    "url": "https://github.com/sheffieldnlp/mlqe-pe/raw/master/data/direct-assessments/test",
    "member": f"{source_lang}-{target_lang}/test20.{source_lang}{target_lang}.df.short.tsv",
    "samples": 1000,
    "path": str(Path(__file__).parent / "data")
}

def download_datasets():
    for dataset in (parallel_data, monolingual_data, news_eval_data, mlqe_eval_data):
        if "filename" in dataset and "url" in dataset:
            identifiers = ((dataset["filename"], dataset["url"]),)
        else:
            identifiers = zip(dataset["filenames"], dataset["urls"])
        for filename, url in identifiers:
            if not isfile(join(dataset["path"], filename)):
                try:
                    urlretrieve(join(url, filename), join(dataset["path"], filename))
                    logging.info(f"Downloaded {filename} dataset.")
                except URLError:
                    pass

def extract_dataset(tokenize, type_, monolingual_full=False, use_mlqe=False):
    if type_ == "parallel":
        parallel_source, parallel_target = list(), list()
        index = 0 if isfile(join(parallel_data["path"], parallel_data["filenames"][0])) else 1
        with gopen(join(parallel_data["path"], parallel_data["filenames"][index]), 'rt') as tsvfile:
            for src, tgt in islice(reader(tsvfile, delimiter="\t", quoting=QUOTE_NONE), parallel_data["samples"]):
                if src.strip() and tgt.strip():
                    parallel_source.append(src if index == 0 else tgt)
                    parallel_target.append(tgt if index == 0 else src)
        return parallel_source, parallel_target

    elif type_ == "monolingual":
        mono_source, mono_target= list(), list()
        mpath, mfilenames = monolingual_data["path"], monolingual_data["filenames"]
        with gopen(join(mpath, mfilenames[0]), "rt") as f, gopen(join(mpath, mfilenames[1]), "rt") as g:
            collected_src_samples, collected_tgt_samples = 0, 0
            for src in f:
                if len(tokenize(src)) < max_monolingual_sent_len:
                    mono_source.append(src.strip())
                    collected_src_samples += 1
                    if not monolingual_full and collected_src_samples >= monolingual_data["samples"]:
                        break
            for tgt in g:
                if len(tokenize(tgt)) < max_monolingual_sent_len:
                    mono_target.append(tgt.strip())
                    collected_tgt_samples += 1
                    if not monolingual_full and collected_tgt_samples >= monolingual_data["samples"]:
                        break
        return mono_source, mono_target

    elif type_ == "scored":
        eval_source, eval_system, eval_scores = list(), list(), list()
        if use_mlqe:
            samples, member = mlqe_eval_data["samples"], mlqe_eval_data["member"]
            with topen(join(mlqe_eval_data["path"], mlqe_eval_data["filename"]), 'r:gz') as tf:
                tsvdata = reader(TextIOWrapper(tf.extractfile(member)), delimiter="\t", quoting=QUOTE_NONE)
                for _, src, mt, *_, score, _ in islice(tsvdata, 1, samples + 1):
                    eval_source.append(src.strip())
                    eval_system.append(mt.strip())
                    eval_scores.append(float(score))
        else:
            samples, members = news_eval_data["samples"], news_eval_data["members"]
            with topen(join(news_eval_data["path"], news_eval_data["filename"]), 'r:gz') as tf:
                for src, mt, score in zip(*map(lambda x: islice(tf.extractfile(x), samples), members)):
                    eval_source.append(src.decode().strip())
                    eval_system.append(mt.decode().strip())
                    eval_scores.append(float(score.decode()))
        return eval_source, eval_system, eval_scores
    else:
        raise ValueError(f"{type_} is not a valid type!")

def bert_tests(use_ratio_margin=False):
    aligner = RatioMarginBertAligner() if use_ratio_margin else XMoverBertAligner()
    parallel_src, parallel_tgt = extract_dataset(aligner.tokenizer.tokenize, "parallel")
    mono_src, mono_tgt = extract_dataset(aligner.tokenizer.tokenize, "monolingual")
    eval_src, eval_system, eval_scores = extract_dataset(aligner.tokenizer.tokenize, "scored")

    logging.info("Evaluating performance before remapping.")
    logging.info(f"Precision @ 1: {aligner.precision(parallel_src, parallel_tgt)}")
    logging.info("Pearson: {}, Spearman: {}".format(*aligner.correlation(eval_src, eval_system, eval_scores)))
    for iteration in range(1, iterations + 1):
        logging.info(f"Remapping iteration {iteration}.")
        aligner.remap(mono_src, mono_tgt)
        logging.info(f"Precision @ 1: {aligner.precision(parallel_src, parallel_tgt)}")
        logging.info("Pearson: {}, Spearman: {}".format(*aligner.correlation(eval_src, eval_system, eval_scores)))
  
def vecmap_tests():
    aligner = XMoverVecMapAligner(src_lang=source_lang, tgt_lang=target_lang)
    parallel_src, parallel_tgt = extract_dataset(tokenize, "parallel")
    eval_src, eval_system, eval_scores = extract_dataset(tokenize, "scored")

    logging.info(f"Precision: {aligner.precision(parallel_src, parallel_tgt)}.")
    logging.info("Pearson: {}, Spearman: {}".format(*aligner.correlation(eval_src, eval_system, eval_scores)))

def nmt_tests():
    aligner = XMoverNMTBertAligner(src_lang=source_lang, tgt_lang=target_lang)
    mono_src, mono_tgt = extract_dataset(aligner.tokenizer.tokenize, "monolingual")
    eval_src, eval_system, eval_scores = extract_dataset(aligner.tokenizer.tokenize, "scored", use_mlqe=True)

    logging.info("Evaluating performance before remapping.")
    logging.info("Pearson: {}, Spearman: {}".format(*aligner.correlation(eval_src, eval_system, eval_scores)))
    logging.info("RMSE: {}, MAE: {}".format(*aligner.error(eval_src, eval_system, eval_scores)))
    for iteration in range(1, iterations + 1):
        logging.info(f"Remapping iteration {iteration}.")
        aligner.remap(mono_src, mono_tgt)
        logging.info("Pearson: {}, Spearman: {}".format(*aligner.correlation(eval_src, eval_system, eval_scores)))
        logging.info("RMSE: {}, MAE: {}".format(*aligner.error(eval_src, eval_system, eval_scores)))
    mono_src, mono_tgt = extract_dataset(aligner.tokenizer.tokenize, "monolingual", True)
    aligner.train(mono_src, mono_tgt, False)

    logging.info("Evaluating performance with NMT model.")
    logging.info("Pearson: {}, Spearman: {}".format(*aligner.correlation(eval_src, eval_system, eval_scores)))
    logging.info("RMSE: {}, MAE: {}".format(*aligner.error(eval_src, eval_system, eval_scores)))

logging.basicConfig(level=logging.INFO, datefmt="%m-%d %H:%M", format="%(asctime)s %(levelname)-8s %(message)s")
download_datasets()
#bert_tests()
#bert_tests(True)
#vecmap_tests()
nmt_tests()
