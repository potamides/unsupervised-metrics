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
from mosestokenizer import MosesDetokenizer
import logging

source_lang, target_lang = "de", "en"
iterations = 5
max_monolingual_sent_len = 80

monolingual_data = {
    "filenames": (f"news.{{}}.{source_lang}.shuffled.deduped.gz", f"news.{{}}.{target_lang}.shuffled.deduped.gz"),
    "urls": (f"http://data.statmt.org/news-crawl/{source_lang}", f"http://data.statmt.org/news-crawl/{target_lang}"),
    "versions": list(range(2007, 2020)),
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
    "filename": f"testset_{source_lang}-{target_lang}.tsv",
    "url": "https://github.com/AIPHES/ACL20-Reference-Free-MT-Evaluation/raw/master/WMT17/testset",
    "samples": 560,
    "path": str(Path(__file__).parent / "data"),
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
            # find first version that exists
            for version in dataset.get("versions", [None]):
                fileversion = filename.format(version)
                if not isfile(join(dataset["path"], fileversion)):
                    try:
                        urlretrieve(join(url, fileversion), join(dataset["path"], fileversion))
                        logging.info(f"Downloaded {fileversion} dataset.")
                        break
                    except URLError:
                        pass
                else:
                    break

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
        mpath, mfiles, versions = monolingual_data["path"], monolingual_data["filenames"], monolingual_data["versions"]
        src_paths = [join(mpath, mfiles[0].format(version)) for version in versions]
        tgt_paths = [join(mpath, mfiles[1].format(version)) for version in versions]
        src_path = next((path for path in src_paths if isfile(path)))
        tgt_path = next((path for path in tgt_paths if isfile(path)))
        with gopen(src_path, "rt") as f, gopen(tgt_path, "rt") as g:
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
            with open(join(news_eval_data["path"], news_eval_data["filename"]), newline='') as f:
                tsvdata = reader(f, delimiter="\t", quoting=QUOTE_NONE)
                with MosesDetokenizer(source_lang) as src_detokenize, MosesDetokenizer(target_lang) as tgt_detokenize:        
                    for _, src, mt, _, score, _ in islice(tsvdata, 1, news_eval_data["samples"] + 1):
                        eval_source.append(src_detokenize(src.split(' ')))
                        eval_system.append(tgt_detokenize(src.split(' ')))
                        eval_scores.append(float(score))
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
    eval_src, eval_system, eval_scores = extract_dataset(aligner.tokenizer.tokenize, "scored")

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
