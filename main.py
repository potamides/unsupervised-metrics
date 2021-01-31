#!/usr/bin/env python
from aligner import XMoverAligner
from csv import reader, QUOTE_NONE
from itertools import islice
from os.path import isfile, join
from gzip import open as gopen
from tarfile import open as topen
from urllib.request import urlretrieve
from pathlib import Path
import logging

source_lang, target_lang = "de", "en"
iterations = 5
max_monolingual_sent_len = 80

parallel_data = {
    "filename": f"news-commentary-v15.{source_lang}-{target_lang}.tsv.gz",
    "url": "http://data.statmt.org/news-commentary/v15/training",
    "samples": 3000,
    "path": str(Path(__file__).parent / "data")
}
monolingual_data = {
    "filenames": (f"news.2007.{source_lang}.shuffled.deduped.gz", f"news.2007.{target_lang}.shuffled.deduped.gz"),
    "urls": (f"http://data.statmt.org/news-crawl/{source_lang}", f"http://data.statmt.org/news-crawl/{target_lang}"),
    "samples": 20000,
    "path": str(Path(__file__).parent / "data")
}
eval_data = {
    "filename": "DAseg-wmt-newstest2016.tar.gz",
    "url": "http://www.computing.dcu.ie/~ygraham",
    "samples": 560,
    "path": str(Path(__file__).parent / "data"),
    "members": (
        f"DAseg-wmt-newstest2016/DAseg.newstest2016.source.{source_lang}-{target_lang}",
        f"DAseg-wmt-newstest2016/DAseg.newstest2016.reference.{source_lang}-{target_lang}",
        f"DAseg-wmt-newstest2016/DAseg.newstest2016.mt-system.{source_lang}-{target_lang}",
        f"DAseg-wmt-newstest2016/DAseg.newstest2016.human.{source_lang}-{target_lang}",
    )
}

def download_datasets():
    for dataset in (parallel_data, monolingual_data, eval_data):
        if "filename" in dataset and "url" in dataset:
            identifiers = ((dataset["filename"], dataset["url"]),)
        else:
            identifiers = zip(dataset["filenames"], dataset["urls"])
        for filename, url in identifiers:
            if not isfile(join(dataset["path"], filename)):
                logging.info(f"Downloading {filename} dataset.")
                urlretrieve(join(url, filename), join(dataset["path"], filename))

def extract_datasets(tokenize):
    parallel_source, parallel_target = list(), list()
    with gopen(join(parallel_data["path"], parallel_data["filename"]), 'rt') as tsvfile:
        for src, tgt in islice(reader(tsvfile, delimiter="\t", quoting=QUOTE_NONE), parallel_data["samples"]):
            parallel_source.append(src)
            parallel_target.append(tgt)

    mono_source, mono_target= list(), list()
    mpath, mfilenames = monolingual_data["path"], monolingual_data["filenames"]
    with gopen(join(mpath, mfilenames[0]), "rt") as f, gopen(join(mpath, mfilenames[1]), "rt") as g:
        collected_src_samples, collected_tgt_samples = 0, 0
        for src in f:
            if len(tokenize(src)) < max_monolingual_sent_len:
                mono_source.append(src.strip())
                collected_src_samples += 1
                if collected_src_samples >= monolingual_data["samples"]:
                    break
        for tgt in g:
            if len(tokenize(tgt)) < max_monolingual_sent_len:
                mono_target.append(tgt.strip())
                collected_tgt_samples += 1
                if collected_tgt_samples >= monolingual_data["samples"]:
                    break

    eval_source, eval_ref, eval_system, eval_scores = list(), list(), list(), list()
    samples, members = eval_data["samples"], eval_data["members"]
    with topen(join(eval_data["path"], eval_data["filename"]), 'r:gz') as tf:
        for src, ref, mt, score in zip(*map(lambda x: islice(tf.extractfile(x), samples), members)):
            eval_source.append(src.decode().strip())
            eval_ref.append(ref.decode().strip())
            eval_system.append(mt.decode().strip())
            eval_scores.append(float(score.decode()))

    return parallel_source, parallel_target, mono_source, mono_target, eval_source, eval_ref, eval_system, eval_scores
  
logging.basicConfig(level=logging.INFO, datefmt="%m-%d %H:%M", format="%(asctime)s %(levelname)-8s %(message)s")
download_datasets()
aligner = XMoverAligner()
parallel_src, parallel_tgt, mono_src, mono_tgt, eval_src, eval_ref, eval_system, eval_scores = extract_datasets(
        aligner.tokenizer.tokenize)

logging.info(f"Precision @ 1 before remapping: {aligner.precision(parallel_src, parallel_tgt)}.")
logging.info(f"Pearson correlation before remapping: {aligner.correlation(eval_src, eval_system, eval_scores)}.")
for iteration in range(1, iterations + 1):
    logging.info(f"Remapping iteration {iteration}.")
    aligner.remap(mono_src, mono_tgt)
    logging.info(f"Precision @ 1 after remapping: {aligner.precision(parallel_src, parallel_tgt)}")
    logging.info(f"Pearson correlation after remapping: {aligner.correlation(eval_src, eval_system, eval_scores)}")
