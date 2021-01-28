#!/usr/bin/env python
from aligner import XMoverAligner
from csv import reader
from itertools import islice
from os.path import isfile
from gzip import open as gopen
from urllib.request import urlretrieve
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, datefmt="%m-%d %H:%M", format="%(asctime)s %(levelname)-8s %(message)s")

source_lang, target_lang = "de", "en"
filename = f'news-commentary-v15.{source_lang}-{target_lang}.tsv.gz'
url = f'http://data.statmt.org/news-commentary/v15/training/{filename}'
path = (Path(__file__).parent / f'data/{filename}').resolve()
samples = 30000

if not isfile(path):
    logging.info(f"Downloading {filename} dataset.")
    urlretrieve(url, path)

source_data, target_data, source_remap, target_remap = list(), list(), list(), list()
with gopen(path, 'rt') as tsvfile:
    for src, tgt in islice(reader(tsvfile, delimiter="\t"), samples):
        source_data.append(src)
        target_data.append(tgt)
    for src, tgt in islice(reader(tsvfile, delimiter="\t"), samples, 2 * samples):
        source_remap.append(src)
        target_remap.append(tgt)
  
aligner = XMoverAligner()
logging.info(f"Precision @ 1 before remapping: {aligner.precision(source_data, target_data)}")
aligner.remap(source_remap, target_remap)
logging.info(f"Precision @ 1 after remapping: {aligner.precision(source_data, target_data)}")
