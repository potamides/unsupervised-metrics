#!/usr/bin/env python
from aligner import XMoverAligner
from csv import reader
from itertools import islice
from os.path import isfile
from gzip import open as gopen
from urllib.request import urlretrieve
from pathlib import Path


filename = 'news-commentary-v15.de-en.tsv.gz'
url = f'http://data.statmt.org/news-commentary/v15/training/{filename}'
path = (Path(__file__).parent / f'datasets/{filename}').resolve()
samples = 3000

if not isfile(path):
    urlretrieve(url, path)

source_data, target_data = list(), list()
with gopen(path, 'rt') as tsvfile:
    for src, tgt in islice(reader(tsvfile, delimiter="\t"), samples):
        source_data.append(src)
        target_data.append(tgt)
  
print(f"Accuracy: {XMoverAligner().accuracy_on_data(source_data, target_data)}")
