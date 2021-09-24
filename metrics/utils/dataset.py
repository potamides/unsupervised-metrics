#!/usr/bin/env python
from csv import reader, QUOTE_NONE
from itertools import islice
from os import makedirs
from os.path import isfile, join, dirname
from gzip import open as gopen
from lzma import open as xopen
from tarfile import open as topen
from urllib.request import urlretrieve, urlopen
from urllib.error import URLError
from io import TextIOWrapper
from tqdm import tqdm
from logging import warn
from re import search
from pickle import load, dump
from gdown import cached_download
from linecache import getline
from .language import LangDetect, WordTokenizer, SentenceSplitter
from .env import DATADIR
from numpy import nan, nanmean, nanstd, empty

class DatasetLoader():
    def __init__(self, source_language, target_language,
            min_monolingual_sent_len=3, max_monolingual_sent_len=30,
            hard_limit=1000):
        """
        Initialize a dataloader for a given source and target language.

        Keyword arguments:
        min_monolingual_sent_len -- minimum amount of tokens in a sentence
            (sentences are tokenized based on language specific tokenizers)
        max_monolingual_sent_len -- maximum amount of tokens in a sentence
            (sentences are tokenized based on language specific tokenizers)
        hard_limit -- maximum allowed amount of characters in a sentence string
            (tokenizers sometimes tokenize very long garbage strings into few
            tokens which can lead to oom errors during training when not
            filtered out)
        """
        self.source_lang = source_language
        self.target_lang = target_language
        self.min_monolingual_sent_len = min_monolingual_sent_len
        self.max_monolingual_sent_len = max_monolingual_sent_len
        self.hard_limit = hard_limit

    @property
    def monolingual_data(self):
        return {
            "filenames": (
                f"news.{{}}.{self.source_lang}.shuffled.deduped.gz",
                f"news.{{}}.{self.target_lang}.shuffled.deduped.gz"
            ),
            "urls": (
                f"http://data.statmt.org/news-crawl/{self.source_lang}",
                f"http://data.statmt.org/news-crawl/{self.target_lang}"
            ),
            "fallback": {
                "filenames": (
                    f"{self.source_lang}.txt.xz",
                    f"{self.target_lang}.txt.xz"
                ),
                "urls": (
                    "http://data.statmt.org/cc-100",
                    "http://data.statmt.org/cc-100"
                ),
            },
            "versions": list(range(2007, 2021)),
            "samples": (40000, 4000000),
        }
    @property
    def parallel_data(self):
        return {
            "filenames": (
                # one of these two urls will exist (order doesn't matter)
                f"news-commentary-v15.{self.source_lang}-{self.target_lang}.tsv.gz",
                f"news-commentary-v15.{self.target_lang}-{self.source_lang}.tsv.gz"
            ),
            "urls": (
                "http://data.statmt.org/news-commentary/v15/training",
                "http://data.statmt.org/news-commentary/v15/training"
            ),
            "samples": (10000, 40000, 100000),
        }
    @property
    def wmt16_eval_data(self):
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
    def wmt17_eval_data(self):
        return {
            "filenames": (
                "wmt17-submitted-data-v1.0.tgz",
                "wmt17-metrics-task-package.tgz"
            ),
            "urls": (
                "http://data.statmt.org/wmt17/translation-task/",
                "http://ufallab.ms.mff.cuni.cz/~bojar/"
            ),
            "samples": 560 if self.source_lang != "zh" else 535
        }
    @property
    def mlqe_eval_data(self):
        return {
            "filename": f"{self.source_lang}-{self.target_lang}-test.tar.gz",
            "url": "https://github.com/sheffieldnlp/mlqe-pe/raw/master/data/direct-assessments/test",
            "member": f"{self.source_lang}-{self.target_lang}/test20.{self.source_lang}{self.target_lang}.df.short.tsv",
            "samples": 1000,
        }
    @property
    def mqm_eval_data(self):
        return {
            "filenames": (
                "newstest2020txt-v2.tar.gz",
                f"mqm_newstest2020_{self.source_lang+self.target_lang}.avg_seg_scores.tsv"
            ),
            "urls": (
                "https://drive.google.com/uc?id=1P-Y1P-GTMCNtWj8qaeq-U-m-0DGGnOaP",
                f"https://github.com/google/wmt-mqm-human-evaluation/raw/main/{self.source_lang+self.target_lang}"
            ),
            "samples": 20000 if self.source_lang == "zh" else 14180,
        }
    @property
    def wikimatrix_data(self):
        return {
            "filename": "WikiMatrix.{}-{}.tsv.gz".format(*sorted([self.source_lang, self.target_lang])),
            "url": "https://dl.fbaipublicfiles.com/laser/WikiMatrix/v1/",
            "samples": 10000,
        }
    @property
    def eval4nlp_eval_data(self):
        return {
            "filename": ("test21.sent.csv", f"eval4nlp-{self.source_lang}-{self.target_lang}-sent.csv"),
            "url": f"https://github.com/eval4nlp/test-data/raw/master/{self.source_lang}-{self.target_lang}-test21",
            "samples": 1410 if self.target_lang == "zh" else 1180,
        }

    def has_eval4nlp_access(self):
        url = self.eval4nlp_eval_data["url"]
        path = join(DATADIR, self.eval4nlp_eval_data["filename"][1])
        try:
            return isfile(path) or bool(urlopen(url))
        except URLError:
            return False

    def download(self, dataset, version=None):
        if "filename" in dataset and "url" in dataset:
            identifiers = ((dataset["filename"], dataset["url"]),)
        else:
            identifiers = zip(dataset["filenames"], dataset["urls"])
        for filename, url in identifiers:
            if isinstance(filename, (tuple, list)):
                filename, targetname = filename
            else:
                filename, targetname = filename, filename
            if version is not None:
                filename = filename.format(version)
                targetname = targetname.format(version)
            def progress(b=1, bsize=1, tsize=None):
                if not hasattr(self, "pbar"):
                    self.pbar = tqdm(unit='B', unit_scale=True, unit_divisor=1024, desc=f"Downloading {filename} dataset")
                if tsize is not None:
                    self.pbar.total = tsize
                return self.pbar.update(b * bsize - self.pbar.n)

            if not isfile(join(DATADIR, targetname)) and "drive.google.com" not in url:
                try:
                    urlretrieve(join(url, filename), join(DATADIR, targetname), progress)
                    del self.pbar
                except URLError as e:
                    if e.status != 404:
                        raise
            elif not isfile(join(DATADIR, targetname)):
                cached_download(url, join(DATADIR, targetname))

    def nanfloat(_, string):
        try:
            return float(string)
        except ValueError:
            return nan

    def zscore(_, scores):
        return (scores - nanmean(scores, 0)) / nanstd(scores, 0)

    def cc100_iter(self, language):
        filename, lines = self.monolingual_data["fallback"]["filenames"][0 if language == self.source_lang else 1], list()
        with xopen(join(DATADIR, filename)) as f, SentenceSplitter(language) as sent_split:
            for line in map(lambda line: line.strip(), f):
                if len(line) == 0:
                    for sentence in sent_split(lines):
                        yield sentence
                    lines.clear()
                else:
                    lines.append(line.decode())

    def filter(self, lang, sents, iterator, size, exclude):
        langdetect = LangDetect(cache_dir=DATADIR)
        with WordTokenizer(lang) as tokenize, SentenceSplitter(lang) as sent_split:
            for sent in map(lambda sent: sent.strip(), iterator):
                if len(sents) < size and all(not search(pattern, sent) for pattern in exclude) \
                and len(sent_split([sent])) == 1 and langdetect.detect(sent) == lang and len(sent) <= self.hard_limit \
                and self.min_monolingual_sent_len <= len(tokenize(sent)) <= self.max_monolingual_sent_len:
                    sents.add(sent)
        return sents

    def load_parallel(self, name, count):
        parallel_source, parallel_target = list(), list()
        if name.startswith("parallel"):
            self.download(self.parallel_data)
            index = 0 if isfile(join(DATADIR, self.parallel_data["filenames"][0])) else 1
            with gopen(join(DATADIR, self.parallel_data["filenames"][index]), 'rt') as tsvfile:
                start = self.parallel_data["samples"][0] if name.endswith("align") else 0
                samples = self.parallel_data["samples"][{"": 0, "align": 1, "train": 2}[name.partition("-")[2]]]
                for src, tgt in islice(reader(tsvfile, delimiter="\t", quoting=QUOTE_NONE), start, None):
                    if src.strip() and tgt.strip() and max(len(src), len(tgt)) < self.hard_limit:
                        parallel_source.append(src if index == 0 else tgt)
                        parallel_target.append(tgt if index == 0 else src)
                    if len(parallel_source) >= samples:
                        break
                else:
                    warn(f"Only obtained {len(parallel_source)} sentence pairs.")
        else:
            self.download(self.wikimatrix_data)
            with gopen(join(DATADIR, self.wikimatrix_data['filename']), 'rt') as f:
                for line in f:
                    score, sent1, sent2 = line.strip().split('\t')
                    sent1, sent2, score = sent1.strip(), sent2.strip(), float(score)
                    if sorted([self.source_lang, self.target_lang]).index(self.source_lang) == 1:
                        sent1, sent2 = sent2, sent1

                    if sent1 != sent2 and sent1 not in parallel_source and sent2 not in parallel_target \
                    and max(len(sent1), len(sent2)) < self.hard_limit:
                        parallel_source.append(sent1)
                        parallel_target.append(sent2)
                    if len(parallel_source) >= count or self.wikimatrix_data['samples']:
                        break
                else:
                    warn(f"Only obtained {len(parallel_source)} sentence pairs.")

        return parallel_source, parallel_target

    def load_monolingual(self, name):
        samples, patterns = self.monolingual_data["samples"][1 if name.endswith("train") else 0], list()
        cache_file = join(DATADIR, "preprocessed-datasets",
                f"{name}-{self.source_lang}-{self.target_lang}-{self.min_monolingual_sent_len}-{self.max_monolingual_sent_len}.pkl")
        makedirs(dirname(cache_file), exist_ok=True)
        if isfile(cache_file):
            with open(cache_file, 'rb') as f:
                return load(f)
        mono_source, mono_target = set(), set()
        for version in self.monolingual_data["versions"]:
            self.download(self.monolingual_data, version)
            patterns = ['https?://', str(version) + ", \d{1,2}:\d{2}"] # filter urls and date strings
            mpath, mfiles = DATADIR, [filename.format(version) for filename in self.monolingual_data["filenames"]]
            if isfile(join(mpath, mfiles[0])) and isfile(join(mpath, mfiles[1])):
                with gopen(join(mpath, mfiles[0]), "rt") as f, gopen(join(mpath, mfiles[1]), "rt") as g:
                    mono_source = self.filter(self.source_lang, mono_source, f, samples, patterns)
                    mono_target = self.filter(self.target_lang, mono_target, g, samples, patterns)
            elif version == self.monolingual_data["versions"][-1]:
                data = self.monolingual_data["fallback"]
                if isfile(join(mpath, mfiles[0])):
                    with gopen(join(mpath, mfiles[0]), "rt") as f:
                        mono_source = self.filter(self.source_lang, mono_source, f, samples, patterns)
                else:
                    self.download({"filename": data["filenames"][0], "url": data["urls"][0]})
                    mono_source = self.filter(self.source_lang, mono_source, self.cc100_iter(self.source_lang), samples, patterns)
                if isfile(join(mpath, mfiles[1])):
                    with gopen(join(mpath, mfiles[1]), "rt") as g:
                        mono_target = self.filter(self.target_lang, mono_target, g, samples, patterns)
                else:
                    self.download({"filename": data["filenames"][1], "url": data["urls"][1]})
                    mono_target = self.filter(self.target_lang, mono_target, self.cc100_iter(self.target_lang), samples, patterns)
            if min(len(mono_source), len(mono_target)) >= samples:
                break
        else:
            if min(len(mono_source), len(mono_target)) < samples:
                warn(f"Only obtained {len(mono_source)} source sentences and {len(mono_target)} target sentences.")
        with open(cache_file, 'wb') as f:
            mono_source, mono_target = list(mono_source), list(mono_target)
            dump((mono_source, mono_target), f)
            return mono_source, mono_target

    def load_scored(self, name):
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
        elif name.endswith("wmt17"):
            self.download(self.wmt17_eval_data)
            wmt_submitted, wmt_metrics = [join(DATADIR, name) for name in self.wmt17_eval_data["filenames"]]
            with topen(wmt_submitted, 'r:gz') as tf:
                members = [join('wmt17-submitted-data/txt', folder) for folder in ["sources", "system-outputs"]]
                tf.extractall(DATADIR, [tarinfo for tarinfo in tf.getmembers() if tarinfo.name.startswith(tuple(members))])
            with topen(wmt_metrics, 'r:gz') as tf:
                tf.extract("./manual-evaluation/DA-seglevel.csv", DATADIR)
            with open(join(DATADIR, "manual-evaluation/DA-seglevel.csv"), "rb") as f:
                for line in f.readlines()[1:]:
                    lp, _, system, sid, human = line.decode().split()
                    src, tgt = lp.split("-")
                    systems = system.split("+")
                    index = int(sid)
                    score = float(human)

                    if "ROCMT.5167" in systems: # doesn't seem to exist for zh-en
                        if len(systems) == 1:
                            continue
                        else:
                            systems.remove("ROCMT.5167")
                    if "CASICT-cons.5144" in systems: # there seem to be some typos for zh-en
                        systems[systems.index("CASICT-cons.5144")] = "CASICT-DCU-NMT.5144"

                    if src == self.source_lang and tgt == self.target_lang:
                        path = join(DATADIR, "wmt17-submitted-data/txt")
                        source_file = join(path, f"sources/newstest2017-{src}{tgt}-src.{src}")
                        hyp_template = join(path, f"system-outputs/newstest2017/{src}-{tgt}/newstest2017.{{}}.{src}-{tgt}")

                        source = getline(source_file, index).strip()
                        hypotheses = list()
                        for system in systems:
                            hyp_file = hyp_template.format(system)
                            hypothesis = getline(hyp_file, index).strip()
                            hypotheses.append(hypothesis)

                        assert min(len(source), len(hypotheses[0])) > 0 and len(set(hypotheses)) == 1

                        eval_source.append(source)
                        eval_system.append(hypotheses[0])
                        eval_scores.append(score)
                assert len(eval_scores) == len(eval_system) == len(eval_scores) == self.wmt17_eval_data['samples']
        elif name.endswith("mqm"):
            self.download(self.mqm_eval_data)
            wmt_submitted, mqm_file = [join(DATADIR, name) for name in self.mqm_eval_data["filenames"]]
            with topen(wmt_submitted, 'r:gz') as tf:
                members = [join('txt', folder) for folder in ["sources", "system-outputs"]]
                tf.extractall(DATADIR, [tarinfo for tarinfo in tf.getmembers() if tarinfo.name.startswith(tuple(members))])
            with open(mqm_file, "rb") as f:
                for line in f.readlines()[1:]:
                    system, mqm_avg_score, seg_id = line.decode().split()
                    seg_id, src, tgt = int(seg_id), self.source_lang, self.target_lang

                    source_file = join(DATADIR, f"txt/sources/newstest2020-{src}{tgt}-src.{src}.txt")
                    hyp_file = join(DATADIR, f"txt/system-outputs/{src}-{tgt}/newstest2020.{src}-{tgt}.{system}.txt")
                    source = getline(source_file, seg_id).strip()
                    hypothesis = getline(hyp_file, seg_id).strip()

                    assert source and hypothesis
                    eval_source.append(source)
                    eval_system.append(hypothesis)
                    eval_scores.append(float(mqm_avg_score))
                assert len(eval_scores) == len(eval_system) == len(eval_scores) == self.mqm_eval_data['samples']
        elif name.endswith("eval4nlp"):
            self.download(self.eval4nlp_eval_data)
            with open(join(DATADIR, self.eval4nlp_eval_data["filename"][1])) as csvfile:
                lines = csvfile.readlines()
                sources, targets, scores = list(), list(), empty((len(lines) - 1, 4))
                for idx, (_, source, target, score1, score2, score3, score4, _) in enumerate(reader(lines[1:])):
                    sources.append(source)
                    targets.append(target)
                    scores[idx] = [self.nanfloat(score1), self.nanfloat(score2), self.nanfloat(score3), self.nanfloat(score4)]

                return sources, targets, nanmean(self.zscore(scores), 1).tolist()
        else:
            self.download(self.wmt16_eval_data)
            samples, members = self.wmt16_eval_data["samples"], self.wmt16_eval_data["members"]
            with topen(join(DATADIR, self.wmt16_eval_data["filename"]), 'r:gz') as tf:
                for src, mt, score in zip(*map(lambda x: islice(tf.extractfile(x), samples), members)):
                    eval_source.append(src.decode().strip())
                    eval_system.append(mt.decode().strip())
                    eval_scores.append(float(score.decode()))
        return eval_source, eval_system, eval_scores

    def load(self, name, wikimatrix_count=None): # in a refactor it would make sense to allow this for all datasets
        if name in ["parallel", "parallel-align", "parallel-train", "wikimatrix"]:
            return self.load_parallel(name, wikimatrix_count)
        elif name in ["monolingual-align", "monolingual-train"]:
            return self.load_monolingual(name)
        elif name in ["scored", "scored-mlqe", "scored-wmt17", "scored-mqm", "scored-eval4nlp"]:
            return self.load_scored(name)
        else:
            raise ValueError(f"{name} is not a valid type!")
