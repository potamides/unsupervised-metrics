#!/usr/bin/env python
from aligner import XMoverBertAligner, XMoverVecMapAligner, XMoverNMTBertAligner
from csv import reader, QUOTE_NONE
from itertools import islice
from os.path import isfile, join
from gzip import open as gopen
from tarfile import open as topen
from urllib.request import urlretrieve
from urllib.error import URLError
from pathlib import Path
from io import TextIOWrapper
from mosestokenizer import MosesTokenizer
from linecache import getline
from truecase import get_true_case
from mosestokenizer import MosesDetokenizer as Detokenizer
import logging

source_lang, target_lang = "de", "en"
iterations = 5
min_monolingual_sent_len, max_monolingual_sent_len = 3, 80

monolingual_data = {
    "filenames": (f"news.2020.{source_lang}.shuffled.deduped.gz", f"news.2020.{target_lang}.shuffled.deduped.gz"),
    "urls": (f"http://data.statmt.org/news-crawl/{source_lang}", f"http://data.statmt.org/news-crawl/{target_lang}"),
    "samples": (40000, 10000000),
    "path": str(Path(__file__).parent / "data")
}
parallel_data = {
    "filenames": (
        # brute force try both directions, since order doesn't matter
        f"news-commentary-v15.{source_lang}-{target_lang}.tsv.gz",
        f"news-commentary-v15.{target_lang}-{source_lang}.tsv.gz"
    ),
    "urls": (
        "http://data.statmt.org/news-commentary/v15/training",
        "http://data.statmt.org/news-commentary/v15/training"
    ),
    "samples": (10000, 40000),
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
wmt_eval_data = {
    "filenames": (
        "DA-seglevel.csv",
        f"newstest2017-{source_lang}{target_lang}-src.{source_lang}",
        f"newstest2017.{{}}.{source_lang}-{target_lang}"
    ),
    "urls": (
        "https://github.com/AIPHES/ACL20-Reference-Free-MT-Evaluation/raw/master/WMT17",
        "https://github.com/AIPHES/ACL20-Reference-Free-MT-Evaluation/raw/master/WMT17/source",
        f"https://github.com/AIPHES/ACL20-Reference-Free-MT-Evaluation/raw/master/WMT17/system-outputs/{source_lang}-{target_lang}"
        ),
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
    for dataset in (parallel_data, monolingual_data, news_eval_data, wmt_eval_data, mlqe_eval_data):
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

def extract_dataset(type_, ):
    if type_ in ["parallel", "parallel-align"]:
        parallel_source, parallel_target = list(), list()
        index = 0 if isfile(join(parallel_data["path"], parallel_data["filenames"][0])) else 1
        with gopen(join(parallel_data["path"], parallel_data["filenames"][index]), 'rt') as tsvfile:
            samples = parallel_data["samples"][1 if type_.endswith("align") else 0]
            for src, tgt in islice(reader(tsvfile, delimiter="\t", quoting=QUOTE_NONE), samples):
                if src.strip() and tgt.strip():
                    parallel_source.append(src if index == 0 else tgt)
                    parallel_target.append(tgt if index == 0 else src)
        return parallel_source, parallel_target

    elif type_ in ["monolingual-align", "monolingual-train"]:
        mono_source, mono_target= list(), list()
        mpath, mfiles = monolingual_data["path"], monolingual_data["filenames"]
        with gopen(join(mpath, mfiles[0]), "rt") as f, gopen(join(mpath, mfiles[1]), "rt") as g, \
            MosesTokenizer(source_lang) as src_tokenize, MosesTokenizer(target_lang) as tgt_tokenize:        
            collected_src_samples, collected_tgt_samples = 0, 0
            for src in f:
                if min_monolingual_sent_len <= len(src_tokenize(src)) <= max_monolingual_sent_len:
                    mono_source.append(src.strip())
                    collected_src_samples += 1
                    if collected_src_samples >= monolingual_data["samples"][1 if type_.endswith("train") else 0]:
                        break
            for tgt in g:
                if min_monolingual_sent_len <= len(tgt_tokenize(tgt)) < max_monolingual_sent_len:
                    mono_target.append(tgt.strip())
                    collected_tgt_samples += 1
                    if collected_tgt_samples >= monolingual_data["samples"][1 if type_.endswith("train") else 0]:
                        break
        return mono_source, mono_target

    elif type_ in ["scored", "scored-mlqe", "scored-wmt"]:
        eval_source, eval_system, eval_scores = list(), list(), list()
        if type_.endswith("mlqe"):
            samples, member = mlqe_eval_data["samples"], mlqe_eval_data["member"]
            with topen(join(mlqe_eval_data["path"], mlqe_eval_data["filename"]), 'r:gz') as tf:
                tsvdata = reader(TextIOWrapper(tf.extractfile(member)), delimiter="\t", quoting=QUOTE_NONE)
                for _, src, mt, *_, score, _ in islice(tsvdata, 1, samples + 1):
                    eval_source.append(src.strip())
                    eval_system.append(mt.strip())
                    eval_scores.append(float(score))
        elif type_.endswith("wmt"):
            path = join(wmt_eval_data["path"], wmt_eval_data["filenames"][0])
            with open(path, "rb") as f, Detokenizer(source_lang) as src_detokenize, Detokenizer(target_lang) as tgt_detokenize:
                for line in f.readlines()[1:]:
                    lp, _, system, sid, human = line.decode().split()
                    (src, tgt), index, score = lp.split("-"), int(sid), float(human)
                    systems = system.split("+")

                    if src == source_lang and tgt == target_lang:
                        source = src_detokenize(getline(join(wmt_eval_data["path"], wmt_eval_data["filenames"][1]), index).split())
                        hypotheses = list()
                        for system in systems:
                            hyp_file = wmt_eval_data["filenames"][2].format(system)
                            if not isfile(join(wmt_eval_data["path"], hyp_file)):
                                urlretrieve(join(wmt_eval_data["urls"][2], hyp_file), join(wmt_eval_data["path"], hyp_file))
                            hypothesis = getline(join(wmt_eval_data["path"], hyp_file), index)
                            hypotheses.append(get_true_case(tgt_detokenize(hypothesis.split())))
                        assert len(set(hypotheses)) == 1
                        eval_source.append(source)
                        eval_system.append(hypotheses[0])
                        eval_scores.append(score)
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

def align_tests(alignment="awesome", mapping="UMD", data="monolingual-align", valid="scored", metric="cosine"):
    aligner = XMoverBertAligner(alignment=alignment, mapping=mapping, use_cosine=True if metric == "cosine" else False)
    parallel_src, parallel_tgt = extract_dataset("parallel")
    mono_src, mono_tgt = extract_dataset(data)
    eval_src, eval_system, eval_scores = extract_dataset(valid)
    suffix = f"{source_lang}-{target_lang}-{alignment}-{metric}-{mapping}-{data}-{aligner.k}-{aligner.remap_size}-{len(mono_src)}"
    results = {"suffix": suffix}

    logging.info("Evaluating performance before remapping.")
    precision = aligner.precision(parallel_src, parallel_tgt)
    pearson, spearman = aligner.correlation(eval_src, eval_system, eval_scores)
    rmse, mae = aligner.error(eval_src, eval_system, eval_scores)
    logging.info(f"Precision @ 1: {precision}")
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}")
    logging.info(f"RMSE: {rmse}, MAE: {mae}")
    results[0] = {"precision": round(precision, 3), "pearson": round(pearson, 3), "spearman": round(spearman, 3),
            "rmse": round(rmse, 3), "mae": round(mae, 3)}

    for iteration in range(1, iterations + 1):
        logging.info(f"Remapping iteration {iteration}.")
        aligner.remap(mono_src, mono_tgt, suffix=suffix + f"-{iteration}", overwrite=False)
        precision = aligner.precision(parallel_src, parallel_tgt)
        pearson, spearman = aligner.correlation(eval_src, eval_system, eval_scores)
        rmse, mae = aligner.error(eval_src, eval_system, eval_scores)
        logging.info(f"Precision @ 1: {precision}")
        logging.info(f"Pearson: {pearson}, Spearman: {spearman}")
        logging.info(f"RMSE: {rmse}, MAE: {mae}")
        results[iteration] = {"precision": round(precision, 3), "pearson": round(pearson, 3),
                "spearman": round(spearman, 3), "rmse": round(rmse, 3), "mae": round(mae, 3)}

    return results
  
def vecmap_tests():
    aligner = XMoverVecMapAligner(src_lang=source_lang, tgt_lang=target_lang)
    parallel_src, parallel_tgt = extract_dataset("parallel")
    eval_src, eval_system, eval_scores = extract_dataset("scored")

    logging.info(f"Precision: {aligner.precision(parallel_src, parallel_tgt)}.")
    logging.info("Pearson: {}, Spearman: {}".format(*aligner.correlation(eval_src, eval_system, eval_scores)))
    logging.info("RMSE: {}, MAE: {}".format(*aligner.error(eval_src, eval_system, eval_scores)))

def nmt_tests(valid="scored", metric="cosine"):
    aligner = XMoverNMTBertAligner(src_lang=source_lang, tgt_lang=target_lang, use_cosine=True if metric == "cosine" else False)
    mono_src, mono_tgt = extract_dataset("monolingual-align")
    eval_src, eval_system, eval_scores = extract_dataset(valid)
    suffix = f"{source_lang}-{target_lang}-awesome-{metric}-{aligner.mapping}-monolingual-align-{aligner.k}-{aligner.remap_size}-{len(mono_src)}"
    results = {"suffix": suffix}

    logging.info("Evaluating performance before remapping.")
    pearson, spearman = aligner.correlation(eval_src, eval_system, eval_scores)
    rmse, mae = aligner.error(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}")
    logging.info(f"RMSE: {rmse}, MAE: {mae}")
    results[0] = {"pearson": round(pearson, 3), "spearman": round(spearman, 3), "rmse": round(rmse, 3),
            "mae": round(mae, 3)}
    for iteration in range(1, iterations + 1):
        logging.info(f"Remapping iteration {iteration}.")
        aligner.remap(mono_src, mono_tgt, suffix=suffix + f"-{iteration}", overwrite=False)
        pearson, spearman = aligner.correlation(eval_src, eval_system, eval_scores)
        rmse, mae = aligner.error(eval_src, eval_system, eval_scores)
        logging.info(f"Pearson: {pearson}, Spearman: {spearman}")
        logging.info(f"RMSE: {rmse}, MAE: {mae}")
        results[iteration] = {"pearson": round(pearson, 3), "spearman": round(spearman, 3), "rmse": round(rmse, 3),
                "mae": round(mae, 3)}
    mono_src, mono_tgt = extract_dataset("monolingual-train")
    aligner.train(mono_src, mono_tgt, suffix=suffix + f"-{iterations}", overwrite=False)

    logging.info("Evaluating performance with NMT model.")
    pearson, spearman = aligner.correlation(eval_src, eval_system, eval_scores)
    rmse, mae = aligner.error(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}")
    logging.info(f"RMSE: {rmse}, MAE: {mae}")
    results[f"{iterations} + NMT"] = {"pearson": round(pearson, 3), "spearman": round(spearman, 3),
            "rmse": round(rmse, 3), "mae": round(mae, 3)}

    return results

logging.basicConfig(level=logging.INFO, datefmt="%m-%d %H:%M", format="%(asctime)s %(levelname)-8s %(message)s")
download_datasets()
print(align_tests(alignment="awesome", data="monolingual-align", mapping="UMD", valid="scored"))
print(align_tests(alignment="awesome", data="monolingual-align", mapping="CLP", valid="scored"))
print(align_tests(alignment="awesome", data="monolingual-align", mapping="UMD", valid="scored-wmt"))
print(align_tests(alignment="awesome", data="monolingual-align", mapping="CLP", valid="scored-wmt"))
print(align_tests(alignment="fast", data="monolingual-align", mapping="UMD", valid="scored"))
print(align_tests(alignment="fast", data="monolingual-align", mapping="CLP", valid="scored"))
print(align_tests(alignment="fast", data="monolingual-align", mapping="UMD", valid="scored-wmt"))
print(align_tests(alignment="fast", data="monolingual-align", mapping="CLP", valid="scored-wmt"))
print(align_tests(alignment="awesome", data="parallel-align", mapping="UMD", valid="scored"))
print(align_tests(alignment="awesome", data="parallel-align", mapping="CLP", valid="scored"))
print(align_tests(alignment="awesome", data="parallel-align", mapping="UMD", valid="scored-wmt"))
print(align_tests(alignment="awesome", data="parallel-align", mapping="CLP", valid="scored-wmt"))
print(align_tests(alignment="fast", data="parallel-align", mapping="UMD", valid="scored"))
print(align_tests(alignment="fast", data="parallel-align", mapping="CLP", valid="scored"))
print(align_tests(alignment="fast", data="parallel-align", mapping="UMD", valid="scored-wmt"))
print(align_tests(alignment="fast", data="parallel-align", mapping="CLP", valid="scored-wmt"))
#vecmap_tests()
print(nmt_tests(valid="scored", metric="cosine"))
print(nmt_tests(valid="scored-wmt", metric="cosine"))
print(nmt_tests(valid="scored", metric="wmd"))
print(nmt_tests(valid="scored-wmt", metric="wmd"))
