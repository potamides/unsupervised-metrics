#!/usr/bin/env python
from metrics.xmoverscore import XMoverNMTLMBertAlignScore
from collections import defaultdict
from tabulate import tabulate
from numpy import linspace
from metrics.utils.dataset import DatasetLoader
import logging

source_lang, target_lang = "de", "en"
iterations = 1

def nmt_tests(metric="wmd", max_len=50):
    assert target_lang == "en", "Target language has to be English for LM to work"
    results = defaultdict(list)
    aligner = XMoverNMTLMBertAlignScore(src_lang=source_lang, tgt_lang=target_lang, use_lm=True, use_cosine=metric=="cosine")
    dataset = DatasetLoader(source_lang, target_lang, max_monolingual_sent_len=max_len)
    mono_src, mono_tgt = dataset.load("monolingual-align")
    eval_src, eval_system, eval_scores = dataset.load("scored")
    suffix = f"{source_lang}-{target_lang}-awesome-{metric}-{aligner.mapping}-monolingual-align-{aligner.k}-{aligner.remap_size}-{len(mono_src)}-{max_len}"

    for iteration in range(1, iterations + 1):
        logging.info(f"Remapping iteration {iteration}.")
        aligner.remap(mono_src, mono_tgt, suffix=suffix + f"-{iteration}", overwrite=False)

    logging.info("Training NMT system.")
    train_src, train_tgt = dataset.load("monolingual-train")
    aligner.train(train_src, train_tgt, suffix=suffix+f"-{iterations}", overwrite=False, k=5 if metric=="cosine" else 1)


    logging.info(f"Evaluating performance with NMT and language model.")
    for lm_weight in linspace(0, 1, 11):
        for nmt_weight in linspace(0, 1, 11):
            if lm_weight + nmt_weight <= 1:
                aligner.nmt_weights = [1 - lm_weight - nmt_weight, nmt_weight]
                aligner.lm_weights = [1, lm_weight]
                pearson, _ = aligner.correlation(eval_src, eval_system, eval_scores)
                logging.info(f"NMT: {nmt_weight}, LM: {lm_weight}, Pearson: {pearson}")
                results[round(lm_weight, 1)].append(round(100 * pearson, 2))
            else:
                results[round(lm_weight, 1)].append("-")

    return suffix, tabulate(results, headers="keys", showindex=linspace(0, 1, 11))

logging.basicConfig(level=logging.INFO, datefmt="%m-%d %H:%M", format="%(asctime)s %(levelname)-8s %(message)s")
print(*nmt_tests(), sep="\n")
