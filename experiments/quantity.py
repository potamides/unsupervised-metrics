#!/usr/bin/env python
from metrics.xmoverscore import XMoverNMTLMBertAlignScore
from collections import defaultdict
from tabulate import tabulate
from metrics.utils.dataset import DatasetLoader
import logging

source_lang, target_lang = "de", "en"

def remap_tests(mapping="UMD", amount=40000):
    scorer = XMoverNMTLMBertAlignScore(src_lang=source_lang, tgt_lang=target_lang, mapping=mapping, alignment="fast", use_cosine=True)
    dataset = DatasetLoader(source_lang, target_lang)
    eval_src, eval_system, eval_scores = dataset.load("scored")
    suffix = f"{source_lang}-{target_lang}-fast-cosine-{mapping}-monolingual-align-{scorer.k}-{scorer.remap_size}-{amount}-30-1"
    results = defaultdict(list)

    scorer.remap(*dataset.load("monolingual-align", amount), suffix=suffix, overwrite=False)
    pearson, spearman = scorer.correlation(eval_src, eval_system, eval_scores)
    rmse, mae = scorer.error(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}")
    results["pearson"].append(round(100 * pearson, 2))
    results["spearman"].append(round(100 * spearman, 2))
    results["rmse"].append(round(rmse, 2))
    results["mae"].append(round(mae, 2))

    return suffix, tabulate(results, headers="keys")

logging.basicConfig(level=logging.INFO, datefmt="%m-%d %H:%M", format="%(asctime)s %(levelname)-8s %(message)s")
for amount in (80000, 40000, 20000, 10000, 5000, 2500, 2000):
    print(f"Using {100 * round(2000/amount, 3)}% of aligned sentences for training.")
    print(*remap_tests(mapping="UMD", amount=amount), sep="\n")
    print(*remap_tests(mapping="CLP", amount=amount), sep="\n")
