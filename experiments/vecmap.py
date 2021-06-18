#!/usr/bin/env python
from metrics.xmoverscore import XMoverVecMapAlignScore
from collections import defaultdict
from tabulate import tabulate
from metrics.utils.dataset import DatasetLoader
import logging

source_lang, target_lang = "de", "en"

def vecmap_tests():
    aligner = XMoverVecMapAlignScore(src_lang=source_lang, tgt_lang=target_lang)
    dataset = DatasetLoader(source_lang, target_lang)
    parallel_src, parallel_tgt = dataset.load("parallel")
    eval_src, eval_system, eval_scores = dataset.load("scored")
    results = defaultdict(list)

    pearson, spearman = aligner.correlation(eval_src, eval_system, eval_scores)
    precision = aligner.precision(parallel_src, parallel_tgt)
    rmse, mae = aligner.error(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}, Precision @ 1: {precision}, RMSE: {rmse}, MAE: {mae}")
    results["pearson"].append(round(100 * pearson, 2))
    results["spearman"].append(round(100 * spearman, 2))
    results["precision"].append(round(100 * precision, 2))
    results["rmse"].append(round(100 * rmse, 2))
    results["mae"].append(round(100 * mae, 2))

    return f"{source_lang}-{target_lang}-vecmap", tabulate(results, headers="keys")

logging.basicConfig(level=logging.INFO, datefmt="%m-%d %H:%M", format="%(asctime)s %(levelname)-8s %(message)s")
print(*vecmap_tests(), sep="\n")
