#!/usr/bin/env python
from metrics.xmoverscore import XMoverBertAlignScore
from collections import defaultdict
from tabulate import tabulate
from metrics.utils.dataset import DatasetLoader
import logging

source_lang, target_lang = "de", "en"
iterations = 5

def align_tests(alignment="awesome", mapping="UMD", data="monolingual-align", metric="cosine"):
    aligner = XMoverBertAlignScore(alignment=alignment, mapping=mapping, use_cosine=True if metric == "cosine" else False)
    dataset = DatasetLoader(source_lang, target_lang)
    parallel_src, parallel_tgt = dataset.load("parallel")
    mono_src, mono_tgt = dataset.load(data)
    eval_src, eval_system, eval_scores = dataset.load("scored")
    suffix = f"{source_lang}-{target_lang}-{alignment}-{metric}-{mapping}-{data}-{aligner.k}-{aligner.remap_size}-{len(mono_src)}"
    results = defaultdict(list)

    logging.info("Evaluating performance before remapping.")
    pearson, spearman = aligner.correlation(eval_src, eval_system, eval_scores)
    precision = aligner.precision(parallel_src, parallel_tgt)
    rmse, mae = aligner.error(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}, Precision @ 1: {precision}, RMSE: {rmse}, MAE: {mae}")
    results["pearson"].append(round(pearson, 3))
    results["spearman"].append(round(spearman, 3))
    results["precision"].append(round(precision, 3))
    results["rmse"].append(round(rmse, 3))
    results["mae"].append(round(mae, 3))

    for iteration in range(1, iterations + 1):
        logging.info(f"Remapping iteration {iteration}.")
        aligner.remap(mono_src, mono_tgt, suffix=suffix + f"-{iteration}", overwrite=False)
        pearson, spearman = aligner.correlation(eval_src, eval_system, eval_scores)
        precision = aligner.precision(parallel_src, parallel_tgt)
        rmse, mae = aligner.error(eval_src, eval_system, eval_scores)
        logging.info(f"Pearson: {pearson}, Spearman: {spearman}, Precision @ 1: {precision}, RMSE: {rmse}, MAE: {mae}")
        results["pearson"].append(round(pearson, 3))
        results["spearman"].append(round(spearman, 3))
        results["precision"].append(round(precision, 3))
        results["rmse"].append(round(rmse, 3))
        results["mae"].append(round(mae, 3))

    return suffix, tabulate(results, headers="keys", showindex=True)

logging.basicConfig(level=logging.INFO, datefmt="%m-%d %H:%M", format="%(asctime)s %(levelname)-8s %(message)s")
print(*align_tests(alignment="awesome", data="monolingual-align", mapping="UMD"), sep="\n")
print(*align_tests(alignment="awesome", data="monolingual-align", mapping="CLP"), sep="\n")
print(*align_tests(alignment="fast", data="monolingual-align", mapping="UMD"), sep="\n")
print(*align_tests(alignment="fast", data="monolingual-align", mapping="CLP"), sep="\n")
print(*align_tests(alignment="awesome", data="parallel-align", mapping="UMD"), sep="\n")
print(*align_tests(alignment="awesome", data="parallel-align", mapping="CLP"), sep="\n")
print(*align_tests(alignment="fast", data="parallel-align", mapping="UMD"), sep="\n")
print(*align_tests(alignment="fast", data="parallel-align", mapping="CLP"), sep="\n")
