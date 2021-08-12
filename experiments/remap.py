#!/usr/bin/env python
from metrics.xmoverscore import XMoverBertAlignScore
from collections import defaultdict
from tabulate import tabulate
from metrics.utils.dataset import DatasetLoader
import logging

source_lang, target_lang = "de", "en"
iterations = 5

def remap_tests(alignment="awesome", mapping="UMD", data="monolingual-align", metric="cosine"):
    scorer = XMoverBertAlignScore(alignment=alignment, mapping=mapping, use_cosine=True if metric == "cosine" else False)
    dataset = DatasetLoader(source_lang, target_lang)
    parallel_src, parallel_tgt = dataset.load("parallel")
    mono_src, mono_tgt = dataset.load(data)
    eval_src, eval_system, eval_scores = dataset.load("scored")
    suffix = f"{source_lang}-{target_lang}-{alignment}-{metric}-{mapping}-{data}-{scorer.k}-{scorer.remap_size}-{len(mono_src)}"
    results = defaultdict(list)

    logging.info("Evaluating performance before remapping.")
    pearson, spearman = scorer.correlation(eval_src, eval_system, eval_scores)
    precision = scorer.precision(parallel_src, parallel_tgt)
    rmse, mae = scorer.error(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}, Precision @ 1: {precision}, RMSE: {rmse}, MAE: {mae}")
    results["pearson"].append(round(100 * pearson, 2))
    results["spearman"].append(round(100 * spearman, 2))
    results["precision"].append(round(100 * precision, 2))
    results["rmse"].append(round(rmse, 2))
    results["mae"].append(round(mae, 2))

    for iteration in range(1, iterations + 1):
        logging.info(f"Remapping iteration {iteration}.")
        scorer.remap(mono_src, mono_tgt, suffix=suffix + f"-{iteration}", overwrite=False)
        pearson, spearman = scorer.correlation(eval_src, eval_system, eval_scores)
        precision = scorer.precision(parallel_src, parallel_tgt)
        rmse, mae = scorer.error(eval_src, eval_system, eval_scores)
        logging.info(f"Pearson: {pearson}, Spearman: {spearman}, Precision @ 1: {precision}, RMSE: {rmse}, MAE: {mae}")
        results["pearson"].append(round(100 * pearson, 2))
        results["spearman"].append(round(100 * spearman, 2))
        results["precision"].append(round(100 * precision, 2))
        results["rmse"].append(round(rmse, 2))
        results["mae"].append(round(mae, 2))

    return suffix, tabulate(results, headers="keys", showindex=True)

logging.basicConfig(level=logging.INFO, datefmt="%m-%d %H:%M", format="%(asctime)s %(levelname)-8s %(message)s")
print(*remap_tests(alignment="awesome", data="monolingual-align", mapping="UMD"), sep="\n")
print(*remap_tests(alignment="awesome", data="monolingual-align", mapping="CLP"), sep="\n")
print(*remap_tests(alignment="awesome-remap", data="monolingual-align", mapping="UMD"), sep="\n")
print(*remap_tests(alignment="awesome-remap", data="monolingual-align", mapping="CLP"), sep="\n")
print(*remap_tests(alignment="fast", data="monolingual-align", mapping="UMD"), sep="\n")
print(*remap_tests(alignment="fast", data="monolingual-align", mapping="CLP"), sep="\n")
print(*remap_tests(alignment="awesome", data="parallel-align", mapping="UMD"), sep="\n")
print(*remap_tests(alignment="awesome", data="parallel-align", mapping="CLP"), sep="\n")
print(*remap_tests(alignment="fast", data="parallel-align", mapping="UMD"), sep="\n")
print(*remap_tests(alignment="fast", data="parallel-align", mapping="CLP"), sep="\n")
