#!/usr/bin/env python
from metrics.xmoverscore import XMoverNMTLMBertAlignScore
from collections import defaultdict
from tabulate import tabulate
from metrics.utils.dataset import DatasetLoader
import logging

source_lang, target_lang = "de", "en"
iterations = 1

def nmt_tests(metric="cosine"):
    aligner = XMoverNMTLMBertAlignScore(src_lang=source_lang, tgt_lang=target_lang, use_cosine=True if metric == "cosine" else False)
    dataset = DatasetLoader(source_lang, target_lang)
    mono_src, mono_tgt = dataset.load("monolingual-align")
    eval_src, eval_system, eval_scores = dataset.load("scored")
    suffix = f"{source_lang}-{target_lang}-awesome-{metric}-{aligner.mapping}-monolingual-align-{aligner.k}-{aligner.remap_size}-{len(mono_src)}"
    results, index = defaultdict(list), list(range(iterations + 1)) + [f"{iterations} + NMT"]

    logging.info("Evaluating performance before remapping.")
    pearson, spearman = aligner.correlation(eval_src, eval_system, eval_scores)
    rmse, mae = aligner.error(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}, RMSE: {rmse}, MAE: {mae}")
    results["pearson"].append(round(pearson, 3))
    results["spearman"].append(round(spearman, 3))
    results["rmse"].append(round(rmse, 3))
    results["mae"].append(round(mae, 3))

    for iteration in range(1, iterations + 1):
        logging.info(f"Remapping iteration {iteration}.")
        aligner.remap(mono_src, mono_tgt, suffix=suffix + f"-{iteration}", overwrite=False)
        pearson, spearman = aligner.correlation(eval_src, eval_system, eval_scores)
        rmse, mae = aligner.error(eval_src, eval_system, eval_scores)
        logging.info(f"Pearson: {pearson}, Spearman: {spearman}, RMSE: {rmse}, MAE: {mae}")
        results["pearson"].append(round(pearson, 3))
        results["spearman"].append(round(spearman, 3))
        results["rmse"].append(round(rmse, 3))
        results["mae"].append(round(mae, 3))

    mono_src, mono_tgt = dataset.load("monolingual-train")
    aligner.train(mono_src, mono_tgt, suffix=suffix + f"-{iterations}", overwrite=False)

    logging.info("Evaluating performance with NMT model.")
    pearson, spearman = aligner.correlation(eval_src, eval_system, eval_scores)
    rmse, mae = aligner.error(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}, RMSE: {rmse}, MAE: {mae}")
    results["pearson"].append(round(pearson, 3))
    results["spearman"].append(round(spearman, 3))
    results["rmse"].append(round(rmse, 3))
    results["mae"].append(round(mae, 3))

    if target_lang == "en":
        logging.info(f"Evaluating performance with language model.")
        aligner.use_lm = True
        index.append(f"{iterations} + NMT + LM")
        pearson, spearman = aligner.correlation(eval_src, eval_system, eval_scores)
        rmse, mae = aligner.error(eval_src, eval_system, eval_scores)
        logging.info(f"Pearson: {pearson}, Spearman: {spearman}, RMSE: {rmse}, MAE: {mae}")
        results["pearson"].append(round(pearson, 3))
        results["spearman"].append(round(spearman, 3))
        results["rmse"].append(round(rmse, 3))
        results["mae"].append(round(mae, 3))

    return suffix, tabulate(results, headers="keys", showindex=index)

logging.basicConfig(level=logging.INFO, datefmt="%m-%d %H:%M", format="%(asctime)s %(levelname)-8s %(message)s")
print(*nmt_tests(metric="cosine"), sep="\n")
print(*nmt_tests(metric="wmd"), sep="\n")
