#!/usr/bin/env python
from metrics.xmoverscore import XMoverScore
from collections import defaultdict
from tabulate import tabulate
from metrics.utils.dataset import DatasetLoader
import logging

source_lang, target_lang = "de", "en"

def align_tests(mapping="UMD"):
    aligner = XMoverScore(mapping=mapping)
    dataset = DatasetLoader(source_lang, target_lang)
    eval_src, eval_system, eval_scores = dataset.load("scored")
    results, index = defaultdict(list), ["Default", mapping]

    logging.info("Evaluating performance before remapping.")
    pearson, spearman = aligner.correlation(eval_src, eval_system, eval_scores)
    rmse, mae = aligner.error(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}, RMSE: {rmse}, MAE: {mae}")
    results["pearson"].append(round(100 * pearson, 2))
    results["spearman"].append(round(100 * spearman, 2))
    results["rmse"].append(round(rmse, 2))
    results["mae"].append(round(mae, 2))

    logging.info(f"Evaluating performance after remapping.")
    aligner.remap(source_lang, target_lang)
    pearson, spearman = aligner.correlation(eval_src, eval_system, eval_scores)
    rmse, mae = aligner.error(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}, RMSE: {rmse}, MAE: {mae}")
    results["pearson"].append(round(100 * pearson, 2))
    results["spearman"].append(round(100 * spearman, 2))
    results["rmse"].append(round(rmse, 2))
    results["mae"].append(round(mae, 2))

    if target_lang == "en":
        logging.info(f"Evaluating performance with language model.")
        aligner.use_lm = True
        index.append(f"{mapping} + LM")
        pearson, spearman = aligner.correlation(eval_src, eval_system, eval_scores)
        rmse, mae = aligner.error(eval_src, eval_system, eval_scores)
        logging.info(f"Pearson: {pearson}, Spearman: {spearman}, RMSE: {rmse}, MAE: {mae}")
        results["pearson"].append(round(100 * pearson, 2))
        results["spearman"].append(round(100 * spearman, 2))
        results["rmse"].append(round(rmse, 2))
        results["mae"].append(round(mae, 2))

    return tabulate(results, headers="keys", showindex=index)

logging.basicConfig(level=logging.INFO, datefmt="%m-%d %H:%M", format="%(asctime)s %(levelname)-8s %(message)s")
print(align_tests(mapping="UMD"))
print(align_tests(mapping="CLP"))
