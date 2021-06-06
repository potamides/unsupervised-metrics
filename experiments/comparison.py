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
    results = defaultdict(list)

    logging.info("Evaluating performance before remapping.")
    pearson, spearman = aligner.correlation(eval_src, eval_system, eval_scores)
    rmse, mae = aligner.error(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}, RMSE: {rmse}, MAE: {mae}")
    results["pearson"].append(round(pearson, 3))
    results["spearman"].append(round(spearman, 3))
    results["rmse"].append(round(rmse, 3))
    results["mae"].append(round(mae, 3))

    logging.info(f"Evaluating performance after remapping.")
    aligner.remap(source_lang, target_lang)
    pearson, spearman = aligner.correlation(eval_src, eval_system, eval_scores)
    rmse, mae = aligner.error(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}, RMSE: {rmse}, MAE: {mae}")
    results["pearson"].append(round(pearson, 3))
    results["spearman"].append(round(spearman, 3))
    results["rmse"].append(round(rmse, 3))
    results["mae"].append(round(mae, 3))

    logging.info(f"Evaluating performance with language model.")
    aligner.use_lm = True
    pearson, spearman = aligner.correlation(eval_src, eval_system, eval_scores)
    rmse, mae = aligner.error(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}, RMSE: {rmse}, MAE: {mae}")
    results["pearson"].append(round(pearson, 3))
    results["spearman"].append(round(spearman, 3))
    results["rmse"].append(round(rmse, 3))
    results["mae"].append(round(mae, 3))

    return tabulate(results, headers="keys", showindex=["Default", mapping, f"{mapping} + LM"])

logging.basicConfig(level=logging.INFO, datefmt="%m-%d %H:%M", format="%(asctime)s %(levelname)-8s %(message)s")
print(align_tests(mapping="UMD"))
print(align_tests(mapping="CLP"))
