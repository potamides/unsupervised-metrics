#!/usr/bin/env python
from metrics.xmoverscore import XMoverScore
from metrics.sentsim import SentSim
from collections import defaultdict
from tabulate import tabulate
from metrics.utils.dataset import DatasetLoader
import logging

source_lang, target_lang = "de", "en"

def xmoverscore_tests(mapping="UMD"):
    scorer = XMoverScore(mapping=mapping)
    dataset = DatasetLoader(source_lang, target_lang)
    eval_src, eval_system, eval_scores = dataset.load("scored")
    results, index = defaultdict(list), ["Default", mapping]

    logging.info("Evaluating performance before remapping.")
    pearson, spearman = scorer.correlation(eval_src, eval_system, eval_scores)
    rmse, mae = scorer.error(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}, RMSE: {rmse}, MAE: {mae}")
    results["pearson"].append(round(100 * pearson, 2))
    results["spearman"].append(round(100 * spearman, 2))
    results["rmse"].append(round(rmse, 2))
    results["mae"].append(round(mae, 2))

    logging.info(f"Evaluating performance after remapping.")
    scorer.remap(source_lang, target_lang)
    pearson, spearman = scorer.correlation(eval_src, eval_system, eval_scores)
    rmse, mae = scorer.error(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}, RMSE: {rmse}, MAE: {mae}")
    results["pearson"].append(round(100 * pearson, 2))
    results["spearman"].append(round(100 * spearman, 2))
    results["rmse"].append(round(rmse, 2))
    results["mae"].append(round(mae, 2))

    if target_lang == "en":
        logging.info(f"Evaluating performance with language model.")
        scorer.use_lm = True
        index.append(f"{mapping} + LM")
        pearson, spearman = scorer.correlation(eval_src, eval_system, eval_scores)
        rmse, mae = scorer.error(eval_src, eval_system, eval_scores)
        logging.info(f"Pearson: {pearson}, Spearman: {spearman}, RMSE: {rmse}, MAE: {mae}")
        results["pearson"].append(round(100 * pearson, 2))
        results["spearman"].append(round(100 * spearman, 2))
        results["rmse"].append(round(rmse, 2))
        results["mae"].append(round(mae, 2))

    return tabulate(results, headers="keys", showindex=index)

def sentsim_tests(word_metric="BERTScore"):
    scorer = SentSim(use_wmd=word_metric=="WMD")
    dataset = DatasetLoader(source_lang, target_lang)
    eval_src, eval_system, eval_scores = dataset.load("scored")
    results, index = defaultdict(list), [f"SentSim ({word_metric})"]

    pearson, spearman = scorer.correlation(eval_src, eval_system, eval_scores)
    rmse, mae = scorer.error(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}, RMSE: {rmse}, MAE: {mae}")
    results["pearson"].append(round(100 * pearson, 2))
    results["spearman"].append(round(100 * spearman, 2))
    results["rmse"].append(round(rmse, 2))
    results["mae"].append(round(mae, 2))

    return tabulate(results, headers="keys", showindex=index)

logging.basicConfig(level=logging.INFO, datefmt="%m-%d %H:%M", format="%(asctime)s %(levelname)-8s %(message)s")
print(xmoverscore_tests(mapping="UMD"))
print(xmoverscore_tests(mapping="CLP"))
print(sentsim_tests(word_metric="BERTScore"))
print(sentsim_tests(word_metric="WMD"))
