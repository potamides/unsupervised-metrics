#!/usr/bin/env python
from metrics.csescore import CSEScore
from collections import defaultdict
from tabulate import tabulate
from metrics.utils.dataset import DatasetLoader
import logging

source_lang, target_lang = "de", "en"

def cse_tests():
    scorer = CSEScore(source_language=source_lang, target_language=target_lang, suffix="mono")
    dataset = DatasetLoader(source_lang, target_lang)
    eval_src, eval_system, eval_scores = dataset.load("scored")
    results, index = defaultdict(list), ["Before Training", "Monolingual Training", "Mixed Training"]

    logging.info("Evaluating performance before training.")
    pearson, spearman = scorer.correlation(eval_src, eval_system, eval_scores)
    rmse, mae = scorer.error(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}, RMSE: {rmse}, MAE: {mae}")
    results["pearson"].append(round(100 * pearson, 2))
    results["spearman"].append(round(100 * spearman, 2))
    results["rmse"].append(round(rmse, 2))
    results["mae"].append(round(mae, 2))

    mono_src, mono_tgt = dataset.load("monolingual-train")

    logging.info("Training with monolingual data only.")
    scorer.train(mono_src, mono_tgt, overwrite=False)
    pearson, spearman = scorer.correlation(eval_src, eval_system, eval_scores)
    rmse, mae = scorer.error(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}, RMSE: {rmse}, MAE: {mae}")
    results["pearson"].append(round(100 * pearson, 2))
    results["spearman"].append(round(100 * spearman, 2))
    results["rmse"].append(round(rmse, 2))
    results["mae"].append(round(mae, 2))

    logging.info("Training with both monolingual and pseudo parallel data.")
    scorer.suffix = "mixed"
    scorer.train(mono_src, mono_tgt, mine_size=10000, overwrite=False)
    pearson, spearman = scorer.correlation(eval_src, eval_system, eval_scores)
    rmse, mae = scorer.error(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}, RMSE: {rmse}, MAE: {mae}")
    results["pearson"].append(round(100 * pearson, 2))
    results["spearman"].append(round(100 * spearman, 2))
    results["rmse"].append(round(rmse, 2))
    results["mae"].append(round(mae, 2))

    return tabulate(results, headers="keys", showindex=index)

logging.basicConfig(level=logging.INFO, datefmt="%m-%d %H:%M", format="%(asctime)s %(levelname)-8s %(message)s")
print(cse_tests())
