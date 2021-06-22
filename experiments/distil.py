#!/usr/bin/env python
from metrics.distilscore import DistilScore
from collections import defaultdict
from tabulate import tabulate
from metrics.utils.dataset import DatasetLoader
import logging

source_lang, target_lang = "de", "en"
iterations = 5

def distil_tests():
    scorer = DistilScore(source_language=source_lang, target_language=target_lang, suffix="1")
    dataset = DatasetLoader(source_lang, target_lang)
    eval_src, eval_system, eval_scores = dataset.load("scored")
    results, index = defaultdict(list), list(range(iterations + 1))

    logging.info("Evaluating performance before distilling.")
    pearson, spearman = scorer.correlation(eval_src, eval_system, eval_scores)
    rmse, mae = scorer.error(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}, RMSE: {rmse}, MAE: {mae}")
    results["pearson"].append(round(100 * pearson, 2))
    results["spearman"].append(round(100 * spearman, 2))
    results["rmse"].append(round(rmse, 2))
    results["mae"].append(round(mae, 2))

    parallel_src, parallel_tgt = dataset.load("parallel")
    mono_src, mono_tgt = dataset.load("monolingual-train")

    for iteration in range(1, iterations + 1):
        logging.info(f"Training iteration {iteration}.")
        scorer.suffix = str(iteration)
        scorer.train(mono_src, mono_tgt, dev_source_sents=parallel_src, dev_target_sents=parallel_tgt, overwrite=False)
        pearson, spearman = scorer.correlation(eval_src, eval_system, eval_scores)
        rmse, mae = scorer.error(eval_src, eval_system, eval_scores)
        logging.info(f"Pearson: {pearson}, Spearman: {spearman}, RMSE: {rmse}, MAE: {mae}")
        results["pearson"].append(round(100 * pearson, 2))
        results["spearman"].append(round(100 * spearman, 2))
        results["rmse"].append(round(rmse, 2))
        results["mae"].append(round(mae, 2))

    return tabulate(results, headers="keys", showindex=index)

logging.basicConfig(level=logging.INFO, datefmt="%m-%d %H:%M", format="%(asctime)s %(levelname)-8s %(message)s")
print(distil_tests())
