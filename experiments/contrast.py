#!/usr/bin/env python
from metrics.contrastscore import ContrastScore
from collections import defaultdict
from tabulate import tabulate
from metrics.utils.dataset import DatasetLoader
import logging

source_lang, target_lang = "de", "en"
iterations = 5

def contrastive_tests():
    scorer = ContrastScore(source_language=source_lang, target_language=target_lang)
    dataset = DatasetLoader(source_lang, target_lang)
    eval_src, eval_system, eval_scores = dataset.load("scored")
    results, index = defaultdict(list), list(range(iterations + 1))

    logging.info("Evaluating performance before training.")
    pearson, spearman = scorer.correlation(eval_src, eval_system, eval_scores)
    rmse, mae = scorer.error(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}, RMSE: {rmse}, MAE: {mae}")
    results["pearson"].append(round(100 * pearson, 2))
    results["spearman"].append(round(100 * spearman, 2))
    results["rmse"].append(round(rmse, 2))
    results["mae"].append(round(mae, 2))

    mono_src, mono_tgt = dataset.load("monolingual-train")

    for iteration in range(1, iterations + 1):
        logging.info(f"Training iteration {iteration}.")
        scorer.suffix = str(iteration)
        scorer.train(mono_src, mono_tgt, overwrite=False)
        pearson, spearman = scorer.correlation(eval_src, eval_system, eval_scores)
        rmse, mae = scorer.error(eval_src, eval_system, eval_scores)
        logging.info(f"Pearson: {pearson}, Spearman: {spearman}, RMSE: {rmse}, MAE: {mae}")
        results["pearson"].append(round(100 * pearson, 2))
        results["spearman"].append(round(100 * spearman, 2))
        results["rmse"].append(round(rmse, 2))
        results["mae"].append(round(mae, 2))

    return tabulate(results, headers="keys", showindex=index)

logging.basicConfig(level=logging.INFO, datefmt="%m-%d %H:%M", format="%(asctime)s %(levelname)-8s %(message)s")
print(contrastive_tests())
