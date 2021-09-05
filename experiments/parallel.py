#!/usr/bin/env python
from metrics.contrastscore import ContrastScore
from metrics.distilscore import DistilScore
from metrics.utils.dataset import DatasetLoader
from collections import defaultdict
from tabulate import tabulate
import logging

datasets = (("Newstest-2016", "scored", ("de", "en")), ("Newstest-2017", "scored-wmt17", ("de", "en")),
        ("MLQE-PE", "scored-mlqe", ("en", "de")), ("MQM-Newstest-2020", "scored-mqm", ("en", "de")))

def distil_tests(source_lang, target_lang, score_model=ContrastScore, eval_dataset="scored"):
    scorer = score_model(source_language=source_lang, target_language=target_lang, suffix="parallel")
    dataset = DatasetLoader(source_lang, target_lang, hard_limit=500)
    eval_src, eval_system, eval_scores = dataset.load(eval_dataset)
    parallel_src, parallel_tgt = dataset.load("parallel-train")
    results, index = defaultdict(list), ["Baseline", "Fine-tuned model"]

    if score_model == ContrastScore:
        scorer.parallelize = True

    logging.info("Evaluating performance before fine-tuning.")
    pearson, spearman = scorer.correlation(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}")
    results["pearson"].append(round(100 * pearson, 2))
    results["spearman"].append(round(100 * spearman, 2))

    scorer.train(parallel_src, parallel_tgt, aligned=True, overwrite=False)

    logging.info(f"Evaluating performance after fine-tuning.")
    pearson, spearman = scorer.correlation(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}")
    results["pearson"].append(round(100 * pearson, 2))
    results["spearman"].append(round(100 * spearman, 2))

    return tabulate(results, headers="keys", showindex=index)

logging.basicConfig(level=logging.INFO, datefmt="%m-%d %H:%M", format="%(asctime)s %(levelname)-8s %(message)s")
for dataset, identifier, pair in datasets:
    print(f"Evaluating {'-'.join(pair)} language direction on {dataset}")
    print("Results using contrastive learning:", distil_tests(*pair, score_model=ContrastScore, eval_dataset=identifier), sep="\n")
    print("Results using knowledge distillation:", distil_tests(*pair, score_model=DistilScore, eval_dataset=identifier), sep="\n")
