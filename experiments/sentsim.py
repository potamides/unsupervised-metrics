#!/usr/bin/env python
from collections import defaultdict
import logging

from tabulate import tabulate

from metrics.sentsim import SentSim
from metrics.utils.dataset import DatasetLoader

mlqe = [
    ("en", "de"),
    ("en", "zh"),
    ("ru", "en"),
    ("ro", "en"),
    ("et", "en"),
    ("ne", "en"),
    ("si", "en"),
]
newstest2017 = [
    ("cs", "en"),
    ("de", "en"),
    ("fi", "en"),
    ("lv", "en"),
    ("ru", "en"),
    ("tr", "en"),
    ("zh", "en"),
    ("en", "zh"),
    ("en", "ru"),
]

# Try to reproduce results of Sentsim on both WMT-17 and WMT-20. The scores for
# WMT-20 are computed for both human-annotated scores and MT model scores.
def sentsim_reproduce(
    source_lang,
    target_lang,
    dataset_name,
    word_metric="BERTScore",
    use_mlqe_model_scores=False,
):
    scorer = SentSim(use_wmd=word_metric == "WMD")
    dataset = DatasetLoader(source_lang, target_lang)
    eval_src, eval_system, eval_scores = dataset.load(
        dataset_name, use_mlqe_model_scores=use_mlqe_model_scores
    )
    results, index = defaultdict(list), [f"SentSim ({word_metric})"]

    pearson, spearman = scorer.correlation(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}")
    results["pearson"].append(round(100 * pearson, 1))
    results["spearman"].append(round(100 * spearman, 1))

    return tabulate(results, headers="keys", showindex=index)


logging.basicConfig(
    level=logging.INFO,
    datefmt="%m-%d %H:%M",
    format="%(asctime)s %(levelname)-8s %(message)s",
)
datasets = (
    ("Newstest-2017", "scored-wmt17", newstest2017),
    ("MLQE-PE (Human)", "scored-mlqe", mlqe),
    ("MLQE-PE (Model)", "scored-mlqe", mlqe),
)
for dataset, identifier, pairs in datasets:
    for source_lang, target_lang in pairs:
        print(f"Evaluating {source_lang}-{target_lang} language direction on {dataset}")
        print(
            sentsim_reproduce(
                source_lang, target_lang, identifier, "BERTScore", "Mode" in dataset
            )
        )
        print(
            sentsim_reproduce(
                source_lang, target_lang, identifier, "WMD", "Mode" in dataset
            )
        )
