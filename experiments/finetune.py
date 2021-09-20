#!/usr/bin/env python
from metrics.xmoverscore import XMoverNMTLMBertAlignScore
from metrics.contrastscore import ContrastScore
from metrics.utils.dataset import DatasetLoader
from datasets import load_dataset
from collections import defaultdict
from tabulate import tabulate
from numpy import corrcoef, argsort
import logging

mlqe = [("en", "de"), ("en", "zh"), ("ru", "en"), ("ro", "en"), ("et", "en"), ("ne", "en"), ("si", "en")]
lm_model = {"en": "gpt2", "de": "dbmdz/german-gpt2", "zh": "uer/gpt2-chinese-cluecorpussmall"}

remap_iterations = 1
nmt_iterations = 1
contrast_iterations = 6

def correlation(model_scores, ref_scores):
    ref_ranks, ranks = argsort(ref_scores).argsort(), argsort(model_scores).argsort()
    return corrcoef(ref_scores, model_scores)[0,1], corrcoef(ref_ranks, ranks)[0,1]

def self_learning_tests(source_lang, target_lang, max_len=30):
    xmover = XMoverNMTLMBertAlignScore(src_lang=source_lang, tgt_lang=target_lang, lm_weights=[1, 0.1],
            nmt_weights=[0.5, 0.4], use_lm=True, lm_model_name=lm_model[target_lang])
    contrast = ContrastScore(source_language=source_lang, target_language=target_lang, parallelize=True)
    dataset = DatasetLoader(source_lang, target_lang, max_monolingual_sent_len=max_len)
    mono_src, mono_tgt = dataset.load("monolingual-align")
    para_src, para_tgt = dataset.load("wikimatrix")
    train_src, train_tgt = dataset.load("monolingual-train")
    eval_src, eval_system, eval_scores = dataset.load("scored-mlqe")
    suffix = f"{source_lang}-{target_lang}-awesome-wmd-{xmover.mapping}-monolingual-align-{xmover.k}-{xmover.remap_size}-{40000}-{max_len}"
    results, index = defaultdict(list), [f"XMoverScore ({max_len} tokens)", f"Fine-tuned XMoverScore ({max_len} tokens)",
            f"ContrastScore ({max_len} tokens)", f"Fine-tuned ContrastScore ({max_len} tokens)",
            f"XMoverScore + ContrastScore ({max_len} tokens)"]

    logging.info("Evaluating XMoverScore")
    for iteration in range(1, remap_iterations + 1):
        logging.info(f"Remapping iteration {iteration}.")
        xmover.remap(mono_src, mono_tgt, suffix=suffix + f"-{iteration}", overwrite=False)
    for iteration in range(nmt_iterations):
        logging.info(f"NMT training iteration {iteration}.")
        xmover.train(train_src, train_tgt, suffix=suffix+f"-{remap_iterations}", iteration=iteration, overwrite=False, k=1)

    pearson, spearman = xmover.correlation(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}")
    results["pearson"].append(round(100 * pearson, 2))
    results["spearman"].append(round(100 * spearman, 2))

    logging.info(f"NMT training on parallel data.")
    xmover.train(para_src, para_tgt, suffix=suffix+"-finetuned", iteration=iteration, aligned=True,
            finetune=True, overwrite=False, k=1)

    pearson, spearman = xmover.correlation(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}")
    results["pearson"].append(round(100 * pearson, 2))
    results["spearman"].append(round(100 * spearman, 2))

    logging.info("Evaluating ContrastScore")
    for iteration in range(1, contrast_iterations + 1):
        logging.info(f"Contrastive Learning iteration {iteration}.")
        contrast.suffix = f"{max_len}-{iteration}"
        contrast.train(train_src, train_tgt, overwrite=False)

    pearson, spearman = contrast.correlation(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}")
    results["pearson"].append(round(100 * pearson, 2))
    results["spearman"].append(round(100 * spearman, 2))

    logging.info(f"Contrastive Learning on parallel data.")
    contrast.suffix = f"{max_len}-finetuned"
    contrast.train(para_src, para_tgt, aligned=True, finetune=True, overwrite=False)

    pearson, spearman = contrast.correlation(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}")
    results["pearson"].append(round(100 * pearson, 2))
    results["spearman"].append(round(100 * spearman, 2))

    logging.info("Evaluating XMoverScore + ContrastScore")
    wmd_scores, contrast_scores = xmover.score(eval_src, eval_system), contrast.score(eval_src, eval_system)
    pearson, spearman = correlation([0.6 * x + 0.4 * y for x, y in zip(wmd_scores, contrast_scores)], eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}")
    results["pearson"].append(round(100 * pearson, 2))
    results["spearman"].append(round(100 * spearman, 2))

    return tabulate(results, headers="keys", showindex=index)

logging.basicConfig(level=logging.INFO, datefmt="%m-%d %H:%M", format="%(asctime)s %(levelname)-8s %(message)s")
for source_lang, target_lang in mlqe:
    print(f"Evaluating {source_lang}-{target_lang} language direction on MLQE-PE")
    print(self_learning_tests(source_lang, target_lang, max_len=30))
    print(self_learning_tests(source_lang, target_lang, max_len=50))
