#!/usr/bin/env python
from metrics.xmoverscore import XMoverNMTLMBertAlignScore
from metrics.contrastscore import ContrastScore
from collections import defaultdict
from tabulate import tabulate
from numpy import linspace
from numpy import corrcoef, argsort
from torch.nn.functional import mse_loss, l1_loss
from torch import FloatTensor
from metrics.utils.dataset import DatasetLoader
import logging

source_lang, target_lang = "de", "en"
remap_iterations = 1
nmt_iterations = 1
contrast_iterations = 2

def error(model_scores, ref_scores):
    rmse = mse_loss(FloatTensor(ref_scores), FloatTensor(model_scores)).sqrt().item()
    mae = l1_loss(FloatTensor(ref_scores), FloatTensor(model_scores)).item()
    return rmse, mae

def correlation(model_scores, ref_scores):
    ref_ranks, ranks = argsort(ref_scores).argsort(), argsort(model_scores).argsort()
    return corrcoef(ref_scores, model_scores)[0,1], corrcoef(ref_ranks, ranks)[0,1]

def combine_tests(max_len=30):
    xmover = XMoverNMTLMBertAlignScore(src_lang=source_lang, tgt_lang=target_lang, nmt_weights=[0.5, 0.5])
    contrast = ContrastScore(source_language=source_lang, target_language=target_lang, parallelize=True)
    dataset = DatasetLoader(source_lang, target_lang, max_monolingual_sent_len=max_len)
    mono_src, mono_tgt = dataset.load("monolingual-align")
    eval_src, eval_system, eval_scores = dataset.load("scored")
    if not {'train_src', 'train_tgt'}.issubset(globals()):
        global train_src, train_tgt
        train_src, train_tgt = dataset.load("monolingual-train")
    suffix = f"{source_lang}-{target_lang}-awesome-wmd-{xmover.mapping}-monolingual-align-{xmover.k}-{xmover.remap_size}-{40000}-{max_len}"
    results, index = defaultdict(list), [f"{round(weight, 2)}-{round(1-weight, 2)}" for weight in linspace(1, 0, 11)]

    logging.info("Preparing XMoverScore")
    for iteration in range(1, remap_iterations + 1):
        logging.info(f"Remapping iteration {iteration}.")
        xmover.remap(mono_src, mono_tgt, suffix=suffix + f"-{iteration}", overwrite=False)
    for iteration in range(nmt_iterations):
        logging.info(f"NMT training iteration {iteration}.")
        xmover.train(train_src, train_tgt, suffix=suffix+f"-{remap_iterations}", iteration=iteration, overwrite=False, k=1)

    logging.info("Preparing ContrastScore")
    for iteration in range(1, contrast_iterations + 1):
        logging.info(f"Contrastive Learning iteration {iteration}.")
        contrast.suffix = f"{max_len}-{iteration}"
        contrast.train(mono_src, mono_tgt, overwrite=False)

    wmd_scores = xmover.score(eval_src, eval_system)
    contrast_scores = contrast.score(eval_src, eval_system)

    for weight in linspace(1, 0, 11):
        pearson, spearman = correlation([weight * x + (1 - weight) * y for x, y in zip(wmd_scores, contrast_scores)], eval_scores)
        rmse, mae = error([weight * x + (1 - weight) * y for x, y in zip(wmd_scores, contrast_scores)], eval_scores)
        results["pearson"].append(round(100 * pearson, 2))
        results["spearman"].append(round(100 * spearman, 2))
        results["rmse"].append(round(rmse, 2))
        results["mae"].append(round(mae, 2))

    return tabulate(results, headers="keys", showindex=index)

logging.basicConfig(level=logging.INFO, datefmt="%m-%d %H:%M", format="%(asctime)s %(levelname)-8s %(message)s")
print("Maximum Sentence Length: 30", combine_tests(), sep="\n")
print("Maximum Sentence Length: 50", combine_tests(max_len=50), sep="\n")
