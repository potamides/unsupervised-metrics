#!/usr/bin/env python
from metrics.xmoverscore import XMoverNMTLMBertAlignScore
from metrics.contrastscore import ContrastScore
from metrics.sentsim import SentSim
from metrics.utils.dataset import DatasetLoader
from collections import defaultdict
from tabulate import tabulate
from numpy import corrcoef, argsort
import logging

mlqe = [("en", "de"), ("en", "zh"), ("ru", "en"), ("ro", "en"), ("et", "en"), ("ne", "en"), ("si", "en")]
lm_model = {"en": "gpt2", "ru": "sberbank-ai/rugpt3small_based_on_gpt2", "de": "dbmdz/german-gpt2",
        "zh": "uer/gpt2-chinese-cluecorpussmall"}

remap_iterations = 1
nmt_iterations = 1
contrast_iterations = 6

def correlation(model_scores, ref_scores):
    ref_ranks, ranks = argsort(ref_scores).argsort(), argsort(model_scores).argsort()
    return corrcoef(ref_scores, model_scores)[0,1], corrcoef(ref_ranks, ranks)[0,1]

def xmover_contrast_combine(source_lang, target_lang, max_len=30):
    xmover = XMoverNMTLMBertAlignScore(src_lang=source_lang, tgt_lang=target_lang, lm_weights=[1, 0.1],
        nmt_weights=[0.5, 0.4], use_lm=source_lang!="ru", lm_model_name=lm_model[target_lang], translate_batch_size=4)
    contrast = ContrastScore(source_language=source_lang, target_language=target_lang, parallelize=True)
    dataset = DatasetLoader(source_lang, target_lang, max_monolingual_sent_len=max_len)
    mono_src, mono_tgt = dataset.load("monolingual-align")
    train_src, train_tgt = dataset.load("monolingual-train")
    para_src, para_tgt = dataset.load("nepali" if "ne" in [source_lang, target_lang] else "wikimatrix", 200000)
    suffix = f"{source_lang}-{target_lang}-awesome-wmd-{xmover.mapping}-monolingual-align-{xmover.k}-{xmover.remap_size}-{40000}-{max_len}"

    for iteration in range(1, remap_iterations + 1):
        logging.info(f"Remapping iteration {iteration}.")
        xmover.remap(mono_src, mono_tgt, suffix=suffix + f"-{iteration}", overwrite=False)
    if source_lang != "ru":
        for iteration in range(nmt_iterations):
            logging.info(f"NMT training iteration {iteration}.")
            xmover.train(train_src, train_tgt, suffix=suffix+f"-{remap_iterations}", iteration=iteration, overwrite=False, k=1)
    for iteration in range(1, contrast_iterations + 1):
        logging.info(f"Contrastive Learning iteration {iteration}.")
        contrast.suffix = f"{max_len}-{iteration}"
        contrast.train(train_src, train_tgt, overwrite=False)

    xmover.mapping = "CLP"
    xmover.remap(para_src, para_tgt, suffix=suffix.replace("UMD", "CLP") + f"-finetuned-200000", aligned=True, overwrite=False)
    if source_lang != "ru":
        xmover.train(para_src, para_tgt, suffix=suffix + f"-finetuned-200000", iteration=iteration, aligned=True,
            finetune=True, overwrite=False, k=1)
    contrast.suffix = f"{max_len}-finetuned-200000"
    contrast.train(para_src, para_tgt, aligned=True, finetune=True, overwrite=False)

    return lambda src, sys: [0.6 * x + 0.4 * y for x, y in zip(xmover.score(src, sys), contrast.score(src, sys))]

def tests(source_lang, target_lang, dataset_name, max_len=30):
    xcontrast = xmover_contrast_combine(source_lang, target_lang, max_len)
    sentsim = SentSim()
    dataset = DatasetLoader(source_lang, target_lang, max_monolingual_sent_len=max_len)
    eval_src, eval_system, eval_scores = dataset.load(dataset_name)
    results, index = defaultdict(list), ["SentSim", f"XMover + Contrast ({max_len} tokens)", f"Ensemble ({max_len} tokens)"]

    for score in [sentsim.score, xcontrast, lambda src, sys:[0.5 * x + 0.5 * y for x, y in zip(xcontrast(src, sys), sentsim.score(src, sys))]]:
        pearson, spearman = correlation(score(eval_src, eval_system), eval_scores)
        logging.info(f"Pearson: {pearson}, Spearman: {spearman}")
        results["pearson"].append(round(100 * pearson, 2))
        results["spearman"].append(round(100 * spearman, 2))

    return tabulate(results, headers="keys", showindex=index)

logging.basicConfig(level=logging.INFO, datefmt="%m-%d %H:%M", format="%(asctime)s %(levelname)-8s %(message)s")
for source_lang, target_lang in mlqe:
    print(f"Evaluating {source_lang}-{target_lang} language direction on MLQE-PE.")
    print(tests(source_lang, target_lang, "scored-mlqe", max_len=30))
    print(tests(source_lang, target_lang, "scored-mlqe", max_len=50))
