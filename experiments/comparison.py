#!/usr/bin/env python
from metrics.xmoverscore import XMoverScore, XMoverNMTLMBertAlignScore
from metrics.distilscore import DistilScore
from metrics.contrastscore import ContrastScore
from metrics.sentsim import SentSim
from metrics.utils.dataset import DatasetLoader
from collections import defaultdict
from tabulate import tabulate
from datasets import load_metric
from numpy import corrcoef, argsort
from comet import download_model, load_from_checkpoint
import logging

from torch.cuda import is_available as cuda_is_available
from transquest.algo.sentence_level.monotransquest.run_model import MonoTransQuestModel

newstest2016 = [
    ("de", "en"),
    ("en", "ru"),
    ("ru", "en"),
    ("ro", "en"),
    ("cs", "en"),
    ("fi", "en"),
    ("tr", "en"),
]
newstest2017 = [
    ("cs", "en"),
    ("de", "en"),
    ("fi", "en"),
    ("lv", "en"),
    ("ru", "en"),
    ("tr", "en"),
    ("zh", "en"),
]
mlqe = [
    ("en", "de"),
    ("en", "zh"),
    ("ru", "en"),
    ("ro", "en"),
    ("et", "en"),
    ("ne", "en"),
    ("si", "en"),
]
eval4nlp = [
    (src, tgt)
    for src, tgt in [("de", "zh"), ("ru", "de")]
    if DatasetLoader(src, tgt).has_eval4nlp_access()
]
mqm = [("en", "de"), ("zh", "en")]
wmt21_flores = [("zu", "xh"), ("xh", "zu"), ("bn", "hi"), ("hi", "bn")]

lm_model = {
    "en": "gpt2",
    "ru": "sberbank-ai/rugpt3small_based_on_gpt2",
    "de": "dbmdz/german-gpt2",
    "zh": "uer/gpt2-chinese-cluecorpussmall",
}

remap_iterations = 1
nmt_iterations = 1
contrast_iterations = 6

def correlation(model_scores, ref_scores):
    ref_ranks, ranks = argsort(ref_scores).argsort(), argsort(model_scores).argsort()
    return corrcoef(ref_scores, model_scores)[0,1], corrcoef(ref_ranks, ranks)[0,1]

def comet_tests(source_lang, target_lang, dataset_name):
    model = load_from_checkpoint(download_model("wmt20-comet-qe-da"))
    dataset = DatasetLoader(source_lang, target_lang)
    results, index = defaultdict(list), ["Comet-QE"]

    eval_src, eval_system, eval_scores = dataset.load(dataset_name)
    data = [{"src": src, "mt": system} for src, system in zip(eval_src, eval_system)]
    scores, _ = model.predict(data, batch_size=8, gpus=cuda_is_available() and 1 or 0)

    pearson, spearman = correlation(scores, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}")
    results["pearson"].append(round(100 * pearson, 2))
    results["spearman"].append(round(100 * spearman, 2))

    return tabulate(results, headers="keys", showindex=index)

def transquest_tests(source_lang, target_lang, dataset_name):
    transquest_models = {
        ("ro", "en"): "TransQuest/monotransquest-da-ro_en-wiki",
        ("et", "en"): "TransQuest/monotransquest-da-et_en-wiki",
        ("ne", "en"): "TransQuest/monotransquest-da-ne_en-wiki",
        ("si", "en"): "TransQuest/monotransquest-da-si_en-wiki",
        ("ru", "en"): "TransQuest/monotransquest-da-ru_en-reddit_wikiquotes",
        ("en", "de"): "TransQuest/monotransquest-da-en_de-wiki",
        ("en", "zh"): "TransQuest/monotransquest-da-en_zh-wiki",
        ("en", None): "TransQuest/monotransquest-da-en_any",
        (None, "en"): "TransQuest/monotransquest-da-any_en",
        (None, None): "TransQuest/monotransquest-da-multilingual",
    }

    if (source_lang, target_lang) in transquest_models:
        model_name = transquest_models[(source_lang, target_lang)]
    elif (source_lang, None) in transquest_models:
        model_name = transquest_models[(source_lang, None)]
    elif (None, target_lang) in transquest_models:
        model_name = transquest_models[(None, target_lang)]
    else:
        model_name = transquest_models[(None, None)]

    model = MonoTransQuestModel("xlmroberta", model_name, num_labels=1, use_cuda=cuda_is_available())
    dataset = DatasetLoader(source_lang, target_lang)
    eval_src, eval_system, eval_scores = dataset.load(dataset_name)
    results, index = defaultdict(list), ["MonoTransQuest"]
    scores, _ = model.predict(list(map(list, zip(eval_src, eval_system))))
    print(scores[0])

    pearson, spearman = correlation(scores, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}")
    results["pearson"].append(round(100 * pearson, 2))
    results["spearman"].append(round(100 * spearman, 2))

    return tabulate(results, headers="keys", showindex=index)

def bleu_test(source_lang, target_lang, dataset_name):
    metric = load_metric("sacrebleu")
    dataset = DatasetLoader(source_lang, target_lang, return_references=True)
    _, eval_ref, eval_system, eval_scores = dataset.load(dataset_name)
    results, index = defaultdict(list), ["BLEU"]

    scores = list()
    for system, ref in zip(eval_system, eval_ref):
        scores.append(metric.compute(predictions=[system], references=[[ref]])['score'])
    pearson, spearman = correlation(scores, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}")
    results["pearson"].append(round(100 * pearson, 2))
    results["spearman"].append(round(100 * spearman, 2))

    return tabulate(results, headers="keys", showindex=index)

def xmoverscore_tests(source_lang, target_lang, dataset_name, mapping="UMD"):
    scorer = XMoverScore(mapping=mapping, use_lm=True, lm_model_name=lm_model[target_lang])
    dataset = DatasetLoader(source_lang, target_lang)
    eval_src, eval_system, eval_scores = dataset.load(dataset_name)
    results, index = defaultdict(list), [f"XMoverScore ({mapping})"]

    try:
        scorer.remap(source_lang, target_lang)
    except ValueError:
        results["pearson"].append("-")
        results["spearman"].append("-")
    else:
        pearson, spearman = scorer.correlation(eval_src, eval_system, eval_scores)
        logging.info(f"Pearson: {pearson}, Spearman: {spearman}")
        results["pearson"].append(round(100 * pearson, 2))
        results["spearman"].append(round(100 * spearman, 2))

    return tabulate(results, headers="keys", showindex=index)

def sentsim_tests(source_lang, target_lang, dataset_name, word_metric="BERTScore"):
    scorer = SentSim(use_wmd=word_metric=="WMD")
    dataset = DatasetLoader(source_lang, target_lang)
    eval_src, eval_system, eval_scores = dataset.load(dataset_name)
    results, index = defaultdict(list), [f"SentSim ({word_metric})"]

    pearson, spearman = scorer.correlation(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}")
    results["pearson"].append(round(100 * pearson, 2))
    results["spearman"].append(round(100 * spearman, 2))

    return tabulate(results, headers="keys", showindex=index)

def distilscore_tests(source_lang, target_lang, dataset_name):
    scorer = DistilScore(student_model_name="xlm-r-bert-base-nli-stsb-mean-tokens", source_language=source_lang,
            target_language=target_lang, student_is_pretrained=True, suffix="1")
    dataset = DatasetLoader(source_lang, target_lang)
    eval_src, eval_system, eval_scores = dataset.load(dataset_name)
    results, index = defaultdict(list), ["DistilScore"]

    pearson, spearman = scorer.correlation(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}")
    results["pearson"].append(round(100 * pearson, 2))
    results["spearman"].append(round(100 * spearman, 2))

    return tabulate(results, headers="keys", showindex=index)

def uscore_tests(source_lang, target_lang, dataset_name, max_len=30):
    xmover = XMoverNMTLMBertAlignScore(src_lang=source_lang, tgt_lang=target_lang, lm_weights=[1, 0.1],
            nmt_weights=[0.5, 0.4], use_lm=True, lm_model_name=lm_model[target_lang])
    contrast = ContrastScore(source_language=source_lang, target_language=target_lang, parallelize=True)
    dataset = DatasetLoader(source_lang, target_lang, max_monolingual_sent_len=max_len)
    mono_src, mono_tgt = dataset.load("monolingual-align")
    train_src, train_tgt = dataset.load("monolingual-train")
    eval_src, eval_system, eval_scores = dataset.load(dataset_name)
    suffix = f"{source_lang}-{target_lang}-awesome-wmd-{xmover.mapping}-monolingual-align-{xmover.k}-{xmover.remap_size}-{40000}-{max_len}"
    results, index = defaultdict(list), [f"UScore (WMD) ({max_len} tokens)", f"UScore (COS) ({max_len} tokens)",
            f"UScore (WMD) + UScore (COS) ({max_len} tokens)"]

    logging.info("Evaluating UScore (WMD)")
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

    logging.info("Evaluating UScore (COS)")
    for iteration in range(1, contrast_iterations + 1):
        logging.info(f"Contrastive Learning iteration {iteration}.")
        contrast.suffix = f"{max_len}-{iteration}"
        contrast.train(train_src, train_tgt, overwrite=False)

    pearson, spearman = contrast.correlation(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}")
    results["pearson"].append(round(100 * pearson, 2))
    results["spearman"].append(round(100 * spearman, 2))

    logging.info("Evaluating UScore (WMD) + UScore (COS)")
    wmd_scores, contrast_scores = xmover.score(eval_src, eval_system), contrast.score(eval_src, eval_system)
    pearson, spearman = correlation([0.6 * x + 0.4 * y for x, y in zip(wmd_scores, contrast_scores)], eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}")
    results["pearson"].append(round(100 * pearson, 2))
    results["spearman"].append(round(100 * spearman, 2))

    return tabulate(results, headers="keys", showindex=index)

logging.basicConfig(
    level=logging.INFO,
    datefmt="%m-%d %H:%M",
    format="%(asctime)s %(levelname)-8s %(message)s",
)
ref_datasets = (
    ("Newstest-2016", "scored", newstest2016),
    ("Newstest-2017", "scored-wmt17", newstest2017),
    ("MQM-Newstest-2020", "scored-mqm", mqm),
)
datasets = ref_datasets + (
    ("MLQE-PE", "scored-mlqe", mlqe),
    ("Eval4NLP-2021", "scored-eval4nlp", eval4nlp),
    ("WMT21-Flores", "scored-wmt21.flores", wmt21_flores),
)
for dataset, identifier, pairs in datasets:
    for source_lang, target_lang in pairs:
        print(f"Evaluating {source_lang}-{target_lang} language direction on {dataset}")
        print(uscore_tests(source_lang, target_lang, identifier, max_len=30))
        print(uscore_tests(source_lang, target_lang, identifier, max_len=50))
        print(xmoverscore_tests(source_lang, target_lang, identifier, mapping="UMD"))
        print(xmoverscore_tests(source_lang, target_lang, identifier, mapping="CLP"))
        print(sentsim_tests(source_lang, target_lang, identifier, word_metric="BERTScore"))
        print(sentsim_tests(source_lang, target_lang, identifier, word_metric="WMD"))
        print(distilscore_tests(source_lang, target_lang, identifier))
        print(transquest_tests(source_lang, target_lang, identifier))
        print(comet_tests(source_lang, target_lang, identifier))

for dataset, identifier, pairs in ref_datasets:
    for source_lang, target_lang in pairs:
        print(bleu_test(source_lang, target_lang, identifier))
