#!/usr/bin/env python
from metrics.xmoverscore import XMoverBertAlignScore, XMoverVecMapAlignScore, XMoverNMTBertAlignScore
from utils.dataset import DatasetLoader
import logging

source_lang, target_lang = "de", "en"
iterations = 5
min_monolingual_sent_len, max_monolingual_sent_len = 3, 80

def align_tests(alignment="awesome", mapping="UMD", data="monolingual-align", valid="scored", metric="cosine"):
    aligner = XMoverBertAlignScore(alignment=alignment, mapping=mapping, use_cosine=True if metric == "cosine" else False)
    dataset = DatasetLoader(source_lang, target_lang, min_monolingual_sent_len, max_monolingual_sent_len)
    parallel_src, parallel_tgt = dataset.load("parallel")
    mono_src, mono_tgt = dataset.load(data)
    eval_src, eval_system, eval_scores = dataset.load(valid)
    suffix = f"{source_lang}-{target_lang}-{alignment}-{metric}-{mapping}-{data}-{aligner.k}-{aligner.remap_size}-{len(mono_src)}"
    results = {"suffix": suffix}

    logging.info("Evaluating performance before remapping.")
    precision = aligner.precision(parallel_src, parallel_tgt)
    pearson, spearman = aligner.correlation(eval_src, eval_system, eval_scores)
    rmse, mae = aligner.error(eval_src, eval_system, eval_scores)
    logging.info(f"Precision @ 1: {precision}")
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}")
    logging.info(f"RMSE: {rmse}, MAE: {mae}")
    results[0] = {"precision": round(precision, 3), "pearson": round(pearson, 3), "spearman": round(spearman, 3),
            "rmse": round(rmse, 3), "mae": round(mae, 3)}

    for iteration in range(1, iterations + 1):
        logging.info(f"Remapping iteration {iteration}.")
        aligner.remap(mono_src, mono_tgt, suffix=suffix + f"-{iteration}", overwrite=False)
        precision = aligner.precision(parallel_src, parallel_tgt)
        pearson, spearman = aligner.correlation(eval_src, eval_system, eval_scores)
        rmse, mae = aligner.error(eval_src, eval_system, eval_scores)
        logging.info(f"Precision @ 1: {precision}")
        logging.info(f"Pearson: {pearson}, Spearman: {spearman}")
        logging.info(f"RMSE: {rmse}, MAE: {mae}")
        results[iteration] = {"precision": round(precision, 3), "pearson": round(pearson, 3),
                "spearman": round(spearman, 3), "rmse": round(rmse, 3), "mae": round(mae, 3)}

    return results
  
def vecmap_tests():
    aligner = XMoverVecMapAlignScore(src_lang=source_lang, tgt_lang=target_lang)
    dataset = DatasetLoader(source_lang, target_lang, min_monolingual_sent_len, max_monolingual_sent_len)
    parallel_src, parallel_tgt = dataset.load("parallel")
    eval_src, eval_system, eval_scores = dataset.load("scored")

    logging.info(f"Precision: {aligner.precision(parallel_src, parallel_tgt)}.")
    logging.info("Pearson: {}, Spearman: {}".format(*aligner.correlation(eval_src, eval_system, eval_scores)))
    logging.info("RMSE: {}, MAE: {}".format(*aligner.error(eval_src, eval_system, eval_scores)))

def nmt_tests(valid="scored", metric="cosine"):
    aligner = XMoverNMTBertAlignScore(src_lang=source_lang, tgt_lang=target_lang, use_cosine=True if metric == "cosine" else False)
    dataset = DatasetLoader(source_lang, target_lang, min_monolingual_sent_len, max_monolingual_sent_len)
    mono_src, mono_tgt = dataset.load("monolingual-align")
    eval_src, eval_system, eval_scores = dataset.load(valid)
    suffix = f"{source_lang}-{target_lang}-awesome-{metric}-{aligner.mapping}-monolingual-align-{aligner.k}-{aligner.remap_size}-{len(mono_src)}"
    results = {"suffix": suffix}

    logging.info("Evaluating performance before remapping.")
    pearson, spearman = aligner.correlation(eval_src, eval_system, eval_scores)
    rmse, mae = aligner.error(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}")
    logging.info(f"RMSE: {rmse}, MAE: {mae}")
    results[0] = {"pearson": round(pearson, 3), "spearman": round(spearman, 3), "rmse": round(rmse, 3),
            "mae": round(mae, 3)}
    for iteration in range(1, iterations + 1):
        logging.info(f"Remapping iteration {iteration}.")
        aligner.remap(mono_src, mono_tgt, suffix=suffix + f"-{iteration}", overwrite=False)
        pearson, spearman = aligner.correlation(eval_src, eval_system, eval_scores)
        rmse, mae = aligner.error(eval_src, eval_system, eval_scores)
        logging.info(f"Pearson: {pearson}, Spearman: {spearman}")
        logging.info(f"RMSE: {rmse}, MAE: {mae}")
        results[iteration] = {"pearson": round(pearson, 3), "spearman": round(spearman, 3), "rmse": round(rmse, 3),
                "mae": round(mae, 3)}
    mono_src, mono_tgt = dataset.load("monolingual-train")
    aligner.train(mono_src, mono_tgt, suffix=suffix + f"-{iterations}", overwrite=False)

    logging.info("Evaluating performance with NMT model.")
    pearson, spearman = aligner.correlation(eval_src, eval_system, eval_scores)
    rmse, mae = aligner.error(eval_src, eval_system, eval_scores)
    logging.info(f"Pearson: {pearson}, Spearman: {spearman}")
    logging.info(f"RMSE: {rmse}, MAE: {mae}")
    results[f"{iterations} + NMT"] = {"pearson": round(pearson, 3), "spearman": round(spearman, 3),
            "rmse": round(rmse, 3), "mae": round(mae, 3)}

    return results

logging.basicConfig(level=logging.INFO, datefmt="%m-%d %H:%M", format="%(asctime)s %(levelname)-8s %(message)s")
print(align_tests(alignment="awesome", data="monolingual-align", mapping="UMD", valid="scored"))
print(align_tests(alignment="awesome", data="monolingual-align", mapping="CLP", valid="scored"))
print(align_tests(alignment="awesome", data="monolingual-align", mapping="UMD", valid="scored-wmt"))
print(align_tests(alignment="awesome", data="monolingual-align", mapping="CLP", valid="scored-wmt"))
print(align_tests(alignment="fast", data="monolingual-align", mapping="UMD", valid="scored"))
print(align_tests(alignment="fast", data="monolingual-align", mapping="CLP", valid="scored"))
print(align_tests(alignment="fast", data="monolingual-align", mapping="UMD", valid="scored-wmt"))
print(align_tests(alignment="fast", data="monolingual-align", mapping="CLP", valid="scored-wmt"))
print(align_tests(alignment="awesome", data="parallel-align", mapping="UMD", valid="scored"))
print(align_tests(alignment="awesome", data="parallel-align", mapping="CLP", valid="scored"))
print(align_tests(alignment="awesome", data="parallel-align", mapping="UMD", valid="scored-wmt"))
print(align_tests(alignment="awesome", data="parallel-align", mapping="CLP", valid="scored-wmt"))
print(align_tests(alignment="fast", data="parallel-align", mapping="UMD", valid="scored"))
print(align_tests(alignment="fast", data="parallel-align", mapping="CLP", valid="scored"))
print(align_tests(alignment="fast", data="parallel-align", mapping="UMD", valid="scored-wmt"))
print(align_tests(alignment="fast", data="parallel-align", mapping="CLP", valid="scored-wmt"))
#vecmap_tests()
print(nmt_tests(valid="scored", metric="cosine"))
print(nmt_tests(valid="scored-wmt", metric="cosine"))
print(nmt_tests(valid="scored", metric="wmd"))
print(nmt_tests(valid="scored-wmt", metric="wmd"))
