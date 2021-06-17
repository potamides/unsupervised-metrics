# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# Adapted, based on https://github.com/huggingface/transformers/blob/v4.5.1/examples/seq2seq/run_translation.py

import logging
import os
from dataclasses import dataclass, field
from typing import Optional
from torch.utils.data import DataLoader
from .dataset import DATADIR

from datasets import load_dataset

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    source_lang: str = field(default=None, metadata={"help": "Source language id for translation."})
    target_lang: str = field(default=None, metadata={"help": "Target language id for translation."})

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a jsonlines)."})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    def __post_init__(self):
        if self.train_file is None:
            raise ValueError("Need a training file.")
        elif self.source_lang is None or self.target_lang is None:
            raise ValueError("Need to specify the source language and the target language.")

        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension == "json", "`train_file` should be a json file."

def load_model_and_tokenizer(model_name_or_path, source_lang, target_lang, use_fast_tokenizer, cache_dir):
    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        use_fast=use_fast_tokenizer,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        cache_dir=cache_dir,
    )

    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[target_lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(target_lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # For translation we set the codes of our source and target languages (only useful for mBART, the others will
    # ignore those attributes).
    if isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        tokenizer.src_lang = source_lang
        tokenizer.tgt_lang = target_lang

    return model, tokenizer

def _train(args=None):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

        if len(list(filter(os.path.isfile, os.listdir(training_args.output_dir)))) > 0:
            logger.info(
                f"Output directory ({training_args.output_dir}) exists already and is not empty. "
                "Skipping training and returning pretrained models."
            )
            return load_model_and_tokenizer(training_args.output_dir, data_args.source_lang,
                    data_args.target_lang, model_args.use_fast_tokenizer, model_args.cache_dir)
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )


    # Set seed before initializing model.
    set_seed(training_args.seed)

    # For translation, only JSON files are supported, with one field named "translation" containing two keys for the
    # source and target languages (unless you adapt what follows).
    data_files = {}
    data_files["train"] = data_args.train_file
    extension = data_args.train_file.split(".")[-1]
    datasets = load_dataset(extension, data_files=data_files, download_mode="force_redownload")
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html

    model, tokenizer = load_model_and_tokenizer(model_args.model_name_or_path, data_args.source_lang,
        data_args.target_lang, model_args.use_fast_tokenizer, model_args.cache_dir)

    # Get the language codes for input/target.
    source_lang = data_args.source_lang.split("_")[0]
    target_lang = data_args.target_lang.split("_")[0]

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warn(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
        inputs = [ex[source_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = datasets["train"]
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=datasets["train"].column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif os.path.isdir(model_args.model_name_or_path):
        checkpoint = model_args.model_name_or_path
    else:
        checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(training_args.output_dir)  # Saves the tokenizer too

    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_state()

    return model, tokenizer

def train(model, source_lang, target_lang, dataset, overwrite, suffix):
    if "mbart" in model:
        language2mBART = {
            "ar": "ar_AR", "cs": "cs_CZ", "de": "de_DE", "en": "en_XX", "es": "es_XX",
            "et": "et_EE", "fi": "fi_FI", "fr": "fr_XX", "gu": "gu_IN", "hi": "hi_IN",
            "it": "it_IT", "ja": "ja_XX", "kk": "kk_KZ", "ko": "ko_KR", "lt": "lt_LT",
            "lv": "lv_LV", "my": "my_MM", "ne": "ne_NP", "nl": "nl_XX", "ro": "ro_RO",
            "ru": "ru_RU", "si": "si_LK", "tr": "tr_TR", "vi": "vi_VN", "zh": "zh_CN" }
        source_lang = language2mBART[source_lang]
        target_lang = language2mBART[target_lang]
    args = [
        "--model_name_or_path", model,
        "--cache_dir", os.path.join(DATADIR, os.path.basename(model), suffix, "cache"),
        "--output_dir", os.path.join(DATADIR, os.path.basename(model), suffix, "output"),
        "--source_lang", source_lang,
        "--target_lang", target_lang,
        "--train_file", dataset,
        "--save_strategy", "epoch",
        "--per_device_train_batch_size", "4", "--do_train"]
    if overwrite:
        args.append("--overwrite_output_dir")

    return _train(args)

def translate(model, tokenizer, sentences, batch_size, device):
    translated = list()
    for batch in DataLoader(sentences, batch_size=batch_size):
        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        translated_tokens = model.generate(**inputs, decoder_start_token_id=model.config.decoder_start_token_id)
        translated.extend(tokenizer.batch_decode(translated_tokens.cpu(), skip_special_tokens=True))
    return translated

if __name__ == "__main__":
    train()
