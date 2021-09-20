# Experiments
This directory contains experiments which were conducted during the
master-thesis *Self-Learning for Unsupervised Evaluation Metrics*. By default
the experiments train the used models from scratch, since it is difficult to
distribute all created model files due to storage limitations. If you need the
original model files due to reproducability reasons, please contact the
maintainers of this repository. Created files are cached in
`${METRICS_HOME:-${XDG_CACHE_HOME:-~/.cache}/metrics}`, so training and
pre-processing only happens once. Please be careful when interrupting a running
process, as created files are not yet checked for their integrity.

Also please bear in mind, that most models were trained on beefy workstations
like the [NVIDIA DGX A100](https://www.nvidia.com/en-us/data-center/dgx-a100).
The majority of experiments require considerably less resources, but in this
case out-of-memory errors are to be expected. Model inference is of course less
resource intensive.

## Included Experiments
* `remap.py` Remap XMoverScore on pseudo-parallel sentences.
* `vecmap.py` Use XMoverScore and mean-pooling metrics with [VecMap](https://github.com/artetxem/vecmap) embeddings.
* `nmt.py` Combine XMoverScore with an unsupervised NMT model.
* `lm.py` Combine XMoverScore with an unsupervised NMT model and a language model of the target language.
* `distil.py` Create distilled cross-lingual sentence embeddings using pseudo-parallel sentences.
* `contrast.py` Created cross-lingual sentence embeddings using a contrastive learning objective.
* `combine.py` Combine XMoverScore with ContrastScore.
* `comparison.py` Combine all self-learned metrics with strong baselines on multiple language directions and datasets.
* `finetune.py` Finetune induced self-learned metrics on small parallel corpora.
* `parallel.py` Create distilled and contrastive cross-lingual sentence embeddings only on parallel data.
