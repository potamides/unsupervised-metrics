# Unsupervised Metrics

## Installation
If your goal is to run the [experiments](experiments) clone the repository and
install it in editable mode:
 ```sh
 git clone https://github.com/potamides/unsupervised-metrics
 pip install -e unsupervised-metrics
 ```
If you want to use this project as a library you can also directly install it
as a package:
```sh
pip install 'git+https://github.com/potamides/unsupervised-metrics.git#egg=metrics'
```
If you want to use [fast-align](https://github.com/clab/fast_align) follow its
install instruction and make sure that the `fast_align` and `atools` programs
are on your `PATH`. This requirement is optional.

## Usage

```python
from metrics.xmoverscore import XMoverNMTLMBertAlignScore as XMoverScore
from metrics.utils.dataset import DatasetLoader

src_lang, tgt_lang = "de", "en"

dataset = DatasetLoader(src_lang, tgt_lang)
# instantiate XMoverScore and enable language model
scorer = XMoverScore(src_lang=src_lang, tgt_lang=tgt_lang, use_lm=True)
# remap XMoverScore with UMD
scorer.remap(*dataset.load("monolingual-align"))
# train NMT model to leverage pseudo translations
scorer.train(*dataset.load("monolingual-train"), k=1)

# print correlations with human judgments
print("Pearson's r: {}, Spearman's ρ: {}".format(*scorer.correlation(*dataset.load("scored"))))
```

For more involved examples take a look at the [experiments](experiments).
