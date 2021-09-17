# Unsupervised Metrics
[Unsupervised-Metrics](https://github.com/potamides/unsupervised-metrics) is a
Python library which allows researches and developers alike to experiment with
state-of-the-art evaluation metrics for machine translation. The focus hereby
lies on reference-free, unsupervised metrics, which do not make use of parallel
data in any way, however wrappers around some weakly-supervised metrics like
[XMoverScore](https://aclanthology.org/2020.acl-main.151) and
[SentSim](https://aclanthology.org/2021.naacl-main.252) are provided out of
convenience.

<details><summary>Implemented Papers</summary><p>

  * Self-Learning for Unsupervised Evaluation Metrics
  * [On the Limitations of Cross-lingual Encoders as Exposed by Reference-Free Machine Translation Evaluation](https://aclanthology.org/2020.acl-main.151)
  * [SentSim: Crosslingual Semantic Evaluation of Machine Translation](https://aclanthology.org/2021.naacl-main.252)
</p></details>

## Installation
If you want to use this project as a library you can install it as a regular
package with [pip](https://pip.pypa.io/en/stable):
```sh
pip install 'git+https://github.com/potamides/unsupervised-metrics.git#egg=metrics'
```
If your goal is to run the included [experiments](experiments) clone the
repository and install it in editable mode:
 ```sh
 git clone https://github.com/potamides/unsupervised-metrics
 pip install -e unsupervised-metrics
 ```
If you want to use [fast-align](https://github.com/clab/fast_align) follow its
install instruction and make sure that the `fast_align` and `atools` programs
are on your `PATH`. This requirement is optional.

## Usage

### Train an exising metric
One focus of this library is to make it easy to fine-tune existing
state-of-the-art metrics for arbitrary language pairs and domains.
A simple example is provided in the code block below. For more involved
examples and means on how to instantiate a pre-trained metric take a look at
the [experiments](experiments).

```python
from metrics.contrastscore import ContrastScore
from metrics.utils.dataset import DatasetLoader

src_lang, tgt_lang = "de", "en"

dataset = DatasetLoader(src_lang, tgt_lang)
# instantiate ContrastScore and enable parallel training on multiple GPUs
scorer = ContrastScore(source_language=src_lang, target_language=tgt_lang, parallelize=True)
# train the underlying language model on pseudo-parallel sentence pairs
scorer.train(*dataset.load("monolingual-train"))

# print correlations with human judgments
print("Pearson's r: {}, Spearman's ρ: {}".format(*scorer.correlation(*dataset.load("scored"))))
```

### Create your own metric
This library can also be used as a framework to create new metrics, as
demonstrated in the code block below. Existing metrics are defined in the
[metrics](metrics) package, which could serve as a source of inspiration.

```python
from .common import CommonScore

class MyOwnMetric(CommonScore):
    def align():
        """
        This method receives a list of sentences in the source language and a
        list of sentences in the target language as parameters and returns
        a list of pseudo aligned sentence pairs.
        """

    def _embed():
        """
        This method receives a list of sentences in the source language and a
        list of sentences in the target language as parameters and returns
        their embeddings, inverse document frequences, tokens and padding
        masks.
        """

    def score():
        """
        This method receives a list of sentences in the source language and a
        list of sentences in the target language as parameters, which are
        assumed to be aligned according to their index. For each sentence pair
        a similarity score is computed and the list of scores is returned.
        """
```

## Acknowledgments
This library is based on the following projects:
* [ACL20-Reference-Free-MT-Evaluation](https://github.com/AIPHES/ACL20-Reference-Free-MT-Evaluation)
* [Unsupervised-crosslingual-Compound-Method-For-MT](https://github.com/Rain9876/Unsupervised-crosslingual-Compound-Method-For-MT)
* [Seq2Seq examples](https://github.com/huggingface/transformers/tree/v4.5.1/examples/seq2seq) of [transformers](https://github.com/huggingface/transformers)
* [VecMap](https://github.com/artetxem/vecmap)
* [CRISS](https://github.com/pytorch/fairseq/tree/master/examples/criss)
