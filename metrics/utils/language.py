from os.path import isfile, join
from fasttext import FastText, load_model
from urllib.request import urlretrieve
from collections import defaultdict
from tempfile import mkdtemp
from mosestokenizer import MosesTokenizer, MosesSentenceSplitter
from Nepali_nlp import Tokenizer
from sinling import SinhalaTokenizer
from jieba import cut
from re import findall, U

class LangDetect():
    url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/"

    def __init__(self, compress=False, cache_dir=mkdtemp()):
        # fixes https://github.com/facebookresearch/fastText/issues/1067 for the time being
        FastText.eprint = lambda _: None
        self.cache_dir = cache_dir
        self.model = self.load_model("lid.176.ftz" if compress else "lid.176.bin")

    def load_model(self, name):
        target_path = join(self.cache_dir, name)
        if not isfile(target_path):
            urlretrieve(join(self.url, name), target_path)
        return load_model(target_path)

    def detect(self, texts, return_score=False):
        texts = [texts] if isinstance(texts, str) else texts
        counter = defaultdict(float)

        for text in texts:
            labels, scores = self.model.predict(text.strip())
            label = labels[0].removeprefix("__label__")
            score = min(float(scores[0]), 1.0)
            counter[label] += score
        label, score = sorted(counter.items(), key=lambda tup: tup[1])[-1]
        return (label, score) if return_score else label


class WordTokenizer():
    def __init__(self, language):
        if language == "si":
            self.tokenize = SinhalaTokenizer().tokenize
        # since bn and hi are related to ne and use the same script we can use the ne tokenizer for all
        elif language in ["ne", "bn", "hi"]:
            self.tokenize = Tokenizer().word_tokenize
        elif language == "zh":
            self.tokenize = lambda sent: list(cut(sent))
        else:
            # zulu and xhosa follow english punctuation
            self.tokenize = MosesTokenizer("en" if language in ["zu", "xh"] else language)

    def __call__(self, sentence):
        return self.tokenize(sentence)

    def __enter__(self):
        return self.tokenize

    def __exit__(self, *_):
        if type(self.tokenize) == MosesTokenizer:
            self.tokenize.close()

    def __del__(self):
        if type(self.tokenize) == MosesTokenizer:
            self.tokenize.close()

class SentenceSplitter():
    def __init__(self, language):
        if language in ["si"]:
            self.split = lambda sents: SinhalaTokenizer().split_sentences(" ".join(sents))
        elif language in ["ne", "bn", "hi"]:
            self.split = lambda sents: Tokenizer().sentence_tokenize(" ".join(sents))
        elif language == "zh":
            self.split = lambda sent: self._split_chinese(sent)
        else:
            self.split = MosesSentenceSplitter("en" if language in ["zu", "xh"] else language, False)

    # taken from https://stackoverflow.com/a/45274695, modified regex of
    # http://aclweb.org/anthology/Y/Y11/Y11-1038.pdf
    def _split_chinese(_, sentences):
        return [sent.strip() for sent in findall(u'[^!?。\.\!\?]+[!?。\.\!\?]?', "".join(sentences), flags=U)]

    def __call__(self, sentence):
        return self.split(sentence)

    def __enter__(self):
        return self.split

    def __exit__(self, *_):
        if type(self.split) == MosesSentenceSplitter:
            self.split.close()

    def __del__(self):
        if type(self.split) == MosesSentenceSplitter:
            self.split.close()
