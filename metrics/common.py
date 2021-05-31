from abc import ABC, abstractmethod
from numpy import corrcoef, argsort
from torch.nn.functional import mse_loss, l1_loss
from torch import FloatTensor

class CommonScore(ABC):
    @abstractmethod
    def align():
        """
        This method receives a list of sentences in the source language and a
        list of sentences in the target language as parameters and returns
        a list of pseudo aligned sentence pairs.
        """

    @abstractmethod
    def _embed():
        """
        This method receives a list of sentences in the source language and a
        list of sentences in the target language as parameters and returns
        their embeddings, inverse document frequences, tokens and padding
        masks.
        """

    @abstractmethod
    def score():
        """
        This method receives a list of sentences in the source language and a
        list of sentences in the target language as parameters, which are
        assumed to be aligned according to their index. For each sentence pair
        a similarity score is computed and the list of scores is returned.
        """

    def precision(self, source_sents, ref_sents):
        """
        This method receives a list of sentences in the source language and a
        list of sentences in the target language as parameters, which are
        assumed to be aligned. The parallel sentences are then shuffled and
        re-aligned through parallel sentence matching. This method then returns
        the Precision @ 1 score.
        """
        pairs, _ = self.align(source_sents, ref_sents)
        return sum([reference == predicted for reference, (_, predicted) in zip(ref_sents, pairs)]) / len(ref_sents)

    def correlation(self, source_sents, system_sents, ref_scores):
        """
        This method receives a list of sentences in the source language, a
        list of sentences in the target language, which are
        assumed to be aligned and reference scores as parameters. The method
        then returns the pearson correlation and the spearman correlation
        between the reference scores and the scores of the metric.
        """
        scores = self.score(source_sents, system_sents)
        ref_ranks, ranks = argsort(ref_scores).argsort(), argsort(scores).argsort()
        return corrcoef(ref_scores, scores)[0,1], corrcoef(ref_ranks, ranks)[0,1]

    def error(self, source_sents, system_sents, ref_scores):
        """
        This method receives a list of sentences in the source language, a
        list of sentences in the target language, which are
        assumed to be aligned and reference scores as parameters. The method
        then returns the Root Mean Squared Error and the Mean Absulute Error
        between the reference scores and the scores of the metric.
        """
        scores = self.score(source_sents, system_sents)
        rmse = mse_loss(FloatTensor(ref_scores), FloatTensor(scores)).sqrt().item()
        mae = l1_loss(FloatTensor(ref_scores), FloatTensor(scores)).item()
        return rmse, mae
