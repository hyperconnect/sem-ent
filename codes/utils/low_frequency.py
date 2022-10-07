from collections import Counter
from typing import List

from nltk.tokenize.casual import casual_tokenize
from nltk.util import ngrams


# Referencing https://openreview.net/pdf?id=S1efxTVYDr
class LFCalculator:
    def __init__(self, thres: int):
        self.thres = thres

    def calculate(self, instances: List[str]) -> float:
        metric_value = self._calculate(instances)
        return metric_value

    def _calculate(self, instances: List[str]) -> float:
        tokenized_responses = [
            casual_tokenize(instance) for instance in instances
        ]
        num_all_ngrams = 0
        all_ngram_list = list()

        for tokens in tokenized_responses:
            token_ngrams = list(ngrams(tokens, 1))
            num_all_ngrams += len(token_ngrams)
            all_ngram_list.extend(token_ngrams)

        counter = Counter(all_ngram_list)
        low_frequent_word_count = 0
        for k, v in list(counter.items()):
            if v <= self.thres:
                low_frequent_word_count += v

        return low_frequent_word_count / len(all_ngram_list)


class LF100Calculator(LFCalculator):
    def __init__(self):
        super().__init__(thres=100)
