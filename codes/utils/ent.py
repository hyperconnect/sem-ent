from collections import Counter
from typing import List
from typing import Tuple

import numpy as np
from nltk.tokenize.casual import casual_tokenize
from nltk.util import ngrams


class EntKCalculator:
    def __init__(self, k: int):
        self.k = k

    def calculate(self, instances: List[str]) -> Tuple[float, int]:
        metric_value = self._calculate(instances)
        return metric_value

    def _calculate(self, instances: List[str]) -> Tuple[float, int]:
        tokenized_responses = [
            casual_tokenize(instance) for instance in instances
        ]
        num_all_ngrams = 0
        all_ngram_list = list()

        for tokens in tokenized_responses:
            token_ngrams = list(ngrams(tokens, self.k))
            num_all_ngrams += len(token_ngrams)
            all_ngram_list.extend(token_ngrams)

        counter = Counter(all_ngram_list)
        ent = 0
        for k, v in list(counter.items()):
            prob = float(v) / float(num_all_ngrams)
            ent -= prob * np.log2(prob)

        return ent, len(all_ngram_list)


class Ent1Calculator(EntKCalculator):
    def __init__(self):
        super().__init__(k=1)


class Ent2Calculator(EntKCalculator):
    def __init__(self):
        super().__init__(k=2)


class Ent3Calculator(EntKCalculator):
    def __init__(self):
        super().__init__(k=3)


if __name__ == "__main__":
    instances = ["Hello", "Hello World", "World Hi"]
    calculator = Ent1Calculator()
    assert calculator.calculate(instances)[0] == -(4 / 5 * np.log2(2 / 5) + 1 / 5 * np.log2(1 / 5))
