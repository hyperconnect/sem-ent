import math

import mauve
import pytest

from examples import load_gpt2_dataset


class TestMauve:

    @pytest.fixture(scope="class")
    def human_texts(self):
        return load_gpt2_dataset('data/amazon.valid.jsonl', num_examples=100)

    @pytest.fixture(scope="class")
    def generated_texts(self):
        return load_gpt2_dataset('data/amazon-xl-1542M.valid.jsonl', num_examples=100)

    def test_default_mauve(self, human_texts, generated_texts):
        out = mauve.compute_mauve(p_text=human_texts,
                                  q_text=generated_texts,
                                  device_id=0,
                                  max_text_length=256,
                                  verbose=False)
        assert math.isclose(out.mauve, 0.9917, abs_tol=1e-4)

    @pytest.mark.parametrize(
        "batch_size",
        [1, 2, 3, 4, 8, 16],
    )
    def test_batchify_mauve(self, human_texts, generated_texts, batch_size):
        out = mauve.compute_mauve(p_text=human_texts,
                                  q_text=generated_texts,
                                  device_id=0,
                                  max_text_length=256,
                                  batch_size=batch_size,
                                  verbose=False)
        assert math.isclose(out.mauve, 0.9917, abs_tol=1e-4)
