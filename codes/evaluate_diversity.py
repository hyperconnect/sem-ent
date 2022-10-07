import json
import random

import numpy as np
import torch
from mauve.utils import get_tokenizer

from parlai.core.agents import create_agent
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript
from parlai.core.script import register_script
from parlai.scripts.eval_model import setup_args as eval_model_setup_args
from parlai.scripts.train_diversify_base import clustering
from parlai.scripts.train_diversify_base import prepare_clustering
from parlai.scripts.train_model import load_eval_worlds
from parlai.utils import logging
from parlai.utils.ent import Ent1Calculator
from parlai.utils.ent import Ent2Calculator
from parlai.utils.ent import Ent3Calculator
from parlai.utils.io import PathManager
from parlai.utils.low_frequency import LF100Calculator


def setup_args(parser=None) -> ParlaiParser:
    """
    Build the ParlAI parser, adding command line args if necessary.

    :param ParlaiParser parser:
        Preexisting parser to append options to. Will be created if needed.

    :returns:
        the ParlaiParser with CLI options added.
    """
    parser = eval_model_setup_args(parser)
    distribution_match = parser.add_argument_group('Distribution Match Related Arguments')
    distribution_match.add_argument(
        '--dm-samples',
        type=int,
        default=5000,
    )
    distribution_match.add_argument(
        '--num-buckets',
        default=None,
        type=int,
    )
    distribution_match.add_argument(
        '--dm-option',
        type=str,
        default="baseline"
    )
    distribution_match.add_argument(
        '--not-using-short-samples',
        type=bool,
        default=False,
    )
    distribution_match.add_argument(
        '--cluster-embedding-model-name',
        type=str,
        default='microsoft/DialoGPT-large',
    )
    distribution_match.add_argument(
        '--split-batch',
        type=int,
        default=None,
    )
    distribution_match.add_argument('--seed', type=int, default=42)
    return parser


def evaluate_diversity(opt):
    random.seed(opt["seed"])
    np.random.seed(opt["seed"])
    if 'train' in opt['datatype'] and 'evalmode' not in opt['datatype']:
        raise ValueError(
            'You should use --datatype train:evalmode if you want to evaluate on '
            'the training set.'
        )

    # load model and possibly print opt
    agent = create_agent(opt, requireModelExists=True)
    agent.opt.log()
    train_world = load_eval_worlds(agent, opt, 'train')
    valid_worlds = load_eval_worlds(agent, opt, 'test')
    tokenizer = get_tokenizer(opt["cluster_embedding_model_name"])

    reports = {}
    for v_world in valid_worlds:
        v_world.reset()
        pca, kmeans, train_texts, train_labels, train_hist = prepare_clustering(
            opt, train_world[0], tokenizer, max_text_length=256,
            num_buckets="auto" if opt["num_buckets"] is None else opt["num_buckets"])
        torch.save(train_texts, f"train_texts.pt")
        torch.save(train_labels, f"train_labels.pt")
        torch.save(train_hist, f"train_hist.pt")
        with torch.no_grad():
            *_, output, p_texts, q_texts, _ = clustering(agent,
                                                         opt,
                                                         v_world,
                                                         is_training=False,
                                                         tokenizer=tokenizer,
                                                         pca=pca,
                                                         kmeans=kmeans)
        id = v_world.world.agents[0].id
        responses = [dialogue[-1] for dialogue in q_texts]
        ent_1 = Ent1Calculator().calculate(responses)[0]
        ent_2 = Ent2Calculator().calculate(responses)[0]
        ent_3 = Ent3Calculator().calculate(responses)[0]
        lf_100 = LF100Calculator().calculate(responses)
        reports[id] = {
            "mauve": output.mauve,
            "num_samples": len(p_texts),
            "sem-ent": {
                "p": output.p_entropy,
                "q": output.q_entropy,
            },
            "dist": output.dist,
            "generated": q_texts,
            "real": p_texts,
            "ent_1": ent_1,
            "ent_2": ent_2,
            "ent_3": ent_3,
            "lf_100": lf_100,
        }

    with PathManager.open(opt["report_filename"] + ".json", "w") as f:
        logging.info(f"Saving model report to {opt['report_filename']}")
        json.dump({'opt': opt, 'report': reports}, f, indent=4)
        f.write("\n")
        torch.save(output.p_hist, f"{opt['report_filename']}_phist")
        torch.save(output.q_hist, f"{opt['report_filename']}_qhist")
        torch.save(output.p_labels, f"{opt['report_filename']}_plabel")
        torch.save(output.p_text, f"{opt['report_filename']}_ptext")
        torch.save(output.q_labels, f"{opt['report_filename']}_qlabel")
        torch.save(output.q_text, f"{opt['report_filename']}_qtext")


@register_script('evaluate_diversity', aliases=['ed'])
class EvaluateDiversity(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return evaluate_diversity(self.opt)


if __name__ == '__main__':
    EvaluateDiversity.main()
