import json
import random
from types import SimpleNamespace

import mauve
import numpy as np
import torch
from mauve.utils import get_model
from mauve.utils import get_tokenizer
from scipy import stats
from sklearn.metrics import auc as compute_area_under_curve
from sklearn.preprocessing import normalize

from parlai.core.agents import create_agent
from parlai.core.exceptions import StopTrainException
from parlai.core.metrics import Metric
from parlai.core.metrics import dict_report
from parlai.core.params import ParlaiParser
from parlai.core.torch_agent import Batch
from parlai.scripts.train_model import TrainLoop
from parlai.scripts.train_model import load_eval_worlds
from parlai.scripts.train_model import setup_args as train_model_setup_args
from parlai.utils import logging
from parlai.utils.dist import Dist1Calculator
from parlai.utils.dist import Dist2Calculator
from parlai.utils.dist import Dist3Calculator
from parlai.utils.distributed import is_primary_worker
from parlai.utils.distributed import sync_object
from parlai.utils.io import PathManager


def kl_multinomial(p, q):
    assert p.shape == q.shape
    if np.logical_and(p != 0, q == 0).any():
        return np.inf
    else:
        idxs = np.logical_and(p != 0, q != 0)
        return np.sum(p[idxs] * np.log(p[idxs] / q[idxs]))


def get_divergence_curve_for_multinomials(p, q, mixture_weights, scaling_factor):
    # TODO: check if extreme points are needed
    divergence_curve = [[0, np.inf]]  # extreme point
    for w in np.sort(mixture_weights):
        r = w * p + (1 - w) * q
        divergence_curve.append([kl_multinomial(q, r), kl_multinomial(p, r)])
    divergence_curve.append([np.inf, 0])  # other extreme point
    return np.exp(-scaling_factor * np.asarray(divergence_curve))


def setup_args(parser=None) -> ParlaiParser:
    """
    Build the ParlAI parser, adding command line args if necessary.

    :param ParlaiParser parser:
        Preexisting parser to append options to. Will be created if needed.

    :returns:
        the ParlaiParser with CLI options added.
    """
    parser = train_model_setup_args(parser)
    diversifier = parser.add_argument_group('Distribution Match Related Arguments')
    diversifier.add_argument(
        '--dm-every-n-epochs',
        type=float,
        default=1.0,
    )
    diversifier.add_argument(
        '--dm-samples',
        type=int,
        default=5000,
    )
    diversifier.add_argument(
        '--dm-match-training',
        type=bool,
        default=True,
    )
    diversifier.add_argument(
        '--dm-option',
        type=str,
    )
    diversifier.add_argument(
        '--num-buckets',
        default=None,
        type=int,
    )
    diversifier.add_argument(
        '--use-fixed-hist',
        type=bool,
        default=False,
    )
    diversifier.add_argument(
        '--use-generated-texts',
        type=bool,
        default=False,
    )
    diversifier.add_argument(
        '--cluster-embedding-model-name',
        type=str,
        default='microsoft/DialoGPT-large',
    )
    diversifier.add_argument(
        '--balancing-coeff',
        type=float,
        default=30.0,
    )
    return parser


def prepare_clustering(opt, world, tokenizer, max_text_length, num_buckets):
    contexts, labels = [], []
    i = 0
    world.reset()
    num_examples = world.world.agents[0].num_examples()
    while i < num_examples // opt['batchsize']:
        batch_actions = world.batch_act(0, batch_observation=None)
        filtered_batch_actions = []
        episode_done = False
        for ba in batch_actions:
            if "text" not in ba:
                episode_done = True
                break
            filtered_batch_actions.append(ba)
        if episode_done:
            logging.info(f"{world}'s episode is done. Finish this world.")
            break
        i += 1

        batch_observations = world.batch_observe(1, filtered_batch_actions, 0)
        world.batch_observe(1, filtered_batch_actions, 1)
        context = [obs['full_text'] for obs in batch_observations]
        label = [obs['labels'][0] if 'labels' in obs else obs['eval_labels'][0]
                 for obs in batch_observations]
        contexts.extend(context)
        labels.extend(label)
    world.reset()

    p_texts = []
    p_resps = []
    for context, label in zip(contexts, labels):
        # generated_text: str
        p_text = context.split("\n") + [label]
        p_texts.append(p_text)
        p_resps.append([label])

    p_token_resps = get_tokenized_text(p_resps, tokenizer, max_text_length)
    with torch.no_grad():
        out = mauve.compute_mauve_single(p_tokens=p_token_resps,
                                         p_text=p_resps,
                                         featurize_model_name=opt["cluster_embedding_model_name"],
                                         device_id=0,
                                         num_buckets=num_buckets,
                                         max_text_length=max_text_length,
                                         verbose=False,
                                         batch_size=32,
                                         )
    pca, kmeans = out.pca, out.kmeans
    return pca, kmeans, p_texts, out.p_labels, out.p_hist


def get_tokenized_text(texts, tokenizer, max_text_length):
    token_texts = []
    for text in texts:
        tokenized_text = []
        for sen in text[:-1]:
            tokenized_text.extend(tokenizer.encode(sen))
            tokenized_text.append(tokenizer.eos_token_id)
        tokenized_text.extend(tokenizer.encode(text[-1]))
        token_text = torch.LongTensor(torch.LongTensor(tokenized_text))
        if len(token_text) > max_text_length:  # Left truncate
            token_text = token_text[-max_text_length:]
        token_texts.append(token_text)
    return token_texts


def get_metrics(out, p_texts, q_texts):
    out.p_entropy = stats.entropy(out.p_hist)
    out.q_entropy = stats.entropy(out.q_hist)
    # calculate dist-n
    dist1_p, word1_p = Dist1Calculator().calculate([p[-1] for p in p_texts])
    dist2_p, word2_p = Dist2Calculator().calculate([p[-1] for p in p_texts])
    dist3_p, word3_p = Dist3Calculator().calculate([p[-1] for p in p_texts])
    dist1_q, word1_q = Dist1Calculator().calculate([q[-1] for q in q_texts])
    dist2_q, word2_q = Dist2Calculator().calculate([q[-1] for q in q_texts])
    dist3_q, word3_q = Dist3Calculator().calculate([q[-1] for q in q_texts])
    out.dist = {
        "dist1_p": dist1_p,
        "dist2_p": dist2_p,
        "dist3_p": dist3_p,
        "dist1_q": dist1_q,
        "dist2_q": dist2_q,
        "dist3_q": dist3_q,
        "word1_p": word1_p,
        "word2_p": word2_p,
        "word3_p": word3_p,
        "word1_q": word1_q,
        "word2_q": word2_q,
        "word3_q": word3_q,
        "avg_p_len": float(word1_p) / dist1_p / len(p_texts),
        "avg_q_len": float(word1_q) / dist1_q / len(q_texts),
    }
    return out


def split_batch(batch, num_splits):
    full_text_vec = batch.full_text_vec
    label_vec = batch.label_vec
    label = batch.labels
    text_vec = batch.text_vec
    valid_indices = batch.valid_indices

    num_batch = len(batch.labels)
    assert num_batch % num_splits == 0
    small_bsz = num_batch // num_splits
    batches = []
    for i in range(num_splits):
        batches.append(
            Batch(text_vec=text_vec[i * small_bsz: (i + 1) * small_bsz],
                  label_vec=label_vec[i * small_bsz: (i + 1) * small_bsz],
                  full_text_vec=full_text_vec[i * small_bsz: (i + 1) * small_bsz],
                  label=label[i * small_bsz: (i + 1) * small_bsz],
                  valid_indices=valid_indices[i * small_bsz: (i + 1) * small_bsz],
                  batchsize=small_bsz)
        )
    return batches


def clustering(agent, opt, world, is_training, pca, kmeans, tokenizer, fixed_p_hist=None):
    """
    Create a new world for distribution matching.

    :param Agent agent:
        The model being trained.

    :param Opt opt:
        The global CLI opts.

    :param BatchWorld world:
        The world which provides training samples.
    """
    # 1. Given world provides samples which agent can consume and generate responses.
    # 2. Then calculate P-Q distribution distance and allocate weights for each generated instances.
    # 3. Finally, provide a new world using those instances.
    agent.model.eval()

    contexts, labels, generated_texts = [], [], []
    while len(contexts) < opt["dm_samples"]:
        batch_actions = world.batch_act(0, batch_observation=None)
        filtered_batch_actions = []
        episode_done = False
        for ba in batch_actions:
            if "text" not in ba:
                episode_done = True
                break
            filtered_batch_actions.append(ba)
        if episode_done:
            if not is_training:
                logging.info(f"{world}'s episode is done. Finish this world.")
                break
            else:
                logging.info(f"Skip this batch")
                continue
        batch_observations = world.batch_observe(1, filtered_batch_actions, 0)
        world.batch_observe(1, filtered_batch_actions, 1)
        batch = agent.batchify(batch_observations, sort=False).to('cuda' if not opt['no_cuda'] else 'cpu')

        context = [obs['full_text'] for obs in batch_observations]
        label = [obs['labels'][0] if 'labels' in obs else obs['eval_labels'][0]
                 for obs in batch_observations]

        if opt.get("split_batch", 1) > 1:
            assert not opt.get("not_using_short_samples", False)
            splitted_batches = split_batch(batch, opt["split_batch"])
            for batch in splitted_batches:
                beam_preds_scores, beams = agent._generate(batch,
                                                           agent.beam_size,
                                                           agent.label_truncate or 256,
                                                           prefix_tokens=agent.get_prefix_tokens(batch))
                preds, scores = zip(*beam_preds_scores)
                text = [agent._v2t(p) for p in preds]  # list which have batchsize number of instances
                generated_texts.extend(text)

        else:
            beam_preds_scores, beams = agent._generate(batch,
                                                       agent.beam_size,
                                                       agent.label_truncate or 256,
                                                       prefix_tokens=agent.get_prefix_tokens(batch))
            preds, scores = zip(*beam_preds_scores)
            text = [agent._v2t(p) for p in preds]  # list which have batchsize number of instances

            if opt.get("not_using_short_samples", False):
                idx = []
                for i, c in enumerate(context):
                    if len(c.split(" ")) > 5:
                        idx.append(i)
                context = [context[i] for i in idx]
                label = [label[i] for i in idx]
                text = [text[i] for i in idx]
            generated_texts.extend(text)

        contexts.extend(context)
        labels.extend(label)

    contexts = contexts[:opt["dm_samples"]]
    labels = labels[:opt["dm_samples"]]
    generated_texts = generated_texts[:opt["dm_samples"]]

    p_texts, q_texts = [], []
    for context, label, generated_text in zip(contexts, labels, generated_texts):
        # generated_text: str
        p_text = context.split("\n") + [label]
        q_text = context.split("\n") + [generated_text]
        p_texts.append(p_text)
        q_texts.append(q_text)

    num_clusters = opt["num_buckets"]
    p_resps = [t[-1:] for t in p_texts]
    p_token_texts = get_tokenized_text(p_resps, tokenizer, max_text_length=256)
    s = np.cumsum(pca.explained_variance_ratio_)
    idx = np.argmax(s >= 0.9)  # last index to consider
    p_features = mauve.get_features_from_input(
        None, p_token_texts, p_resps, opt["cluster_embedding_model_name"], 256,
        0, name="p", verbose=False, batch_size=32,
    )
    p_features = normalize(p_features, norm='l2', axis=1)
    p_features = pca.transform(p_features)[:, :idx + 1].astype(np.float32)
    _, p_labels = kmeans.index.search(p_features, 1)
    p_labels = p_labels.reshape(-1)
    p_bins = np.histogram(p_labels, bins=num_clusters,
                          range=[0, num_clusters], density=True)[0]
    p_hist = p_bins / p_bins.sum()

    q_resps = [t[-1:] for t in q_texts]
    q_token_texts = get_tokenized_text(q_resps, tokenizer, max_text_length=256)
    q_features = mauve.get_features_from_input(
        None, q_token_texts, q_resps, opt["cluster_embedding_model_name"], 256,
        0, name="q", verbose=False, batch_size=32,
    )
    q_features = normalize(q_features, norm='l2', axis=1)
    q_features = pca.transform(q_features)[:, :idx + 1].astype(np.float32)
    _, q_labels = kmeans.index.search(q_features, 1)
    q_labels = q_labels.reshape(-1)
    q_bins = np.histogram(q_labels, bins=num_clusters,
                          range=[0, num_clusters], density=True)[0]
    q_hist = q_bins / q_bins.sum()

    divergence_curve_discretization_size = 25
    mixture_weights = np.linspace(1e-6, 1 - 1e-6, divergence_curve_discretization_size)
    divergence_curve = get_divergence_curve_for_multinomials(p_hist, q_hist, mixture_weights, 5)
    x, y = divergence_curve.T
    idxs1 = np.argsort(x)
    idxs2 = np.argsort(y)

    mauve_score = 0.5 * (
        compute_area_under_curve(x[idxs1], y[idxs1]) +
        compute_area_under_curve(y[idxs2], x[idxs2])
    )

    out = SimpleNamespace(
        p_hist=p_hist,
        q_hist=q_hist,
        num_buckets=num_clusters,
        p_labels=p_labels,
        q_labels=q_labels,
        p_text=p_resps,
        q_text=q_resps,
        kmeans=kmeans,
        pca=pca,
        mauve=mauve_score,
    )
    out = get_metrics(out, p_resps, q_resps)
    assert fixed_p_hist is not None
    hist = (1.0 - fixed_p_hist) ** opt.get("balancing_coeff", 30.0)
    thres = 0.1
    instance_weights = np.array([0.0 for _ in range(len(p_labels))])
    head_class = []
    for i in range(len(q_hist)):
        if q_hist[i] > thres:
            head_class.append(i)
    gen_instance_weights = np.array([-1.0 if q_label in head_class else 0.0 for q_label in q_labels])

    return instance_weights, gen_instance_weights, out, p_texts, q_texts, hist


def cluster_without_generation(opt, p_hist):
    hist = (1.0 - p_hist) ** opt.get("balancing_coeff", 30.0)
    out = SimpleNamespace()
    return None, None, out, None, None, hist


class TrainDiversifyLoopBase(TrainLoop):
    """
    TrainLoop contains the core training loop logic.
    """

    def __init__(self, opt):
        super().__init__(opt)
        random.seed(42)
        np.random.seed(42)
        self.cluster_embedding_tokenizer = get_tokenizer(opt["cluster_embedding_model_name"])
        self.cluster_embedding_model = get_model(opt["cluster_embedding_model_name"],
                                                 self.cluster_embedding_tokenizer,
                                                 0)
        self.last_dm_epoch = 0
        self.dm_every_n_epochs = opt['dm_every_n_epochs']
        trainstats_suffix = '.trainstats'  # we might load training statistics from here
        if (
            opt['load_from_checkpoint']
            and opt.get('model_file')
            and PathManager.exists(opt['model_file'] + '.checkpoint')
        ):
            opt['init_model'] = opt['model_file'] + '.checkpoint'
            trainstats_suffix = '.checkpoint.trainstats'

        if opt.get('model_file') and PathManager.exists(
            opt['model_file'] + trainstats_suffix
        ):
            # looks like we were preempted. make sure we load up our total
            # training stats, etc
            with PathManager.open(opt['model_file'] + trainstats_suffix) as ts:
                obj = json.load(ts)
                self.last_dm_epoch = obj.get('last_dm_epoch', 0)
        self.pca = None
        self.kmeans = None

    def _save_train_stats(self, suffix=None):
        fn = self.opt['model_file']
        if suffix:
            fn += suffix
        fn += '.trainstats'
        with PathManager.open(fn, 'w') as f:
            json.dump(
                {
                    'parleys': self.parleys,
                    'train_time': self.train_time.time(),
                    'train_steps': self._train_steps,
                    'total_epochs': self._total_epochs,
                    'train_reports': self.train_reports,
                    'valid_reports': self.valid_reports,
                    'best_valid': self.best_valid,
                    'impatience': self.impatience,
                    'last_dm_epoch': self.last_dm_epoch,
                },
                f,
                indent=4,
            )

    def train_steps(self):
        """
        Core training loop.
        Yields a metrics dict with each log.
        """
        logging.info('training...')
        opt = self.opt
        world = self.world
        with world:
            while True:
                # do one example / batch of examples
                try:
                    world.parley()
                except StopTrainException as e:
                    logging.info(f"Stopping from {e}")
                    break

                self.parleys += 1
                self._train_steps = self.parleys // self.update_freq
                self._last_log_steps += 1 / self.update_freq

                # the following additionally updates self._total_epochs
                train_time, log_time, validate_time, save_time = self._get_time(world)
                # get the total training examples done, compute epochs
                exs_per_epoch = world.num_examples()
                self._total_exs = int(np.round(self._total_epochs * exs_per_epoch))

                # Run additional training to minimize type I and type II loss.
                if self._total_epochs - self.last_dm_epoch >= self.dm_every_n_epochs:
                    self.diversifying()
                    self.last_dm_epoch = self._total_epochs

                # check counters and timers
                if self._total_epochs >= self.max_num_epochs:
                    yield self.log()
                    logging.info(
                        f'num_epochs completed:{self.max_num_epochs} time elapsed:{train_time}s'
                    )
                    break
                if train_time > self.max_train_time:
                    logging.info(f'max_train_time elapsed:{train_time}s')
                    break
                if self._train_steps >= self.max_train_steps:
                    logging.info(
                        f'max_train_steps elapsed:{self._train_steps} '
                        f'time elapsed:{train_time}s'
                    )
                    break
                if (
                    log_time > self.log_every_n_secs
                    or self._last_log_steps >= self.log_every_n_steps
                ):
                    yield self.log()
                if (
                    validate_time > self.val_every_n_secs
                    or self._total_epochs - self.last_valid_epoch
                    >= self.val_every_n_epochs
                    or self._train_steps - self._last_valid_steps
                    >= self.val_every_n_steps
                ):
                    try:
                        # log before we validate
                        if self._last_log_steps:
                            yield self.log()
                        world.reset_metrics()
                        stop_training = self.validate()
                    except StopTrainException:
                        break
                    # reset the log time because we logged right before validating
                    self.log_time.reset()
                    self.last_valid_epoch = self._total_epochs
                    self._last_valid_steps = self._train_steps
                    if stop_training:
                        break
                    # make sure metrics are clean before we log
                    world.reset_metrics()
                if save_time > self.save_every_n_secs and opt.get('model_file'):
                    logging.info(
                        f"saving model checkpoint: {opt['model_file']}.checkpoint"
                    )
                    if opt['tensorboard_log'] and is_primary_worker():
                        self.tb_logger.flush()
                    self.save_model('.checkpoint')
                    self.save_time.reset()

        if not sync_object(self.saved):
            # save agent
            self.save_model()

        # there's a rare edge case where the we never saved the model, and we try
        # # to reload it. This sync_object ensures all workers wait for the primary
        # worker to finish flushing before loading from disk.
        sync_object(None)
        if opt.get('model_file'):
            # clean up all our memory, just to make sure we don't OOM on GPU when
            # reloading the world
            del world
            del self.world
            del self.agent
            del self.valid_worlds
            # reload best validation model
            self.agent = create_agent(opt)

    def diversifying(self, current_epoch=None):
        opt = self.opt
        if self.pca is None and self.kmeans is None:
            self.pca, self.kmeans, self.p_texts, self.p_labels, p_hist = prepare_clustering(
                opt, self.world, self.cluster_embedding_tokenizer, max_text_length=256,
                num_buckets="auto" if opt["num_buckets"] is None else opt["num_buckets"])
            if opt["use_fixed_hist"]:
                self.p_hist = p_hist
            else:
                self.p_hist = None

        if opt["dm_match_training"]:
            with torch.no_grad():
                (instance_weights, gen_instance_weights, output,
                 p_texts, q_texts, hist) = clustering(self.agent,
                                                      opt,
                                                      self.world,
                                                      is_training=True,
                                                      pca=self.pca,
                                                      kmeans=self.kmeans,
                                                      tokenizer=self.cluster_embedding_tokenizer,
                                                      fixed_p_hist=self.p_hist)
            batch_idx = 0
            batchsize = opt["batchsize"]
            contexts, responses, batch_weights = [], [], []
            while batch_idx * batchsize < len(self.p_texts):
                text = self.p_texts[batch_idx * batchsize:
                                    (batch_idx + 1) * batchsize]
                labels = self.p_labels[batch_idx * batchsize: (batch_idx + 1) * batchsize]
                instance_weight = np.array([hist[label] for label in labels])
                context, response = [], []
                for p in text:
                    context.append("\n".join(p[:-1]))
                    response.append(p[-1])

                batch_weight = []
                for iw in instance_weight:
                    batch_weight.append(iw)

                contexts.append(context)
                responses.append(response)
                batch_weights.append(batch_weight)
                batch_idx += 1

            batch_idx = 0
            if opt["use_generated_texts"]:
                batchsize = opt["batchsize"]
                while batch_idx * batchsize < len(q_texts):
                    q_text = q_texts[batch_idx * batchsize:
                                     (batch_idx + 1) * batchsize]
                    gen_instance_weight = gen_instance_weights[batch_idx * batchsize:
                                                               (batch_idx + 1) * batchsize]
                    context, response = [], []
                    for q in q_text:
                        context.append("\n".join(q[:-1]))
                        response.append(q[-1])

                    batch_weight = []
                    for giw in gen_instance_weight:
                        batch_weight.append(giw)

                    contexts.append(context)
                    responses.append(response)
                    batch_weights.append(batch_weight)
                    batch_idx += 1

            batch_idx = 0
            idxs = list(range(len(contexts)))
            random.shuffle(idxs)
            for idx in idxs:
                context, response, batch_weight = contexts[idx], responses[idx], batch_weights[idx]
                batch_idx = self._train_step(context, response, batch_weight, batch_idx, current_epoch=current_epoch)

        if self.valid_worlds is None:
            # we need to load the world now
            self.valid_worlds = load_eval_worlds(self.agent, opt, 'valid')

        for v_world in self.valid_worlds:
            v_world.reset()
            with torch.no_grad():
                *_, output, _, _, _ = clustering(self.agent,
                                                 opt,
                                                 v_world,
                                                 is_training=False,
                                                 pca=self.pca,
                                                 kmeans=self.kmeans,
                                                 tokenizer=self.cluster_embedding_tokenizer,
                                                 fixed_p_hist=self.p_hist)
            id = v_world.world.agents[0].id

            if opt['tensorboard_log'] and is_primary_worker():
                self.tb_logger.writer.add_scalar(f"dist-1/{id}/valid",
                                                 output.dist["dist1_q"],
                                                 global_step=self._train_steps)
                self.tb_logger.writer.add_scalar(f"dist-2/{id}/valid",
                                                 output.dist["dist2_q"],
                                                 global_step=self._train_steps)
                self.tb_logger.writer.add_scalar(f"dist-3/{id}/valid",
                                                 output.dist["dist3_q"],
                                                 global_step=self._train_steps)
                self.tb_logger.writer.add_scalar(f"p-entropy/{id}/valid",
                                                 output.p_entropy,
                                                 global_step=self._train_steps)
                self.tb_logger.writer.add_scalar(f"q-entropy/{id}/valid",
                                                 output.q_entropy,
                                                 global_step=self._train_steps)

    def validate(self):
        """
        Perform a validation run, checking whether we should stop training.

        :return: boolean indicating whether training should stop
        :rtype: bool
        """
        opt = self.opt

        if self.valid_worlds is None:
            # we need to load the world now
            self.valid_worlds = load_eval_worlds(self.agent, opt, 'valid')

        # run evaluation on valid set
        valid_report = self._run_eval(
            self.valid_worlds, opt, 'valid', opt['validation_max_exs']
        )
        v = dict_report(valid_report)
        v['train_time'] = self.train_time.time()
        v['parleys'] = self.parleys
        v['train_steps'] = self._train_steps
        v['total_exs'] = self._total_exs
        v['total_epochs'] = self._total_epochs
        self.valid_reports.append(v)
        # logging
        if opt['tensorboard_log'] and is_primary_worker():
            valid_report['total_exs'] = self._total_exs
            self.tb_logger.log_metrics('valid', self.parleys, valid_report)
            # flush on a validation
            self.tb_logger.flush()
        if opt['wandb_log'] and is_primary_worker():
            valid_report['total_exs'] = self._total_exs
            self.wb_logger.log_metrics('valid', self.parleys, valid_report)

        # send valid metrics to agent if the agent wants them
        if hasattr(self.agent, 'receive_metrics'):
            self.agent.receive_metrics(valid_report)

        # check which metric to look at
        new_valid = valid_report[opt['validation_metric']]

        if isinstance(new_valid, Metric):
            new_valid = new_valid.value()

        # check if this is the best validation so far
        if (
            self.best_valid is None
            or self.valid_optim * new_valid > self.valid_optim * self.best_valid
        ):
            logging.success(
                'new best {}: {:.4g}{}'.format(
                    opt['validation_metric'],
                    new_valid,
                    ' (previous best was {:.4g})'.format(self.best_valid)
                    if self.best_valid is not None
                    else '',
                )
            )
            self.best_valid = new_valid
            self.impatience = 0
            if opt.get('model_file'):
                logging.info(f"saving best valid model: {opt['model_file']}")
                self.save_model()
                self.saved = True
            if (
                opt['validation_metric_mode'] == 'max'
                and self.best_valid >= opt['validation_cutoff']
            ) or (
                opt['validation_metric_mode'] == 'min'
                and self.best_valid <= opt['validation_cutoff']
            ):
                logging.info('task solved! stopping.')
                return True
        else:
            self.impatience += 1
            logging.report(
                'did not beat best {}: {} impatience: {}'.format(
                    opt['validation_metric'], round(self.best_valid, 4), self.impatience
                )
            )
        self.validate_time.reset()

        # saving
        if opt.get('model_file') and opt.get('save_after_valid'):
            logging.info(f"saving model checkpoint: {opt['model_file']}.checkpoint")
            self.save_model(f'.{self._total_epochs:.2f}.checkpoint')

        # check if we are out of patience
        if (
            0 < opt['validation_patience'] <= self.impatience
        ):
            logging.info('ran out of patience! stopping training.')
            return True
        return False
