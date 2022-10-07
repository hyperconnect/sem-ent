import numpy as np
import torch

from parlai.core.agents import create_agent
from parlai.core.exceptions import StopTrainException
from parlai.core.script import ParlaiScript
from parlai.core.script import register_script
from parlai.scripts.train_diversify_base import TrainDiversifyLoopBase
from parlai.scripts.train_diversify_base import setup_args
from parlai.utils import logging
from parlai.utils.distributed import sync_object


class TrainDressLoop(TrainDiversifyLoopBase):
    """
    TrainLoop contains the core training loop logic.
    """

    def _train_step(self, context, response, batch_weight, batch_idx, current_epoch):
        batch_actions = []
        for ctx, resp in zip(context, response):
            batch_actions.append({"text": ctx, "labels": [resp], "episode_done": True})

        batch_observations = []
        for action in batch_actions:
            obs = self.agent.observe(action)
            batch_observations.append(obs)
            self.agent.self_observe(obs)
        batch = self.agent.batchify(batch_observations, sort=False).to(
            'cuda' if not self.opt['no_cuda'] else 'cpu')
        batch_weight = torch.FloatTensor(batch_weight).to(batch.label_vec.device)

        self.agent._cache_dummy_batch(batch)
        self.agent.model.train()
        self.agent.zero_grad()
        loss = self.agent.compute_loss_with_weight(batch, batch_weight)
        self.agent.backward(loss)
        self.agent.update_params()
        if batch_idx % 10 == 0:
            self.log()
        batch_idx += 1
        return batch_idx

    def train_steps(self):
        """
        Core training loop.

        Yields a metrics dict with each log.
        """
        logging.info('training...')
        opt = self.opt
        world = self.world
        self._total_epochs = self._preempted_epochs
        with world:
            while True:
                # do one example / batch of examples
                self.parleys += 1
                self._train_steps = self.parleys // self.update_freq
                self._last_log_steps += 1 / self.update_freq

                # the following additionally updates self._total_epochs
                # get the total training examples done, compute epochs
                exs_per_epoch = world.num_examples()
                self._total_exs = int(np.round(self._total_epochs * exs_per_epoch))

                self.diversifying()
                self._total_epochs += 1
                if (
                    self._last_log_steps >= self.log_every_n_steps
                ):
                    yield self.log()
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


@register_script('train_dress', aliases=['tdr'])
class TrainDressModel(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        self.train_loop = TrainDressLoop(self.opt)
        return self.train_loop.train()


if __name__ == '__main__':
    TrainDressModel.main()
