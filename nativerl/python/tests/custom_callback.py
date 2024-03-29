from typing import Dict

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy


def get_callback():
    class Callbacks(DefaultCallbacks):
        def on_episode_start(
            self,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: MultiAgentEpisode,
            **kwargs
        ):
            pass

        def on_episode_end(
            self,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: MultiAgentEpisode,
            **kwargs
        ):
            pass

        def on_train_result(self, trainer, result: dict, **kwargs):
            pass

    return Callbacks
