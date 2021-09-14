from typing import Dict

import importlib
import ray
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks


def get_callback_function(callback_function_name):
    """Get callback function from a string interpreted as Python module
    :param callback_function_name: name of the python module and function as string
    :return: callback function
    """
    class_name = callback_function_name.split('.')[-1]
    module = callback_function_name.replace(f'.{class_name}', '')
    lib = importlib.import_module(module)
    return getattr(lib, class_name)


def get_callbacks(debug_metrics, use_reward_terms, is_gym):

    class Callbacks(DefaultCallbacks):
        def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv,
                             policies: Dict[str, Policy],
                             episode: MultiAgentEpisode, **kwargs):
            episode.hist_data["metrics_raw"] = []

        def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                           policies: Dict[str, Policy], episode: MultiAgentEpisode, **kwargs):
            if not is_gym:
                metrics = worker.env.getMetrics().tolist()
                if debug_metrics:
                    episode.hist_data["metrics_raw"] = metrics

                for i, val in enumerate(metrics):
                    episode.custom_metrics[f"metrics_{str(i)}"] = metrics[i]

                if use_reward_terms:
                    term_contributions = worker.env.getRewardTermContributions().tolist()
                    for i, val in enumerate(term_contributions):
                        episode.custom_metrics[f"metrics_term_{str(i)}"] = term_contributions[i]

        def on_train_result(self, trainer, result: dict, **kwargs):
            if not is_gym:
                results = ray.get(
                    [w.apply.remote(lambda worker: worker.env.getMetrics()) for w in trainer.workers.remote_workers()])

                if use_reward_terms:
                    period = trainer.config["env_config"]["reward_balance_period"]
                    num_reward_terms = trainer.config["env_config"]["num_reward_terms"]

                    if result["training_iteration"] % period == 0:
                        # First "num_reward_terms" amount of custom metrics will be reserved for raw reward term contributions
                        betas = [1.0 / result["custom_metrics"][f"metrics_term_{str(i)}_mean"]
                                 if result["custom_metrics"][f"metrics_term_{str(i)}_mean"] != 0.0
#                                 else 1.0
                                 else 0.0
                                 for i in range(num_reward_terms)]
                        for w in trainer.workers.remote_workers():
                            w.apply.remote(lambda worker: worker.env.updateBetas(betas))

                result["last_metrics"] = results[0].tolist() if results is not None and len(results) > 0 else -1

    return Callbacks
