import importlib
from typing import Dict

import numpy as np
import ray
from pathmind_training.exports import export_policy_from_checkpoint
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy


def get_callback_function(callback_function_name):
    """Get callback function from a string interpreted as Python module
    :param callback_function_name: name of the python module and function as string
    :return: callback function
    """
    class_name = callback_function_name.split(".")[-1]
    module = callback_function_name.replace(f".{class_name}", "")
    lib = importlib.import_module(module)
    return getattr(lib, class_name)


def get_callbacks(debug_metrics, use_reward_terms, is_gym, checkpoint_frequency):
    class Callbacks(DefaultCallbacks):
        def on_episode_start(
            self,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: MultiAgentEpisode,
            **kwargs,
        ):
            episode.hist_data["metrics_raw"] = []

        def on_episode_end(
            self,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: MultiAgentEpisode,
            **kwargs,
        ):
            if not is_gym:
                metrics = worker.env.getMetrics().tolist()
                if debug_metrics:
                    episode.hist_data["metrics_raw"] = metrics

                for i, val in enumerate(metrics):
                    episode.custom_metrics[f"metrics_{str(i)}"] = metrics[i]

                if use_reward_terms:
                    term_contributions = (
                        worker.env.getRewardTermContributions().tolist()
                    )
                    betas = worker.env.betas
                    for i, val in enumerate(term_contributions):
                        episode.custom_metrics[
                            f"metrics_term_{str(i)}"
                        ] = term_contributions[i]
                    for i, val in enumerate(betas):
                        episode.custom_metrics[f"beta_{str(i)}"] = betas[i]

        def on_train_result(self, trainer, result: dict, **kwargs):
            if not is_gym:
                results = ray.get(
                    [
                        w.apply.remote(lambda worker: worker.env.getMetrics())
                        for w in trainer.workers.remote_workers()
                    ]
                )

                use_auto_norm = trainer.config["env_config"]["use_auto_norm"]

                if use_auto_norm:
                    period = trainer.config["env_config"]["reward_balance_period"]
                    num_reward_terms = trainer.config["env_config"]["num_reward_terms"]

                    if (
                        result["training_iteration"] % period == 0
                        or result["training_iteration"] == 1
                    ):

                        if result["training_iteration"] == 1:
                            lr = 1.0
                        else:
                            lr = max(
                                0.025,
                                min(
                                    0.4,
                                    20
                                    * round(period / result["training_iteration"], 4),
                                ),
                            )

                        betas = [
                            1.0
                            / max(
                                abs(
                                    result["custom_metrics"][
                                        f"metrics_term_{str(i)}_min"
                                    ]
                                ),
                                abs(
                                    result["custom_metrics"][
                                        f"metrics_term_{str(i)}_max"
                                    ]
                                ),
                            )
                            if max(
                                abs(
                                    result["custom_metrics"][
                                        f"metrics_term_{str(i)}_min"
                                    ]
                                ),
                                abs(
                                    result["custom_metrics"][
                                        f"metrics_term_{str(i)}_max"
                                    ]
                                ),
                            )
                            != 0.0
                            else 1.0
                            for i in range(num_reward_terms)
                        ]

                        for w in trainer.workers.remote_workers():
                            w.apply.remote(
                                lambda worker: worker.env.updateBetas(betas, lr=lr)
                            )
                        betas_list = ray.get(
                            [
                                w.apply.remote(lambda worker: worker.env.betas)
                                for w in trainer.workers.remote_workers()
                            ]
                        )
                        betas = np.array(betas_list)
                        with open("../betas.npy", "wb") as f:
                            np.save(f, betas)

                if (
                    result["training_iteration"] % checkpoint_frequency == 0
                    and result["training_iteration"] > 1
                ):
                    export_policy_from_checkpoint(trainer)

                result["last_metrics"] = (
                    results[0].tolist()
                    if results is not None and len(results) > 0
                    else -1
                )

    return Callbacks
