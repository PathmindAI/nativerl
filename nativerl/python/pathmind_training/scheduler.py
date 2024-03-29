import numpy as np
from ray.tune.schedulers import PopulationBasedTraining


def get_scheduler(scheduler_name, train_batch_size=None):
    if scheduler_name == "PBT":
        return PopulationBasedTraining(
            time_attr="training_iteration",
            metric="episode_reward_mean",
            mode="max",
            perturbation_interval=20,
            quantile_fraction=0.25,
            resample_probability=0.25,
            log_config=True,
            hyperparam_mutations={
                "lambda": np.linspace(0.9, 1.0, 5).tolist(),
                "clip_param": np.linspace(0.01, 0.5, 5).tolist(),
                "entropy_coeff": np.linspace(0, 0.03, 5).tolist(),
                "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
                "num_sgd_iter": [5, 10, 15, 20, 30],
                "sgd_minibatch_size": [128, 256, 512, 1024, 2048],
                "train_batch_size": [train_batch_size]
                if train_batch_size
                else [4000, 6000, 8000, 10000, 12000],
            },
        )
    elif scheduler_name == "PB2":
        from ray.tune.schedulers.pb2 import PB2

        return PB2(
            time_attr="training_iteration",
            metric="episode_reward_mean",
            mode="max",
            perturbation_interval=20,
            quantile_fraction=0.25,
            log_config=True,
            hyperparam_bounds={
                "lambda": [0.9, 1.0],
                "clip_param": [0.01, 0.5],
                "entropy_coeff": [0, 5],
                "lr": [1e-3, 1e-5],
                "num_sgd_iter": [5, 30],
                "sgd_minibatch_size": [128, 2048],
                "train_batch_size": [train_batch_size]
                if train_batch_size
                else [4000, 12000],
            },
        )
    else:
        raise ValueError(f"{scheduler_name} not supported")
