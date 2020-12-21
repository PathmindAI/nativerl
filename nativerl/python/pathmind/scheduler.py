import ray
from ray.tune.schedulers import PopulationBasedTraining
import numpy as np


def get_scheduler():
    return ray.tune.schedulers.PopulationBasedTraining(
        time_attr='training_iteration',
        metric='episode_reward_mean',
        mode='max',
        perturbation_interval=20,
        quantile_fraction=0.25,
        resample_probability=0.25,
        log_config=True,
        hyperparam_mutations={
            'lambda': np.linspace(0.9, 1.0, 5).tolist(),
            'clip_param': np.linspace(0.01, 0.5, 5).tolist(),
            'entropy_coeff': np.linspace(0, 0.03, 5).tolist(),
            'lr': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            'num_sgd_iter': [5, 10, 15, 20, 30],
            'sgd_minibatch_size': [128, 256, 512, 1024, 2048],
            'train_batch_size': [4000, 6000, 8000, 10000, 12000]
        }
    )
