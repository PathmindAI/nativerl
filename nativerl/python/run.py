import random
import os
import fire

import numpy as np
import ray
from ray.tune import run, sample_from

from pathmind import get_loggers, write_completion_report, Stopper, get_scheduler, modify_anylogic_db_properties
from pathmind.environments import get_environment, get_gym_environment
from pathmind.models import get_custom_model
from pathmind.callbacks import get_callbacks
from pathmind.freezing import freeze_trained_policy


def main(environment: str,
         is_gym: bool = False,
         algorithm: str = 'PPO',
         output_dir: str = os.getcwd(),
         multi_agent: bool = True,
         max_memory_in_mb: int = 4096,
         num_cpus: int = 1,
         num_gpus: int = 0,
         num_workers: int = 1,
         num_hidden_layers: int = 2,
         num_hidden_nodes: int = 256,
         max_iterations: int = 500,
         max_time_in_sec: int = 43200,
         max_episodes: int = 50000,
         num_samples: int = 4,
         resume: bool = False,
         checkpoint_frequency: int = 50,
         debug_metrics: bool = False,
         user_log: bool = False,
         autoregressive: bool = False,
         episode_reward_range_th: float = 0.01,
         entropy_slope_th: float = 0.01,
         vf_loss_range_th: float = 0.1,
         value_pred_th: float = 0.01,
         action_masking: bool = False,
         freezing: bool = False,
         discrete: bool = True,
         ):
    """

    :param environment: The name of a subclass of "Environment" to use as environment for training.
    :param is_gym: if True, "environment" must be a gym environment.
    :param algorithm: The algorithm to use with RLlib for training and the PythonPolicyHelper.
    :param output_dir: The directory where to output the logs of RLlib.
    :param multi_agent: Indicates that we need multi-agent support with the Environment class provided.
    :param max_memory_in_mb: The maximum amount of memory in MB to use for Java environments.
    :param num_cpus: The number of CPU cores to let RLlib use during training.
    :param num_gpus: The number of GPUs to let RLlib use during training.
    :param num_workers: The number of parallel workers that RLlib should execute during training.
    :param num_hidden_layers: The number of hidden layers in the MLP to use for the learning model.
    :param num_hidden_nodes: The number of nodes per layer in the MLP to use for the learning model.
    :param max_iterations: The maximum number of training iterations as a stopping criterion.
    :param max_time_in_sec: Maximum amount of  time in seconds.
    :param max_episodes: Maximum number of episodes per trial.
    :param num_samples: Number of population-based training samples.
    :param resume: Resume training when AWS spot instance terminates.
    :param checkpoint_frequency: Periodic checkpointing to allow training to recover from AWS spot instance termination.
    :param debug_metrics: Indicates that we save raw metrics data to metrics_raw column in progress.csv.
    :param user_log: Reduce size of output log file.
    :param autoregressive: Whether to use auto-regressive models.
    :param episode_reward_range_th: Episode reward range threshold
    :param entropy_slope_th: Entropy slope threshold
    :param vf_loss_range_th: VF loss range threshold
    :param value_pred_th: value pred threshold
    :param action_masking: Whether to use action masking or not.
    :param freezing: Whether to use policy freezing or not
    :param discrete: Discrete vs continuous actions, defaults to True (i.e. discrete)

    :return: runs training for the given environment, with nativerl
    """

    jar_dir = os.getcwd()
    os.chdir(jar_dir)
    output_dir = os.path.abspath(output_dir)
    modify_anylogic_db_properties()

    if is_gym:
        env, env_creator = get_gym_environment(environment_name=environment)
    else:
        env = get_environment(
            jar_dir=jar_dir,
            is_multi_agent=multi_agent,
            environment_name=environment,
            max_memory_in_mb=max_memory_in_mb
        )
        env_creator = env

    env_instance = env_creator(env_config={})
    env_instance.max_steps = env_instance._max_episode_steps if hasattr(env_instance, "_max_episode_steps") \
        else 20000

    ray.init(log_to_driver=user_log, dashboard_host='127.0.0.1')

    model = get_custom_model(
        num_hidden_nodes=num_hidden_nodes,
        num_hidden_layers=num_hidden_layers,
        autoregressive=autoregressive,
        action_masking=action_masking,
        discrete=discrete
    )

    stopper = Stopper(
        output_dir=output_dir, algorithm=algorithm, max_iterations=max_iterations,
        max_time_in_sec=max_time_in_sec, max_episodes=max_episodes,
        episode_reward_range_th=episode_reward_range_th, entropy_slope_th=entropy_slope_th,
        vf_loss_range_th=vf_loss_range_th, value_pred_th=value_pred_th
    )

    callbacks = get_callbacks(debug_metrics, is_gym)
    scheduler = get_scheduler()
    loggers = get_loggers()

    config = {
        'env': env,
        'callbacks': callbacks,
        'num_gpus': num_gpus,
        'num_workers': num_workers,
        'num_cpus_per_worker': num_cpus,
        'model': model,
        'use_gae': True,
        'vf_loss_coeff': 1.0,
        'vf_clip_param': np.inf,
        'lambda': 0.95,
        'clip_param': 0.2,
        'lr': 1e-4,
        'entropy_coeff': 0.0,
        'num_sgd_iter': sample_from(lambda spec: random.choice([10, 20, 30])),
        'sgd_minibatch_size': sample_from(lambda spec: random.choice([128, 512, 2048])),
        'train_batch_size': sample_from(lambda spec: random.choice([4000, 8000, 12000])),
        'batch_mode': 'complete_episodes',  # Set rollout samples to episode length
        'horizon': env_instance.max_steps, # Set max steps per episode
        'no_done_at_end': multi_agent  # Disable "de-allocation" of agents for simplicity
    }

    trials = run(
        algorithm,
        scheduler=scheduler,
        num_samples=num_samples,
        stop=stopper.stop,
        loggers=loggers,
        config=config,
        local_dir=output_dir if output_dir else None,
        resume=resume,
        checkpoint_freq=checkpoint_frequency,
        checkpoint_at_end=True,
        max_failures=3,
        export_formats=['model'],
        queue_trials=True
    )

    write_completion_report(trials=trials, output_dir=output_dir, algorithm=algorithm)

    if freezing:
        freeze_trained_policy(env=env_instance, trials=trials, algorithm=algorithm,
                              output_dir=output_dir, is_discrete=discrete)

    ray.shutdown()


def from_config(config_file="./config.json"):
    """Run training from a config file
    :param config_file: JSON file with arguments as per "training" CLI.
    :return:
    """
    import json
    with open(config_file, 'r') as f:
        config_string = f.read()
        config = json.loads(config_string)
    return main(**config)


if __name__ == '__main__':
    fire.Fire({
        "training": main,
        "from_config": from_config
    })