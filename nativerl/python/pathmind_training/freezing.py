import logging
from ray.rllib.agents.registry import get_agent_class
from ray.tune import run

from pathmind.distributions import register_freezing_distributions
from pathmind.utils import write_file


def find(key, value):
    for k, v in value.items():
        if k == key:
            yield v
        elif isinstance(v, dict):
            for result in find(key, v):
                yield result
        elif isinstance(v, list):
            for d in v:
                for result in find(key, d):
                    yield result


def mc_rollout(steps, checkpoint, environment, env_name, callbacks, output_dir, input_config,
               step_tolerance=1000000, algorithm='PPO'):
    """
    Monte Carlo rollout. Currently set to return brief summary of rewards and metrics.
    """

    config = {
        'env': env_name,
        'callbacks': callbacks,
        'num_gpus': 0,
        'num_workers': 6,
        'num_cpus_per_worker': 1,
        'model': input_config['model'],
        'lr': 0.0,
        'num_sgd_iter': 1,
        'sgd_minibatch_size': 1,
        'train_batch_size': steps,
        'batch_mode': 'complete_episodes',  # Set rollout samples to episode length
        'horizon': environment.max_steps,  # Set max steps per episode
    }

    trials = run(
        algorithm,
        num_samples=1,
        stop={'training_iteration': 1},
        config=config,
        local_dir=output_dir,
        restore=checkpoint,
        max_failures=10,
    )

    max_reward = next(find('episode_reward_max', trials.results))
    min_reward = next(find('episode_reward_min', trials.results))

    range_of_rewards = max_reward - min_reward
    mean_reward = next(find('episode_reward_mean', trials.results))

    return mean_reward, range_of_rewards


def freeze_trained_policy(env, env_name, callbacks, trials, output_dir: str, algorithm: str, is_discrete: bool,
                          filter_tolerance: float = 0.85, mc_steps: int = 10000,
                          step_tolerance: int = 100_000_000):
    """Freeze the trained policy at several temperatures and pick the best ones.

    :param env: nativerl.Environment instance
    :param env_name: name of the env (str)
    :param callbacks: ray Callbacks class
    :param trials: The trials returned from a ray tune run.
    :param output_dir: output directory for logs
    :param algorithm: the Rllib algorithm used (defaults to "PPO")
    :param is_discrete: for continuous actions we currently skip this step.
    :param filter_tolerance: Used for removing low mean performance policies from the reliability selection pool.
    0.85 means policies will be at least 85% of the top performer.
    :param mc_steps: Number of steps by which to judge policy
    :param step_tolerance: Maximum allowed step count for each iteration of Monte Carlo
    :return:
    """
    if not is_discrete:
        logger = logging.getLogger('freezing')
        logger.warning('Freezing skipped. Only supported for models with discrete actions.')
        return

    temperature_list = ["icy", "cold", "cool", "vanilla", "warm", "hot"]

    register_freezing_distributions(env=env)

    best_trial = trials.get_best_trial(metric="episode_reward_mean", mode="max")
    config = trials.get_best_config(metric="episode_reward_mean", mode="max")
    checkpoint_path = trials.get_best_checkpoint(trial=best_trial, metric="episode_reward_mean", mode="max")

    mean_reward_dict = dict.fromkeys(temperature_list)
    range_reward_dict = dict.fromkeys(temperature_list)

    for temp in temperature_list:
        if temp != "vanilla":
            config['model'] = {'custom_action_dist': temp}

        mean_reward_dict[temp], range_reward_dict[temp] = \
            mc_rollout(mc_steps, checkpoint_path, env, env_name, callbacks, output_dir, config,
                       step_tolerance, algorithm)

    # Filter out policies with under (filter_tolerance*100)% of max mean reward
    filter_tolerance = filter_tolerance if max(mean_reward_dict.values()) > 0 else 1. / filter_tolerance
    filtered_range_reward_dict = {temp: range_reward_dict[temp]
                                  for temp in mean_reward_dict.keys()
                                  if mean_reward_dict[temp] > filter_tolerance * max(mean_reward_dict.values())}

    top_performing_temp = max(mean_reward_dict, key=lambda k: mean_reward_dict[k])
    most_reliable_temp = min(filtered_range_reward_dict, key=lambda k: filtered_range_reward_dict[k])

    for temp in temperature_list:
        if temp != "vanilla":
            config['model'] = {'custom_action_dist': temp}
        trainer_class = get_agent_class(algorithm)
        agent = trainer_class(env=config['env'], config=config)
        agent.restore(checkpoint_path)
        if temp == top_performing_temp:
            agent.export_policy_model(f"{output_dir}/models/{temp}-top-mean-reward")
        if temp == most_reliable_temp:
            agent.export_policy_model(f"{output_dir}/models/{temp}-most-reliable")
            agent.export_policy_model(f"{output_dir}/model")
        else:
            agent.export_policy_model(f"{output_dir}/models/{temp}")

    # Write freezing completion report
    message_list = [
        f"Mean reward dict: {mean_reward_dict}",
        f"Range reward dict: {range_reward_dict}",
        f"Top performing reward temperature: {top_performing_temp}",
        f"Filtered temperature list: {filtered_range_reward_dict}",
        f"Most reliable temperature (our policy of choice): {most_reliable_temp}"
    ]
    write_file(message_list, "FreezingCompletionReport.txt", output_dir, algorithm)
