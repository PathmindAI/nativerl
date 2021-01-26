import logging
from math import sqrt

from ray.rllib.agents.registry import get_agent_class

from pathmind.distributions import register_freezing_distributions
from pathmind.utils import get_mock_env, write_file


def mc_rollout(episodes, agent, environment, step_tolerance=1000000):
    """
    Monte Carlo rollout. Currently set to return brief summary of rewards and metrics.

    :param episodes:
    :param agent:
    :param environment:
    :param step_tolerance:
    :return:
    """
    reward_list = []
    episode_count = -1
    while episode_count + 1 < episodes:
        episode_count += 1
        observation = environment.reset()
        step_count = 0
        episode_reward = 0
        done = False
        while not done:
            action = agent.compute_action(observation)
            obs, reward, done, info = environment.step(action)
            step_count += 1
            episode_reward += reward
            if step_count > step_tolerance:
                done = True
            if done:
                reward_list.append(episode_reward)

    mean_reward = float(sum(reward_list) / len(reward_list))
    std_reward = sqrt(sum((x - mean_reward)**2 for x in reward_list) / len(reward_list))
    range_of_rewards = max(reward_list) - min(reward_list)

    return mean_reward, std_reward, range_of_rewards


def freeze_trained_policy(env, trials, output_dir: str, algorithm: str, is_discrete: bool,
                          filter_tolerance: float = 0.85, mc_iterations: int = 100,
                          step_tolerance: int = 100_000_000):
    """Freeze the trained policy at several temperatures and pick the best ones.

    :param env: nativerl.Environment instance
    :param trials: The trials returned from a ray tune run.
    :param output_dir: output directory for logs
    :param algorithm: the Rllib algorithm used (defaults to "PPO")
    :param is_discrete: for continuous actions we currently skip this step.
    :param filter_tolerance: Used for removing low mean performance policies from the reliability selection pool.
    0.85 means policies will be at least 85% of the top performer.
    :param mc_iterations: Number of episodes by which to judge policy
    :param step_tolerance: Maximum allowed step count for each iteration of Monte Carlo
    :return:
    """
    if not is_discrete:
        logger = logging.getLogger('freezing')
        logger.warning('Freezing skipped. Only supported for models with discrete actions.')
        return

    temperature_list = temperature_list = ["icy", "cold", "cool", "vanilla", "warm", "hot"]

    register_freezing_distributions(env=env)

    best_trial = trials.get_best_trial(metric="episode_reward_mean", mode="max")
    config = trials.get_best_config(metric="episode_reward_mean", mode="max")
    checkpoint_path = trials.get_best_checkpoint(trial=best_trial, metric="episode_reward_mean", mode="max")

    mean_reward_dict = dict.fromkeys(temperature_list)
    std_reward_dict = dict.fromkeys(temperature_list)
    range_reward_dict = dict.fromkeys(temperature_list)

    trainer_class = get_agent_class('PPO')
    mock_env = get_mock_env(env)

# Naive freezing ---------------------------------------------------

    for temp in temperature_list:
        if temp != "vanilla":
            config['model'] =  {'custom_action_dist': temp}
        
        # Create trainer agent
        trainer = get_agent_class('PPO')
        agent = trainer(env=mock_env, config=config)
    
        # Restore from checkpoint
        agent.restore(checkpoint_path)
        agent.export_policy_model(f"{output_dir}/model/{temp}")
    
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


#    for temp in temperature_list:
#        if temp != "vanilla":
#            config['model'] = {'custom_action_dist': temp}
#
#        agent = trainer_class(env=mock_env, config=config)
#        agent.restore(checkpoint_path)
#
#        mean_reward_dict[temp], std_reward_dict[temp], range_reward_dict[temp] = \
#            mc_rollout(mc_iterations, agent, env, step_tolerance)
#
#    # Filter out policies with under (filter_tolerance*100)% of max mean reward
#    filtered_range_reward_dict = {temp: range_reward_dict[temp]
#                                  for temp in mean_reward_dict.keys()
#                                  if mean_reward_dict[temp] > filter_tolerance * max(mean_reward_dict.values())}
#
#    top_performing_temp = max(mean_reward_dict, key=lambda k: mean_reward_dict[k])
#    most_clustered_temp = min(range_reward_dict, key=lambda k: std_reward_dict[k])
#    most_reliable_temp = min(filtered_range_reward_dict, key=lambda k: filtered_range_reward_dict[k])
#
#    for temp in temperature_list:
#        if temp != "vanilla":
#            config['model'] = {'custom_action_dist': temp}
#        trainer_class = get_agent_class(algorithm)
#        agent = trainer_class(env=mock_env, config=config)
#        agent.restore(checkpoint_path)
#        if temp == top_performing_temp:
#            agent.export_policy_model(f"{output_dir}/model/{temp}-top-mean-reward")
#        if temp == most_clustered_temp:
#            agent.export_policy_model(f"{output_dir}/model/{temp}-most-clustered")
#        if temp == most_reliable_temp:
#            agent.export_policy_model(f"{output_dir}/model/{temp}-most-reliable")
#        else:
#            agent.export_policy_model(f"{output_dir}/model/{temp}")
#
#    # Write freezing completion report
#    message_list = [
#        f"Mean reward dict: {mean_reward_dict}",
#        f"Standard deviation reward dict: {std_reward_dict}",
#        f"Range reward dict: {range_reward_dict}",
#        f"Top performing reward temperature: {top_performing_temp}",
#        f"Filtered temperature list: {filtered_range_reward_dict}",
#        f"Most clustered reward temparature: {most_clustered_temp}",
#        f"Most reliable temperature (our policy of choice): {most_reliable_temp}"
#    ]
#    write_file(message_list, "FreezingCompletionReport.txt", output_dir, algorithm)
