from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import Analysis

def export_policy_from_checkpoint(experiment_dir: str, env: str)
    analysis = Analysis(exp_dir)
    agent = PPOTrainer(env)
    for trial in analysis.trials:
        checkpoint_path = analysis.get_best_checkpoint(trial, metric="episode_reward_mean", mode="max")
        agent.restore(checkpoint_path)
        export_dir = os.path.join(experiment_dir, trial)
        agent.export_policy_model(export_dir)
