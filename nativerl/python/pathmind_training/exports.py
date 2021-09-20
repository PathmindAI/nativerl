from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import Analysis

import ipdb

def export_policy_from_checkpoint(experiment_dir: str, trainer):
    analysis = Analysis(experiment_dir, default_metric="episode_reward_mean", default_mode="max")
    trial_logdir = analysis.get_best_logdir()
    checkpoint_path = analysis.get_best_checkpoint(trial_logdir)
#    if checkpoint_path is None:
#        ipdb.set_trace(context=20)
    trainer.restore(checkpoint_path)
    export_dir = os.path.join(experiment_dir, checkpoint_path, os.pardir)
    trainer.export_policy_model(export_dir)
