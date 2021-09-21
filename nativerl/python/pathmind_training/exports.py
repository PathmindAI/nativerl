import os
import shutil

from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import Analysis

def export_policy_from_checkpoint(experiment_dir: str, trainer):
    analysis = Analysis(experiment_dir, default_metric="episode_reward_mean", default_mode="max")
    trial_logdir = analysis.get_best_logdir()
    checkpoint_path = analysis.get_best_checkpoint(trial_logdir)
    trainer.restore(checkpoint_path)
    export_dir = os.path.join(trial_logdir, "model")
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    trainer.export_policy_model(export_dir)
