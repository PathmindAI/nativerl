import os
import shutil

from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import Analysis

def export_policy_from_checkpoint(experiment_dir: str, trainer):
    # Get best trial directory
    analysis = Analysis(experiment_dir, default_metric="episode_reward_mean", default_mode="max")
    best_trial_logdir = analysis.get_best_logdir()
    # Save policy in best trial directory
    export_dir = os.path.join(best_trial_logdir, "checkpoint_model")
    # If directory exists, remove it
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    # Generate policy
    trainer.export_policy_model(export_dir)
    # Move best policy to experiment root directory
    checkpoint_model_dir = os.path.join(os.pardir, "checkpoint_model")
    if os.path.exists(checkpoint_model_dir):
        shutil.rmtree(checkpoint_model_dir)
    shutil.move(export_dir, os.pardir)
