import os
import shutil

def export_policy_from_checkpoint(trainer):
    # Save to experiment root directory
    checkpoint_model_dir = os.path.join(os.pardir, "checkpoint_model")
    # If model directory already exist, remove it first
    if os.path.exists(checkpoint_model_dir):
        shutil.rmtree(checkpoint_model_dir)
    # Generate policy
    trainer.export_policy_model(checkpoint_model_dir)
