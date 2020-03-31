import json
import os
import types
from ray.utils import binary_to_hex, hex_to_binary
from ray.tune.trial import Trial
import ray.cloudpickle as cloudpickle
import logging
import socket
from pathlib import Path

logger = logging.getLogger(__name__)

class _TuneFunctionEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, types.FunctionType):
            return self._to_cloudpickle(obj)
        try:
            return super(_TuneFunctionEncoder, self).default(obj)
        except Exception:
            logger.debug("Unable to encode. Falling back to cloudpickle.")
            return self._to_cloudpickle(obj)

    def _to_cloudpickle(self, obj):
        return {
            "_type": "CLOUDPICKLE_FALLBACK",
            "value": binary_to_hex(cloudpickle.dumps(obj))
        }


class _TuneFunctionDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(
            self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if obj.get("_type") == "CLOUDPICKLE_FALLBACK":
            return self._from_cloudpickle(obj)
        return obj

    def _from_cloudpickle(self, obj):
        return cloudpickle.loads(hex_to_binary(obj["value"]))

def _find_newest_ckpt(ckpt_dir):
    """Returns path to most recently modified checkpoint."""
    full_paths = [
        os.path.join(ckpt_dir, fname) for fname in os.listdir(ckpt_dir)
        if fname.startswith("experiment_state") and fname.endswith(".json")
    ]
    return max(full_paths)

def _get_host():
    return os.getenv("NODE_IP", socket.gethostbyname(socket.gethostname()))

def update_checkpoint_node_info(newest_ckpt_path):
    with open(newest_ckpt_path, "r") as f:
        runner_state = json.load(f, cls=_TuneFunctionDecoder)
        checkpoint_file = newest_ckpt_path

    trial_states = {}
    for trial_cp in runner_state["checkpoints"]:
        trial = Trial(trial_cp["trainable_name"])

        # save the current status of trial since trial.__setstate__ change this (RUNNING -> PENDING)
        trial_status = trial_cp["status"]

        trial.__setstate__(trial_cp)

        # change ip for trials that has checkpoint
        if trial._checkpoint.value is not None:
            trial._checkpoint.last_result["hostname"] = socket.gethostname()
            trial._checkpoint.last_result["node_ip"] = _get_host()

        trial_state = trial.__getstate__()
        trial_state["status"] = trial_status 
        trial_states[trial.trial_id] = trial_state

    runner_state["checkpoints"] = list(trial_states.values())

    tmp_file_name = os.path.join(str(Path(newest_ckpt_path).parent),
                                     ".tmp_checkpoint")
    with open(tmp_file_name, "w") as f:
        json.dump(runner_state, f, indent=2, cls=_TuneFunctionEncoder)

    os.rename(tmp_file_name, checkpoint_file)

def update_existing_output_node_info(newest_ckpt_path):
    with open(newest_ckpt_path, "r") as f:
        runner_state = json.load(f)
    if len(runner_state["checkpoints"]) == 0:
        return False

    for cp in runner_state["checkpoints"]:
        if cp["last_result"] is not None and "node_ip" in cp["last_result"]:
            previous_ip = cp["last_result"]["node_ip"]
            previous_host_name = cp["last_result"]["hostname"]
            os.system("find ./* -type f -exec sed -i 's/" + previous_ip + "/" + _get_host() + "/g' {} \;")
            break
    

local_checkpoint_dir = os.path.join(os.getcwd(), "PPO")
newest_ckpt_path = _find_newest_ckpt(local_checkpoint_dir)

if os.path.exists(newest_ckpt_path):
    update_existing_output_node_info(newest_ckpt_path)
    update_checkpoint_node_info(newest_ckpt_path)
    print("updating node information for resume training is done")
else:
    print(local_checkpoint_dir + " doesn't have experiment_stat_xxx.json file")
