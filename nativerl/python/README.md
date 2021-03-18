# Pathmind RL Python package

This is the Python package running all RL tasks in the Pathmind backend. You
can use it to run exported AnyLogic experiments, any OpenAI gym environment or
even OR gym environments. This package is a drop-in replacement for the output
of the old `RLlibHelper`, which generated a file called `rllibtrain.py`. This
module is more flexible, modular, and extensible and doesn't need any code
generation. Notably, it comes with a pure Python implementation of the
nativerl interface, which allows you to run quick tests without bothering
with an elaborated setup. Just tweak the RL experiments you care about and
start a test.

## Installation

We might properly package this library at some point and maybe put it on a
private PyPI server for installation. For now, there's nothing to install,
as we treat the main `run.py` tool as a simple script which accesses dependencies
from the `pathmind/` folder. To use this library, e.g., for AnyLogic models,
simply make sure to copy `run.py` and `pathmind/` to where ever your nativerl
JARs and models reside (the same spot your `rllibtrain.py` script would have
been).

## Usage

The `run.py` script exposes a simple command line interface with two main
commands, namely `training` and `from_config`. The former works with the same
command line arguments that we had before and has just one required, positional
argument `environment`. This environment is either the full qualifier of your
AnyLogic model, i.e. package name and model name, or a gym (or OR gym) environment
referenced either by name or relative Python package import. To give an
example, you can run a gym cart pole example like this:

```shell
USE_PY_NATIVERL=True python run.py training CartPole-v0 --is-gym  --freezing --max-episodes 1
```

There are a few things of note here. First, to use pure Python you use
the `USE_PY_NATIVERL` environment variable, which can also be exported if
need be (`export USE_PY_NATIVERL=True`). Second, this script works with any
built-in OpenAI gym environment. To let the tool know we're dealing with a
gym environment, we're adding the `--is_gym` flag. Also, note that we freeze
the trained policy at various temperatures with the `freezing` flag and set
the `max-episodes` parameter to `1` to get quick feedback.


After setting `USE_PY_NATIVERL`
to `True`, the same can be achieved by running the code example of the
cart pole in the `tests` folder like this:

```shell
python run.py training tests.gym_cartpole.CartPoleEnv --is-gym
```

A third way of running a cart pole example is using a Python implementation
of the nativerl interface for the above gym example, like so:

```shell
python run.py training tests.cartpole.PathmindEnvironment
```

This last one does not need a `--is_gym` flag, because it's a nativerl
`Environment`.

If you have a configuration JSON file ready, you can achieve the same thing
by running

```shell
python run.py from_config config.json
```

This latter way is likely better long-term, at least for experimentation purposes,
as it frees you from setting environment variables like a pedestrian.

If you want to register custom callbacks for more advanced modifications, e.g., elaborate
curriculum learning scenarios, use the `custom-callback` flag to register a callback like
this:

```shell
python run.py training CartPole-v0 --is_gym --max_episodes=1 --freezing --custom-callback tests.custom_callback.get_callback
```

### Lagor PoC

We added the fairly advanced LPoC factory model as a test case to this module, which
comes with four basic environments. Here are examples to run them:

```shell
python run.py training tests.factory.environments.FactoryEnv --is_gym --max_episodes=1
python run.py training tests.factory.environments.TupleFactoryEnv --is_gym --max_episodes=1
python run.py training tests.factory.environments.RoundRobinFactoryEnv --is_gym --max_episodes=1
python run.py training tests.factory.environments.MultiAgentFactoryEnv --is_gym --max_episodes=1
```

The configuration of the underlying factory can be modified in `tests/factory/config.yml`.
It should be interesting to use this non-trivial case for faster prototyping, e.g. for
evaluating action masking or using autoregression.

## Help

The auto-generated help for this module is available via

```shell
python run.py training --help
```

which shows you all input arguments in detail. You can also access
`python run.py --help` for general help and `python run.py from_config --help`
for help with the "from configuration" trainer.

Here's a snapshot of the current `training` help page, to give you an overview:

```text
NAME
    run.py training

SYNOPSIS
    run.py training ENVIRONMENT <flags>

POSITIONAL ARGUMENTS
    ENVIRONMENT
        The name of a subclass of "Environment" to use as environment for training.

FLAGS
    --is_gym=IS_GYM
        if True, "environment" must be a gym environment.
    --algorithm=ALGORITHM
        The algorithm to use with RLlib for training and the PythonPolicyHelper.
    --scheduler=SCHEDULER
        The tune scheduler used for picking trials, currently supports "PBT" (and "PB2", once we upgrade to at least ray==1.0.1.post1)
    --output_dir=OUTPUT_DIR
        The directory where to output the logs of RLlib.
    --multi_agent=MULTI_AGENT
        Indicates that we need multi-agent support with the Environment class provided.
    --max_memory_in_mb=MAX_MEMORY_IN_MB
        The maximum amount of memory in MB to use for Java environments.
    --num_cpus=NUM_CPUS
        The number of CPU cores to let RLlib use during training.
    --num_gpus=NUM_GPUS
        The number of GPUs to let RLlib use during training.
    --num_workers=NUM_WORKERS
        The number of parallel workers that RLlib should execute during training.
    --num_hidden_layers=NUM_HIDDEN_LAYERS
        The number of hidden layers in the MLP to use for the learning model.
    --num_hidden_nodes=NUM_HIDDEN_NODES
        The number of nodes per layer in the MLP to use for the learning model.
    --max_iterations=MAX_ITERATIONS
        The maximum number of training iterations as a stopping criterion.
    --max_time_in_sec=MAX_TIME_IN_SEC
        Maximum amount of  time in seconds.
    --max_episodes=MAX_EPISODES
        Maximum number of episodes per trial.
    --num_samples=NUM_SAMPLES
        Number of population-based training samples.
    --resume=RESUME
        Resume training when AWS spot instance terminates.
    --checkpoint_frequency=CHECKPOINT_FREQUENCY
        Periodic checkpointing to allow training to recover from AWS spot instance termination.
    --debug_metrics=DEBUG_METRICS
        Indicates that we save raw metrics data to metrics_raw column in progress.csv.
    --user_log=USER_LOG
        Reduce size of output log file.
    --autoregressive=AUTOREGRESSIVE
        Whether to use auto-regressive models.
    --episode_reward_range=EPISODE_REWARD_RANGE
        Episode reward range threshold
    --entropy_slope=ENTROPY_SLOPE
        Entropy slope threshold
    --vf_loss_range=VF_LOSS_RANGE
        VF loss range threshold
    --value_pred=VALUE_PRED
        value pred threshold
    --action_masking=ACTION_MASKING
        Whether to use action masking or not.
    --freezing=FREEZING
        Whether to use policy freezing or not
    --discrete=DISCRETE
        Discrete vs continuous actions, defaults to True (i.e. discrete)
    --random_seed=RANDOM_SEED
        Optional random seed for this experiment.
    --custom_callback=CUSTOM_CALLBACK
        Optional name of a custom Python function returning a callback implementation of Ray's "DefaultCallbacks", e.g. "tests.custom_callback.get_callback"

NOTES
    You can also use flags syntax for POSITIONAL ARGUMENTS
```

## Tests

Make sure to install `pytest` first and then run

```shell
python -m pytest -m "not integration"
```

to run all non-integration tests. To run all tests, simply drop the last argument.