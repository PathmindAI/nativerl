# Pathmind RL Python package

This is the Python package running all RL tasks in the Pathmind backend. You
can use it to run exported AnyLogic experiments, any OpenAI gym environment or
even OR gym environments. This package is a drop-in replacement for the output
of the old `RLlibHelper`, which generated a file called `rllibtrain.py`. This
module is more flexible, modular, and extensible and doesn't need any code
generation. Notably, it comes with a pure Python implementation of the
`nativerl` interface called `pynativerl`, which allows you to run quick tests without bothering
with an elaborated setup. Just tweak the RL experiments you care about and
start a test.

## Training support

NativeRL training (run.py) accepts the following environments:

1. OpenAI gym.Env / or-gym referenced by name (python)
2. Pathmind Simulation (python)
3. AnyLogic (.jar file)

(1) remain as gym.Env

(2) and (3) get reformatted into rllib.env.MultiAgentEnv OR gym.Env, depending on whether def number_of_agents or "Number of Controlled Agents" (PathmindHelper) is a number greater than one.

## Installation

We might properly package this library at some point and maybe put it on a
private PyPI server for installation. For now, there's nothing to install,
as we treat the main `run.py` tool as a simple script which accesses dependencies
from the `pathmind_training/` folder. To use this library, e.g., for AnyLogic models,
simply make sure to copy `run.py` and `pathmind_training/` to where ever your nativerl
JARs and models reside (the same spot your `rllibtrain.py` script would have
been).

However, the `pathmind` package now exposes a `Simulation` interface that can
be run with `pynativerl`. To install that, simply run `python setup.py install`.

## Usage

The `run.py` script exposes a simple command line interface with two main
commands, namely `training` and `from_config`. The former works with the same
command line arguments that we had before and has just one required, positional
argument `environment`. This environment is either the full qualifier of your
AnyLogic model, i.e. package name and model name, or a gym (or OR gym) environment
referenced either by name or relative Python package import. To give an
example, you can run a gym cart pole example like this:

```shell
USE_PY_NATIVERL=True python run.py training CartPole-v0 --is_gym  --freezing --max_episodes 1
```

There are a few things of note here. First, to use pure Python you use
the `USE_PY_NATIVERL` environment variable, which can also be exported if
need be (`export USE_PY_NATIVERL=True`). Second, this script works with any
built-in OpenAI gym environment. To let the tool know we're dealing with a
gym environment, we're adding the `--is_gym` flag. Also, note that we freeze
the trained policy at various temperatures with the `freezing` flag and set
the `max_episodes` parameter to `1` to get quick feedback.

After setting `USE_PY_NATIVERL`
to `True`, the same can be achieved by running the code example of the
cart pole in the `tests` folder like this:

```shell
python run.py training tests.gym_cartpole.CartPoleEnv --is_gym
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
curriculum learning scenarios, use the `custom_callback` flag to register a callback like
this:

```shell
python run.py training CartPole-v0 --is_gym --max_episodes=1 --freezing --custom_callback tests.custom_callback.get_callback
```

### Running Pathmind Python simulations

To run training for simulations defined with the `pathmind` package, you need to

- specify your `Simulation` implementation as a Python package reference,
- provide the training script with the `--is_pathmind_simulation` flag,
- optionally provide the path to an observation selection YAML file with `--obs_selection`,
- and optionally provide a Python reference to your custom reward function with `--rew_fct_name`

Here's an example to run:

```shell
python run.py training tests.mouse.multi_mouse_env_pathmind.MultiMouseAndCheese \
--is_pathmind_simulation \
--obs_selection tests/mouse/obs.yaml \
--rew_fct_name tests.mouse.reward.reward_function
```

### Running Pathmind Python simulations

To run training for simulations defined with the `pathmind` package, you need to

- specify your `Simulation` implementation as a Python package reference,
- provide the training script with the `--is_pathmind_simulation` flag,
- optionally provide the path to an observation selection YAML file with `--obs_selection`,
- and optionally provide a Python reference to your custom reward function with `--rew_fct_name`

Here's an example to run:

```shell
python run.py training tests.mouse.multi_mouse_env_pathmind.MultiMouseAndCheese \
--is_pathmind_simulation \
--obs_selection tests/mouse/obs.yaml \
--rew_fct_name tests.mouse.reward.reward_function \
--multi_agent
```

Note that if no observation selection YAML is specified, all observations previously defined will be used, and if
no reward function is defined, the backend will simply sum up all reward terms.

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

### Reward balancing

If you have multi-objective rewards (or many terms due to reward shaping) and want to control the relative influences of each term,
then you may define "simulation.reward_weights: List[double]" along with "simulation.auto_norm_reward: bool"
while building your Pathmind simulation.

If you don't weight the reward terms but use auto-norm, then they'll be made to have equal influence on trainings.
If you provide reward term weights but do not use auto-norm, then your reward terms will be only multiplied by weight.

When you are not using a Pathmind Simulation OR are using one but haven't defined a these class attributes(see above),
then you can provide reward weights and whether to use auto-norm as "alphas" and "use_auto_norm" flags, respectively.

Option 1: Set up reward balancing inside Pathmind Simulation

```shell
python run.py training tests.mouse.two_reward_balance.TwoRewardMouseAndCheese --is_pathmind_simulation=True --max_episodes=1
```

Option 2: Pathmind Simulation (w/o reward balancing defined) but with reward balancing argument flags

```shell
python run.py training tests.mouse.two_reward_no_balance.TwoRewardMouseAndCheeseNoBalance --is_pathmind_simulation=True --alphas="1.0, 0.5" --num_reward_terms=2 --use_auto_norm=True --max_episodes=1
```

## Help

The auto-generated help for this module is available via

```shell
python run.py training --help
```

which shows you all input arguments in detail. You can also access
`python run.py --help` for general help and `python run.py from_config --help`
for help with the "from configuration" trainer.

## Tests

Make sure to install `pytest` first and then run

```shell
python -m pytest -m "not integration"
```

to run all non-integration tests. To run all tests, simply drop the last argument.
