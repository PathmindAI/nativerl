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
USE_PY_NATIVERL=True python run.py training CartPole-v0 --is_gym
```

There are a few things of note here. First, to use pure Python you use
the `USE_PY_NATIVERL` environment variable, which can also be exported if
need be (`export USE_PY_NATIVERL=True`). Second, this script works with any
built-in OpenAI gym environment. To let the tool know we're dealing with a
gym environment, we're adding the `--is_gym` flag. After setting `USE_PY_NATIVERL`
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