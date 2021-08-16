NativeRL
========

Introduction
------------

This enables using environments for reinforcement learning implemented in Java or C++, with Python frameworks such as RLlib or Intel Coach.

It defines an intermediary C++ interface via the classes in `nativerl.h`, which are mapped to Java using JavaCPP and the classes in the `nativerl` submodule. The `nativerl::Environment` interface is meant to be subclassed by users to implement new environments, such as the ones outputted from code generated by `AnyLogicHelper.java`. The C++ classes are then made available to Python via pybind11 as per the bindings defined in `nativerl.cpp`, which one can use afterwards to implement environments as part of Python APIs, for example, OpenAI Gym or RLlib, as exemplified in code generated by `RLlibHelper.java`.


Required Software
-----------------

 * Linux, Mac, or Windows
 * Clang, GCC, or MSVC
   * On Windows, please also install MSYS2
 * CMake 3+  https://cmake.org/download/
 * JDK 8+
   * On Windows, make sure that `jvm.dll` can be found in the `PATH`.
 * Maven 3+  https://maven.apache.org/download.cgi
 * JavaCPP 1.5.1+  https://github.com/bytedeco/javacpp
 * Python 3.7+  https://www.python.org/downloads/
 * pybind11 2.2.4+  https://github.com/pybind/pybind11
 * RLlib  https://ray.readthedocs.io/en/latest/rllib.html


Build Instructions
------------------

 1. Install CMake, the JDK, Maven, and Python on the system
    * On Windows, from the "Visual Studio 2019" folder found inside the Start menu, open:
        - "x64 Native Tools Command Prompt for VS 2019" and run `c:\msys64\mingw64.exe` inside
        - Making sure the `MSYS2_PATH_TYPE=inherit` line is *not* commented out in `mingw64.ini` or `mingw32.ini`.
 2. Run `mvn clean install -Djavacpp.platform.custom -Djavacpp.platform.linux-x86_64 -Djavacpp.platform.macosx-x86_64 -Djavacpp.platform.windows-x86_64`
    * To build for TensorFlow 1.x, append `-Dtfv2=false` to that command.
 3. Find all output files inside the `nativerl/target/nativerl-1.7.0-SNAPSHOT-bin.zip` archive
    * This also produces `nativerl-policy/target/nativerl-policy-1.7.0-SNAPSHOT.jar` (~231mb) for the PathmindHelper

### Building with Docker

First build the image.

```bash
docker build . -t nativerl
```

Then start the container by mounting your current working directory.

```bash
docker run -v $HOME/.m2:/root/.m2 --mount "src=$(pwd),target=/app,type=bind" nativerl
```

After a successful build you'll find the results in the `target` folder. These instructions work
on Unix machines, on Windows you'll likely have to use `${PWD}` instead of `$(pwd)` in the `run` step.

### Sample Build Steps on CentOS 7 with Anaconda

```bash
sudo yum update
sudo yum install centos-release-scl
sudo yum install gcc-c++ cmake3 make java-1.8.0-openjdk-devel git wget devtoolset-7 rh-maven35
sudo ln -s /usr/bin/cmake3 /usr/bin/cmake

wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
bash Anaconda3-2019.03-Linux-x86_64.sh
conda install tensorflow
pip install ray[rllib]

scl enable devtoolset-7 rh-maven35 bash
mvn clean install -Djavacpp.platform=linux-x86_64
```

We can also package the Anaconda environment this way:

```
conda install -c conda-forge conda-pack
conda pack -o rllibpack.tar.gz
```

Once this is done, we can take all those archives and extract them on another machine ready for execution:

```
cd /path/to/rllibpack/
tar --totals -xf /path/to/rllibpack.tar.gz
source bin/activate
conda-unpack

cd /path/to/anylogic_model/
unzip -j nativerl-1.7.0-SNAPSHOT-bin.zip
```


### Running the Tests

 1. Follow the build instructions above
 2. Include in the `PATH` the directory containing the `anylogic` executable
    * The version of AnyLogic installed there needs to have PathmindHelper available in its Palette
 3. Inside the `nativerl-tests` subdirectory, run `mvn clean test`
    * We can also run the tests from the parent directory by appending `-Ptests`


Example Using RLlib and PathmindHelper for Traffic Light Phases
---------------------------------------------------------------

 1. Install RLlib, for example, `pip3 install --user psutil requests setproctitle tensorflow ray[rllib]`
    * Remove these lines from `~/.local/lib/python3.7/site-packages/ray/worker.py`, or else the JVM will crash:
    ```python
        # Enable nice stack traces on SIGSEGV etc.
        if not faulthandler.is_enabled():
            faulthandler.enable(all_threads=False)
    ```

 2. Inside AnyLogic:
    1. Add `PathmindHelper.jar` to the class path of the project, and fill up fields as per [End User WorkFlow](PathmindPolicyHelper/README.md#end-user-workflow)
    2. Create a new "Simulation" and adjust anything required
    3. Export the "Simulation" experiment to a "Standalone Java application" into some directory

 3. Extract the native libraries from `nativerl-1.7.0-SNAPSHOT-bin.zip` inside that directory
 4. Copy as well [`nativerl/examples/traintraffic.sh`](nativerl/examples/traintraffic.sh) into that directory
 5. Execute `bash traintraffic.sh` inside the directory and wait for training to complete
    * For a manually managed cluster, the sequence of operation is:
    1. On the "head node", execute `ray start --head --redis-port=6379`
    2. On other nodes, execute `ray start --redis-address 10.x.x.x:6379`
    3. Add `--redis-address 10.x.x.x:6379` option to `RLlibHelper` in `traintraffic.sh`, and increase `--num_workers` accordingly
    4. Execute `bash traintraffic.sh` on any node

 6. Once we get a checkpoint file, we can use it as a policy inside AnyLogic by loading it with PathmindHelper.


Example Using RLlib and Cartpole in Python
------------------------------------------

 1. Extract the native libraries from `nativerl-1.7.0-SNAPSHOT-bin.zip` somewhere
 2. Copy as well [`nativerl/examples/traincartpole.sh`](nativerl/examples/traincartpole.sh) into that directory
 3. Execute `bash traincartpole.sh` inside the directory and wait for training to complete
    * The script outputs the `cartpole.py` file that should actually be generated via some helper...


Support for Multiagent Environments
-----------------------------------

NativeRL's helpers currently implement only the simplest possible multiagent support available in RLlib, but it might just be what is needed in "99%" of the cases anyway, and it does allow us to have the "multiagent checkbox" ticked. Specifically, the policy optimized is shared among all agents, so they must be homogeneous enough to allow that. This corresponds to "level 1" described on this blog post: https://bair.berkeley.edu/blog/2018/12/12/rllib/

To use this basic level of support, we need to increase the "Number of Agents" value in the PathmindHelper, and make the values for "Observations", "Reward", "Actions", and "ActionMasks" depend on the `int agentId` argument that gets passed to them. This way, NativeRL can get 1 observation array per agent, 1 reward per agent, and can send 1 action per agent.
