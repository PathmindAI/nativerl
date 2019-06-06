NativeRL
========

Introduction
------------

This is a hack to allow using environments for reinforcement learning implemented in Java or C++, with Python frameworks such as RLlib or Intel Coach.

It defines an intermediary C++ interface via the classes in `nativerl.h`, which are mapped to Java using JavaCPP and the classes in the `nativerl` subdirectory. The `nativerl::Environment` class is meant to be subclassed by users to implement new environments, such as `TrafficEnvironment.java`. The C++ classes are then made available to Python via pybind11 as per the bindings defined in `nativerl.cpp`, which one can use afterwards to implement environments as part of Python APIs, for example, OpenAI Gym, as exemplified in `rllibtest.py`.


Required Software
-----------------

 * Linux, Mac, or Windows (untested)
 * Clang, GCC, or MSVC (untested)
 * JDK 8+
 * Maven 3+  https://maven.apache.org/download.cgi
 * JavaCPP 1.5.1+  https://github.com/bytedeco/javacpp
 * Python 3.7+  https://www.python.org/downloads/
 * pybind11 2.2.4+  https://github.com/pybind/pybind11
 * RLlib  https://ray.readthedocs.io/en/latest/rllib.html


Build Instructions
------------------

 1. Install the JDK, Maven, and Python on the system
 2. Install pybind11 with a command like `pip3 install --user pybind11`
 3. Run `mvn clean package` where the `pom.xml` file resides
 4. Find all output files inside the `target/nativerl-1.0.0-SNAPSHOT-bin.zip` archive


### Sample Build Steps on CentOS 7 with Anaconda

```bash
sudo yum update
sudo yum install centos-release-scl
sudo yum install gcc-c++ java-1.8.0-openjdk-devel git wget devtoolset-7 rh-maven35

wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
bash Anaconda3-2019.03-Linux-x86_64.sh
conda install pybind11 tensorflow
pip install ray[rllib]

scl enable devtoolset-7 rh-maven35 bash
mvn clean package -Djavacpp.platform=linux-x86_64
```


Example Using RLlib and TrafficEnvironment
------------------------------------------

 1. Install RLlib, for example, `pip3 install --user psutil requests setproctitle tensorflow ray[rllib]`
    * Remove these lines from `~/.local/lib/python3.7/site-packages/ray/worker.py`, or else the JVM will crash:
    ```python
        # Enable nice stack traces on SIGSEGV etc.
        if not faulthandler.is_enabled():
            faulthandler.enable(all_threads=False)
    ```

 2. Inside AnyLogic:
    1. Add `nativerl-1.0.0-SNAPSHOT.jar` to the class path of the project
    2. Create a new "Custom Experiment" named "Training" and erase all of its code
    3. Create a new "Java Class" named "TrafficEnvironment", copying all content from [`examples/TrafficEnvironment.java`](examples/TrafficEnvironment.java)
    4. Export the "Training" experiment to a "Standalone Java application" into some directory
    * To ensure we can execute multiple simulations in parallel, append this line to `database/db.properties`:
    ```
        hsqldb.lock_file=false
    ```

 3. Extract the native libraries from `nativerl-1.0.0-SNAPSHOT-bin.zip` inside that directory
 4. Copy as well [`examples/rllibtest.py`](examples/rllibtest.py) into that directory
 5. Execute `python3 rllibtest.py` inside the directory and wait for training to complete
    * For a manually managed cluster, the sequence of operation is:
    1. On the "head node", execute `ray start --head --redis-port=6379`
    2. On other nodes, execute `ray start --redis-address 10.x.x.x:6379`
    3. Replace `ray.init()` with `ray.init(redis_address="10.x.x.x:6379")` in `rllibtest.py` and increase `num_workers` accordingly
    4. Execute `python3 rllibtest.py` on any node

 6. Once we get a checkpoint file, we can use it as a policy inside AnyLogic with an event containing code like this:

    ```java
    // In the Main "Imports section"
    import ai.skymind.nativerl.*;

    // In the Main "Additional class code"
    RLlibHelper rllibHelper = null;

    // In the Event "Action"
    if (rllibHelper == null && usePolicy) {
        try {
            // Directories where to find all modules required by RLlib
            File[] path = {
                new File("/usr/lib64/python3.7/lib-dynload/"),
                new File("/usr/lib64/python3.7/site-packages/"),
                new File("/usr/lib/python3.7/site-packages/"),
                new File(System.getProperty("user.home") + "/.local/lib/python3.7/site-packages/")
            };
            File checkpoint = new File("/path/to/checkpoint_100/checkpoint-100");
            rllibHelper = new RLlibHelper(path, "PPO", checkpoint, "Traffic", 2, 10);
        } catch (IOException e) {
            traceln(e);
        }
    }

    if (rllibHelper != null && usePolicy) {
        float[] a = TrafficEnvironment.normalize(getState());
        int action = (int)rllibHelper.computeDiscreteAction(a);
        traceln("Action: " + action);
        step(action);
    }
    ```

