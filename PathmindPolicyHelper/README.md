PathmindHelper
==============

Introduction
------------

This is the "PathmindHelper" palette item that users can import in their AnyLogic model to make them ready to be uploaded to the Pathmind Web App. It assumes they will get processed by NativeRL.


Required Software
-----------------

 * AnyLogic on Linux, Mac, or Windows


Build Instructions
------------------

 1. Run the build for NativeRL to get `../nativerl-policy/target/nativerl-policy-1.7.2-SNAPSHOT.jar`
 2. Launch AnyLogic and inside it:
    1. Open the `PathmindPolicyHelper.alp` model
    2. Make sure `../nativerl-policy/target/nativerl-policy-1.7.2-SNAPSHOT.jar` is found as a dependency
    3. Click on "Pathmind" in the Projects view
    4. Go to Exporting -> Export the Library -> Finish

By default, this outputs a `PathmindHelper.jar` file and a copy of its dependencies. We can further add to that JAR the files from `../nativerl-policy/target/nativerl-policy-1.7.2-SNAPSHOT.jar` to simplify the end user experience, but this also requires modifying the `library.xml` file manually to remove the dependency on the JAR file.
  * The `bundle.sh` script file automates this process and outputs the final archive to `target/PathmindHelper.jar`
  * We can also call `fixup.sh` instead to rename the JAR file for NativeRL Policy to `PathmindPolicy.jar` and fix up the class path in `PathmindHelper.jar` accordingly. This way, however, AnyLogic won't copy `PathmindPolicy.jar` or its content on export.


End User Workflow
-----------------

This is an overall end user workflow as reference about how the user experience is meant to be, in this case for the Traffic Light Phases example:

 1. Drag PathmindHelper from the palette to the Main Agent's drawing space
 2. Fill up Observations, Reward Variables, Actions, Action Masks, etc (among other options), like this:
    ```java
        class Observations {
            double obs[] = getObservation(false);
        }
        class Reward {
            double vars[] = getObservation(true);
        }
        class Actions {
            @Discrete(n = 2) long action;
            void doIt() { doAction(action); }
        }
        class ActionMasks {
            boolean[] mask = getMask();
        }
    ```
    * Here we're asking users to define *private* inner classes since AnyLogic doesn't offer any way to let them define *public* inner classes. **(This is something they need to fix.)**

 3. Export model via the dummy Simulation and upload to the Pathmind Web App
 4. Write code snippets like this in the web app:
    ```java
        CLASS_SNIPPET='
            int simCount = 0;
            String combinations[][] = {
                    {"constant_moderate", "constant_moderate"},
                    {"none_til_heavy_afternoon_peak", "constant_moderate"},
                    {"constant_moderate", "none_til_heavy_afternoon_peak"},
                    {"peak_afternoon", "peak_morning"},
                    {"peak_morning", "peak_afternoon"}
            };
        '

        RESET_SNIPPET='
            simCount++;
            agent.schedNameNS = combinations[simCount % combinations.length][0];
            agent.schedNameEW = combinations[simCount % combinations.length][1];
        '

        OBSERVATION_SNIPPET='
            out = in.obs;
        '

        REWARD_SNIPPET='
            double[] s0 = before.vars, s1 = after.vars;
            // change in forward + intersection delay
            double delay0 = s0[0] + s0[2] + s0[4] + s0[6] + s0[8];
            double delay1 = s1[0] + s1[2] + s1[4] + s1[6] + s1[8];
            reward = delay0 - delay1;
            if (delay0 > 0 || delay1 > 0) {
                reward /= Math.max(delay0, delay1);
            }
        '

        METRICS_SNIPPET='
            metrics = new double[] { agent.tisDS.getYMean() };
        '
    ```

 5. Perform training, etc (the web app doesn't need to do anything more than it's already doing for RLlib)
 6. Export and download policies back to AnyLogic
    * The web app here needs to take the `PolicyObservationFilter.class` file generated and compiled by NativeRL, which contains the `OBSERVATION_SNIPPET` and implements `ObservationFilter`, and bundle it in the zip file along with the TensorFlow SavedModel. Right now, because AnyLogic doesn't support *public* inner classes, we need to make do with *private* inner classes, and that involves a couple of ugly hacks, but it's workable for now. **(Again, this is something they need to fix!)**
 7. Run the Simulation as usual and everything just works!


End User Guide
--------------

Currently maintained on [Basecamp](https://3.basecamp.com/3684163/buckets/11875773/messages/2017431518).
