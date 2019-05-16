import java.io.File;
import java.util.Arrays;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import nativerl.*;

import com.anylogic.engine.Engine;
import traffic_light_opt.EmptyExperiment;

public class TrafficEnvironment extends Environment {
    final static EmptyExperiment ex = new EmptyExperiment(null);

    Engine engine;
    traffic_light_opt.Main root;
    int simCount = 0;
    String combinations[][] = {
            {"constant_moderate", "constant_moderate"},
            {"none_til_heavy_afternoon_peak", "constant_moderate"},
            {"constant_moderate", "none_til_heavy_afternoon_peak"},
            {"peak_afternoon", "peak_morning"},
            {"peak_morning", "peak_afternoon"}
    };

    Space actionSpace = new Discrete(2);
    Space observationSpace = new Continuous(
            new FloatVector(0, 0, 0, 0, 0, 0, 0, 0, 0, -1),
            new FloatVector(1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            new SSizeTVector());
    Array observation = new Array(new SSizeTVector().put(10));

    @Override public void close() {
        super.close();

        // Destroy the model:
        engine.stop();
    }

    @Override public Space getActionSpace() {
        return actionSpace;
    }

    @Override public Space getObservationSpace() {
        return observationSpace;
    }

    @Override public Array getObservation() {
        double[] state = root.getState();
        float[] normalized = new float[state.length];
        double totalDelay = Arrays.stream(state, 0, 9).sum();
        for (int i = 0; i < state.length; i++) {
            normalized[i] = (float)state[i];
            if (totalDelay > 0 && i < 9) {
                normalized[i] /= totalDelay;
            }
        }
        observation.data().put(normalized);
        return observation;
    }

    @Override public boolean isDone() {
        return engine.time() >= engine.getStopTime();
    }

    @Override public void reset() {
        simCount++;
        if (engine != null) {
            engine.stop();
        }
        // Create Engine, initialize random number generator:
        engine = ex.createEngine();
        // Fixed seed (reproducible simulation runs)
        engine.getDefaultRandomGenerator().setSeed(1);
        // Selection mode for simultaneous events:
        engine.setSimultaneousEventsSelectionMode(Engine.EVENT_SELECTION_LIFO);
        // Set stop time:
        engine.setStopTime(28800);
        // Create new root object:
        root = new traffic_light_opt.Main(engine, null, null);
//        root.policy = null;

        root.setParametersToDefaultValues();
        root.usePolicy = false;
        root.manual = false;
        root.schedNameNS = combinations[simCount % combinations.length][0];
        root.schedNameEW = combinations[simCount % combinations.length][1];

        engine.start(root);
	    /*
	    // Start simulation in fast mode:
	    engine.runFast();
	    // TODO Process results of simulation here
	    // traceln( "carSource_WE:" );
	    // traceln( inspectOf( root.carSource_WE ) );
	    // ...
	    */
    }

    @Override public float step(long action) {
        double[] s0 = root.getState();
        // {modified step function to ignore action if in yellow phase}
        root.step((int)action);
        engine.runFast(root.time() + 10);
        double[] s1 = root.getState();

        // change in forward + intersection delay
        double delay0 = s0[0] + s0[2] + s0[4] + s0[6] + s0[8];
        double delay1 = s1[0] + s1[2] + s1[4] + s1[6] + s1[8];
        double reward = delay0 - delay1;
        if (delay0 > 0 || delay1 > 0) {
            reward /= Math.max(delay0, delay1);
        }
        return (float)reward;
    }
}
