package ai.skymind.nativerl;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;

public class AnyLogicHelper {

    String environmentClassName = "AnyLogicEnvironment";
    String agentClassName = "MainAgent";
    long discreteActions;
    long continuousActions;
    long continuousObservations;
    long randomSeed = 0;
    long stepTime = 1;
    long stopTime = 1000;
    String classSnippet = "";
    String resetSnippet = "";
    String rewardSnippet = "";

    String environmentClassName() {
        return environmentClassName;
    }
    AnyLogicHelper environmentClassName(String environmentClassName) {
        this.environmentClassName = environmentClassName;
        return this;
    }

    String agentClassName() {
        return agentClassName;
    }
    AnyLogicHelper agentClassName(String agentClassName) {
        this.agentClassName = agentClassName;
        return this;
    }

    long discreteActions() {
        return discreteActions;
    }
    AnyLogicHelper discreteActions(long discreteActions) {
        this.discreteActions = discreteActions;
        return this;
    }

    long continuousActions() {
        return continuousActions;
    }
    AnyLogicHelper continuousActions(long continuousActions) {
        this.continuousActions = continuousActions;
        return this;
    }

    long continuousObservations() {
        return continuousObservations;
    }
    AnyLogicHelper continuousObservations(long continuousObservations) {
        this.continuousObservations = continuousObservations;
        return this;
    }

    long randomSeed() {
        return randomSeed;
    }
    AnyLogicHelper randomSeed(long randomSeed) {
        this.randomSeed = randomSeed;
        return this;
    }

    long stepTime() {
        return stepTime;
    }
    AnyLogicHelper stepTime(long stepTime) {
        this.stepTime = stepTime;
        return this;
    }

    long stopTime() {
        return stopTime;
    }
    AnyLogicHelper stopTime(long stopTime) {
        this.stopTime = stopTime;
        return this;
    }

    String classSnippet() {
        return classSnippet;
    }
    AnyLogicHelper classSnippet(String classSnippet) {
        this.classSnippet = classSnippet;
        return this;
    }

    String resetSnippet() {
        return resetSnippet;
    }
    AnyLogicHelper resetSnippet(String resetSnippet) {
        this.resetSnippet = resetSnippet;
        return this;
    }

    String rewardSnippet() {
        return rewardSnippet;
    }
    AnyLogicHelper rewardSnippet(String rewardSnippet) {
        this.rewardSnippet = rewardSnippet;
        return this;
    }

    public void generateEnvironment(File file) throws IOException {
        File directory = file.getParentFile();
        if (directory != null) {
            directory.mkdirs();
        }
        Files.write(file.toPath(), generateEnvironment().getBytes());
    }

    public String generateEnvironment() {
        int n = environmentClassName.lastIndexOf(".");
        String className = environmentClassName.substring(n + 1);
        String packageName = n > 0 ? environmentClassName.substring(0, n) : null;

        String env = (packageName != null ? "package " + packageName + ";\n" : "")
            + "import ai.skymind.nativerl.*;\n"
            + "import com.anylogic.engine.*;\n"
            + "import java.util.Arrays;\n"
            + "\n"
            + "public class " + className + " extends AbstractEnvironment {\n"
            + "    final static Training experiment = new Training(null);\n"
            + "    protected Engine engine;\n"
            + "    protected " + agentClassName + " agent;\n"
            + "\n"
            + classSnippet
            + "\n"
            + "    public " + className + "() {\n"
            + "        super(" + discreteActions + ", " + continuousObservations + ");\n"
            + "        System.setProperty(\"ai.skymind.nativerl.disablePolicyHelper\", \"true\");\n"
            + "    }\n"
            + "\n"
            + "    @Override public void close() {\n"
            + "        super.close();\n"
            + "\n"
            + "        // Destroy the model:\n"
            + "        engine.stop();\n"
            + "    }\n"
            + "\n"
            + "    @Override public Array getObservation() {\n"
            + "        double[] state = agent.getObservation(false);\n"
            + "        float[] array = new float[state.length];\n"
            + "        for (int i = 0; i < state.length; i++) {\n"
            + "            array[i] = (float)state[i];\n"
            + "        }\n"
            + "        observation.data().put(array);\n"
            + "        return observation;\n"
            + "    }\n"
            + "\n"
            + "    @Override public boolean isDone() {\n"
            + "        return engine.time() >= engine.getStopTime();\n"
            + "    }\n"
            + "\n"
            + "    @Override public void reset() {\n"
            + "        if (engine != null) {\n"
            + "            engine.stop();\n"
            + "        }\n"
            + "        // Create Engine, initialize random number generator:\n"
            + "        engine = experiment.createEngine();\n"
            + "        // Fixed seed (reproducible simulation runs)\n"
            + "        engine.getDefaultRandomGenerator().setSeed(" + randomSeed + ");\n"
            + "        // Selection mode for simultaneous events:\n"
            + "        engine.setSimultaneousEventsSelectionMode(Engine.EVENT_SELECTION_LIFO);\n"
            + "        // Set stop time:\n"
            + "        engine.setStopTime(" + stopTime + ");\n"
            + "        // Create new agent object:\n"
            + "        agent = new " + agentClassName + "(engine, null, null);\n"
            + "        agent.setParametersToDefaultValues();\n"
            + "\n"
            + resetSnippet
            + "\n"
            + "        engine.start(agent);\n"
            + "    }\n"
            + "\n"
            + "    @Override public float step(long action) {\n"
            + "        double[] state0 = agent.getObservation(true);\n"
            + "        agent.doAction((int)action);\n"
            + "        engine.runFast(agent.time() + " + stepTime + ");\n"
            + "        double[] state1 = agent.getObservation(true);\n"
            + "\n"
            + rewardSnippet
            + "\n"
            + "        return (float)reward;\n"
            + "    }\n"
            + "}\n";
        return env;
    }

    public static void main(String[] args) throws Exception {
        AnyLogicHelper helper = new AnyLogicHelper();
        File output = null;
        for (int i = 0; i < args.length; i++) {
            if ("-help".equals(args[i]) || "--help".equals(args[i])) {
                System.out.println("usage: AnyLogicHelper [options] [output]");
                System.out.println();
                System.out.println("options:");
                System.out.println("    --environment-class-name");
                System.out.println("    --agent-class-name");
                System.out.println("    --discrete-actions");
                System.out.println("    --continuous-actions");
                System.out.println("    --continuous-observations");
                System.out.println("    --random-seed");
                System.out.println("    --class-snippet");
                System.out.println("    --reset-snippet");
                System.out.println("    --reward-snippet");
                System.exit(0);
            } else if ("--environment-class-name".equals(args[i])) {
                helper.environmentClassName(args[++i]);
            } else if ("--agent-class-name".equals(args[i])) {
                helper.agentClassName(args[++i]);
            } else if ("--discrete-actions".equals(args[i])) {
                helper.discreteActions(Integer.parseInt(args[++i]));
            } else if ("--continuous-actions".equals(args[i])) {
                helper.continuousActions(Integer.parseInt(args[++i]));
            } else if ("--continuous-observations".equals(args[i])) {
                helper.continuousObservations(Integer.parseInt(args[++i]));
            } else if ("--random-seed".equals(args[i])) {
                helper.randomSeed(Long.parseLong(args[++i]));
            } else if ("--step-time".equals(args[i])) {
                helper.stepTime(Long.parseLong(args[++i]));
            } else if ("--stop-time".equals(args[i])) {
                helper.stopTime(Long.parseLong(args[++i]));
            } else if ("--class-snippet".equals(args[i])) {
                helper.classSnippet(args[++i]);
            } else if ("--reset-snippet".equals(args[i])) {
                helper.resetSnippet(args[++i]);
            } else if ("--reward-snippet".equals(args[i])) {
                helper.rewardSnippet(args[++i]);
            } else {
                output = new File(args[i]);
            }
        }
        if (output == null) {
            output = new File(helper.environmentClassName.replace('.', '/') + ".java");
        }
        helper.generateEnvironment(output);
    }
}
