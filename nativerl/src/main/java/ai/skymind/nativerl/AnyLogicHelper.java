package ai.skymind.nativerl;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.file.Files;

public class AnyLogicHelper {

    String environmentClassName = "AnyLogicEnvironment";
    String agentClassName = "MainAgent";
    long discreteActions;
    long continuousActions;
    long continuousObservations;
    long stepTime = 1;
    long stopTime = 1000;
    String classSnippet = "";
    String resetSnippet = "";
    String rewardSnippet = "";
    String metricsSnippet = "";
    String policyHelper = null;
    int testIterations = 10;
    boolean stepless = false;
    boolean multiAgent = false;
    int actionTupleSize;

    public String environmentClassName() {
        return environmentClassName;
    }
    public AnyLogicHelper environmentClassName(String environmentClassName) {
        this.environmentClassName = environmentClassName;
        return this;
    }

    public String agentClassName() {
        return agentClassName;
    }
    public AnyLogicHelper agentClassName(String agentClassName) {
        this.agentClassName = agentClassName;
        return this;
    }

    public long discreteActions() {
        return discreteActions;
    }
    public AnyLogicHelper discreteActions(long discreteActions) {
        this.discreteActions = discreteActions;
        return this;
    }

    public long continuousActions() {
        return continuousActions;
    }
    public AnyLogicHelper continuousActions(long continuousActions) {
        this.continuousActions = continuousActions;
        return this;
    }

    public long continuousObservations() {
        return continuousObservations;
    }
    public AnyLogicHelper continuousObservations(long continuousObservations) {
        this.continuousObservations = continuousObservations;
        return this;
    }

    public long stepTime() {
        return stepTime;
    }
    public AnyLogicHelper stepTime(long stepTime) {
        this.stepTime = stepTime;
        return this;
    }

    public long stopTime() {
        return stopTime;
    }
    public AnyLogicHelper stopTime(long stopTime) {
        this.stopTime = stopTime;
        return this;
    }

    public String classSnippet() {
        return classSnippet;
    }
    public AnyLogicHelper classSnippet(String classSnippet) {
        this.classSnippet = classSnippet;
        return this;
    }

    public String resetSnippet() {
        return resetSnippet;
    }
    public AnyLogicHelper resetSnippet(String resetSnippet) {
        this.resetSnippet = resetSnippet;
        return this;
    }

    public String rewardSnippet() {
        return rewardSnippet;
    }
    public AnyLogicHelper rewardSnippet(String rewardSnippet) {
        this.rewardSnippet = rewardSnippet;
        return this;
    }

    public String metricsSnippet() {
        return metricsSnippet;
    }
    public AnyLogicHelper metricsSnippet(String metricsSnippet) {
        this.metricsSnippet = metricsSnippet;
        return this;
    }

    public String policyHelper() {
        return policyHelper;
    }
    public AnyLogicHelper policyHelper(String policyHelper) {
        this.policyHelper = policyHelper;
        return this;
    }

    public int testIterations() {
        return testIterations;
    }
    public AnyLogicHelper testIterations(int testIterations) {
        this.testIterations = testIterations;
        return this;
    }

    public boolean isStepless() {
        return stepless;
    }

    public void setStepless(boolean stepless) {
        this.stepless = stepless;
    }

    public boolean isMultiAgent() {
        return multiAgent;
    }
    public AnyLogicHelper setMultiAgent(boolean multiAgent) {
        this.multiAgent = multiAgent;
        return this;
    }

    public int actionTupleSize() {
        return actionTupleSize;
    }
    public AnyLogicHelper actionTupleSize(int actionTupleSize) {
        this.actionTupleSize = actionTupleSize;
        return this;
    }

    AnyLogicHelper checkAgentClass() throws ClassNotFoundException, NoSuchMethodException, NoSuchFieldException {
        int n = agentClassName.lastIndexOf(".");
        String className = agentClassName.substring(n + 1);
        String packageName = n > 0 ? agentClassName.substring(0, n) : null;

        Class.forName((packageName != null ? packageName + "." : "") + "Training");
        Class c = Class.forName(agentClassName);
        if (multiAgent) {
            c.getDeclaredMethod("doAction", int[].class);
        } else {
            c.getDeclaredMethod("doAction", int.class);
        }
        Method m = c.getDeclaredMethod("getObservation", boolean.class);
        if (multiAgent && m.getReturnType() != double[][].class) {
            throw new NoSuchMethodException(m + " must return " + double[][].class);
        } else if (!multiAgent && m.getReturnType() != double[].class) {
            throw new NoSuchMethodException(m + " must return " + double[].class);
        }
        Field f = c.getDeclaredField("policyHelper");
        if (f.getType() != PolicyHelper.class) {
            throw new NoSuchMethodException(f + " must be " + PolicyHelper.class);
        }
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
            + "import java.io.File;\n"
            + "import java.nio.charset.Charset;\n"
            + "import java.nio.file.Files;\n"
            + "import java.nio.file.Paths;\n"
            + "import java.util.ArrayList;\n"
            + "import java.util.Arrays;\n"
            + "import pathmind.policyhelper.PathmindHelperRegistry;\n"
            + "\n"
            + "public class " + className + " extends AbstractEnvironment {\n"
            + "    final static Training experiment = new Training(null);\n"
            + "    protected Engine engine;\n"
            + "    protected " + agentClassName + " agent;\n"
            + "    protected PolicyHelper policyHelper;\n"
            + "\n"
            + classSnippet
            + "\n"
            + "    public " + className + "() {\n"
            + "        super(" + discreteActions + ", " + continuousObservations + ");\n"
            + "        System.setProperty(\"ai.skymind.nativerl.disablePolicyHelper\", \"true\");\n"
            + "    }\n"
            + "\n"
            + "    public " + className + "(PolicyHelper policyHelper) {\n"
            + "        super(" + discreteActions + ", " + continuousObservations + ");\n"
            + "        this.policyHelper = policyHelper;\n"
            + "    }\n"
            + "\n"
            + "    @Override public void close() {\n"
            + "        super.close();\n"
            + "\n"
            + "        // Destroy the model:\n"
            + "        engine.stop();\n"
            + "    }\n"
            + "\n"
            + (multiAgent
                ? "    @Override public Array getObservation() {\n"
                + "        double[][] obs = PathmindHelperRegistry.getHelper().observationForTraining();\n"
                + "        int obssize = obs[0].length;\n"
                + "        float[] array = new float[obs.length * obssize];\n"
                + "        for (int i = 0; i < array.length; i++) {\n"
                + "            array[i] = (float)obs[i / obssize][i % obssize];\n"
                + "        }\n"
                + "        if (observation == null || observation.shape().size() < 2 || observation.length() != array.length) {\n"
                + "            observation = new Array(new SSizeTVector().put(new long[] {obs.length, obssize}));\n"
                + "        }\n"
                + "        observation.data().put(array);\n"
                + "        return observation;\n"
                + "    }\n"
                + "\n"

                : "    @Override public Array getObservation() {\n"
                + "        double[] obs = PathmindHelperRegistry.getHelper().observationForTraining();\n"
                + "        float[] array = new float[obs.length];\n"
                + "        for (int i = 0; i < obs.length; i++) {\n"
                + "            array[i] = (float)obs[i];\n"
                + "        }\n"
                + "        observation.data().put(array);\n"
                + "        return observation;\n"
                + "    }\n"
                + "\n")
            + "    @Override public boolean isDone() {\n"
            + "        return PathmindHelperRegistry.getHelper().isDone();\n"
            + "    }\n"
            + "\n"
            + "    @Override public void reset() {\n"
            + "        if (engine != null) {\n"
            + "            engine.stop();\n"
            + "        }\n"
            + "        // Create Engine, initialize random number generator:\n"
            + "        engine = experiment.createEngine();\n"
            + "        Simulation sim = new Simulation();\n"
            + "        sim.setupEngine(engine);\n"
            + "        sim.initDefaultRandomNumberGenerator(engine);\n"
            + "        // Create new agent object:\n"
            + "        agent = new " + agentClassName + "(engine, null, null);\n"
            + "        agent.setParametersToDefaultValues();\n"
            + "        PathmindHelperRegistry.setForceLoadPolicy(policyHelper);\n"
            + "\n"
            + resetSnippet
            + "\n"
            + "        engine.start(agent);\n"
            + "    }\n"
            + "\n"
            + (multiAgent
                ? "    @Override public Array step(Array action) {\n"
                + "        double[] reward = new double[(int)action.length()];\n"
                + "        double[][] before = PathmindHelperRegistry.getHelper().observationForReward();\n"
                + "        engine.runFast();\n"
                + "        int[] array = new int[(int)action.length()];\n"
                + "        for (int i = 0; i < array.length; i++) {\n"
                + "            array[i] = (int)action.data().get(i);\n"
                + "        }\n"
                + "        PathmindHelperRegistry.getHelper().doAction(array);\n"
                + "        double[][] after = PathmindHelperRegistry.getHelper().observationForReward();\n"
                + "\n"
                + rewardSnippet
                + "\n"
                + "        float[] array2 = new float[reward.length];\n"
                + "        for (int i = 0; i < reward.length; i++) {\n"
                + "            array2[i] = (float)reward[i];\n"
                + "        }\n"
                + "        if (this.reward == null || this.reward.length() != array2.length) {\n"
                + "            this.reward = new Array(new SSizeTVector().put(reward.length));\n"
                + "        }\n"
                + "        this.reward.data().put(array2);\n"
                + "        return this.reward;\n"
                + "    }\n"
                + "\n"

                : "    @Override public float step(long action) {\n"
                + "        double reward = 0;\n"
                + "        double[] before = PathmindHelperRegistry.getHelper().observationForReward();\n"
                + "        engine.runFast();\n"
                + "        long[] array = new long[(int)action.length()];\n"
                + "        for (int i = 0; i < array.length; i++) {\n"
                + "            array[i] = (long)action.data().get(i);\n"
                + "        }\n"
                + "        PathmindHelperRegistry.getHelper().doAction(array);\n"
                + "        double[] after = PathmindHelperRegistry.getHelper().observationForReward();\n"
                + "\n"
                + rewardSnippet
                + "\n"
                + "        return (float)reward;\n"
                + "    }\n"
                + "\n")
            + "    public double[] test() {\n"
            + "        double[] metrics = null;\n"
            + "        reset();\n"
            + "        while (!isDone()) {\n"
            + "            engine.runFast();\n"
            + "        }\n"
            + "\n"
            + metricsSnippet
            + "\n"
            + "        return metrics;\n"
            + "    }\n"
            + "\n"
            + "    public static void main(String[] args) throws Exception {\n"
            + (policyHelper != null
                    ? "        " + className + " e = new " + className + "(new " + policyHelper + "(new File(args[0]), " + actionTupleSize + "));\n"
                    + "        ArrayList<String> lines = new ArrayList<String>(" + testIterations + ");\n"
                    + "        for (int i = 0; i < " + testIterations + "; i++) {\n"
                    + "            lines.add(Arrays.toString(e.test()));\n"
                    + "        }\n" : "")
                    + "        Files.write(Paths.get(args[0], \"metrics.txt\"), lines, Charset.defaultCharset());\n"
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
                System.out.println("    --step-time");
                System.out.println("    --stop-time");
                System.out.println("    --class-snippet");
                System.out.println("    --reset-snippet");
                System.out.println("    --reward-snippet");
                System.out.println("    --metrics-snippet");
                System.out.println("    --policy-helper");
                System.out.println("    --test-iterations");
                System.out.println("    --stepless");
                System.out.println("    --action-tuple-size");
                System.out.println("    --multi-agent");
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
            } else if ("--metrics-snippet".equals(args[i])) {
                helper.metricsSnippet(args[++i]);
            } else if ("--policy-helper".equals(args[i])) {
                helper.policyHelper(args[++i]);
            } else if ("--test-iterations".equals(args[i])) {
                helper.testIterations(Integer.parseInt(args[++i]));
            } else if ("--stepless".equals(args[i])) {
                helper.setStepless(true);
            } else if ("--multi-agent".equals(args[i])) {
                helper.multiAgent = true;
            } else if ("--action-tuple-size".equals(args[i])) {
                helper.actionTupleSize(Integer.parseInt(args[++i]));
            } else {
                output = new File(args[i]);
            }
        }
        if (output == null) {
            output = new File(helper.environmentClassName.replace('.', '/') + ".java");
        }
        helper/*.checkAgentClass()*/.generateEnvironment(output);
    }
}
