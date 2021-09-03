package ai.skymind.nativerl;

import com.github.jknack.handlebars.Handlebars;
import com.github.jknack.handlebars.Template;
import com.github.jknack.handlebars.helper.ConditionalHelpers;
import com.github.jknack.handlebars.io.ClassPathTemplateLoader;
import com.github.jknack.handlebars.io.TemplateLoader;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * This is a helper class to help users implement the reinforcement learning
 * Environment interface based on a simulation model from AnyLogic. The output
 * is the source code of a Java class that can be compiled and executed together
 * with an exported simulation model that is using the Pathmind Helper.
 */
@Getter
@Builder
public class AnyLogicHelper {

    /** The name of the class to generate. */
    @Builder.Default
    String environmentClassName = "AnyLogicEnvironment";

    /** The class name of the simulation class to import. */
    @Builder.Default
    String simulationClassName = "Simulation";

    /** The class name of the main AnyLogic agent to use. */
    @Builder.Default
    String agentClassName = "MainAgent";

    /** The type of Experiment, now we have Simulation or RLExperiment.  */
    @Builder.Default
    String experimentType;

    /** The algorithm to use with RLlib for training and the PythonPolicyHelper. */
    @Builder.Default
    String algorithm = "PPO";

    /** The directory where to output the logs of RLlib. */
    @Builder.Default
    File outputDir = new File(".");

    /** Arbitrary code to add to the generated class such as fields or methods. */
    @Builder.Default
    String classSnippet = "";

    /** Arbitrary code to add to the reset() method of the generated class. */
    @Builder.Default
    String resetSnippet = "";

    /** Arbitrary code to add to the reset() method of the generated class for simulation parameters. */
    @Builder.Default
    String simulationParameterSnippet = "";

    /** Arbitrary code to add to the getObservation() method of the generated class to filter the observations. */
    @Builder.Default
    String observationSnippet = "";

    /** Arbitrary code to add to the getReward() method of the generated class to calculate the reward. */
    @Builder.Default
    String rewardSnippet = "";

    /** The weights for the reward terms */
    String rewardTermWeights = "";

    /** Arbitrary code to add to the test() method of the generated class to compute custom metrics. */
    @Builder.Default
    String metricsSnippet = "";

    /** Arbitrary code to add to the test() method of the generated class to compute reward terms. */
    @Builder.Default
    String rewardTermsSnippet = "";

    /** The name of the PolicyHelper, such as RLlibPolicyHelper, to run the metrics code as part of the main() method. */
    @Builder.Default
    String policyHelper = null;

    /** The number of episodes to run the PolicyHelper and compute the metrics on as part of the main() method. */
    @Builder.Default
    int testIterations = 10;

    /** Indicates that we need multiagent support with the Environment class provided, but where all agents share the same policy. */
    @Builder.Default
    boolean multiAgent = false;

    /** Indicates that the reward snippet needs to use reward objects for "before" and "after", instead of arrays of doubles. */
    @Builder.Default
    boolean namedVariables = false;

    @Setter
    String className, packageName, observationClassName, rewardClassName;

    @Setter
    boolean isRLExperiment;

    @Setter
    boolean isPLE;

    @Setter
    List<String> setObs;

    @Setter
    int numRewardTerms;

    public boolean getIsRLExperiment() {
        return isRLExperiment;
    }

    public boolean getIsPLE() {
        return isPLE;
    }

    /** Calls {@link #generateEnvironment()} and writes the result to a File. */
    public void generateEnvironment(File file) throws IOException, ReflectiveOperationException {
        File directory = file.getParentFile();
        if (directory != null) {
            directory.mkdirs();
        }
        Files.write(file.toPath(), generateEnvironment().getBytes());
    }

    /** Takes the parameters from an instance of this class, and returns a Java class that extends AbstractEnvironment. */
    public String generateEnvironment() throws IOException, ReflectiveOperationException {
        if (rewardTermsSnippet == null || rewardTermsSnippet.isEmpty()) {
            List<String> processedRewardTerms = Arrays.stream(rewardSnippet.split("\n"))
                    .map(t -> {
                        if (t.startsWith("reward += ")) {
                            t = t.replaceFirst("reward \\+= ", "");
                        } else if (t.startsWith("reward -= ")) {
                            t = t.replaceFirst("reward \\-= ", "");
                            int index = t.lastIndexOf(";");
                            t = "- ( " + t.substring(0, index) + " ) " + t.substring(index);
                        }
                        return t;
                    })
                    .collect(Collectors.toList());

            List<String> tempRewardTermsSnippet = new ArrayList<>();
            for (int i = 0; i < processedRewardTerms.size(); i++) {
                tempRewardTermsSnippet.add("rewardTermsRaw[" + i + "] = " + processedRewardTerms.get(i));
            }
            rewardTermsSnippet = String.join("\n", tempRewardTermsSnippet);
        }

        int n = environmentClassName.lastIndexOf(".");
        String className = environmentClassName.substring(n + 1);
        String packageName = n > 0 ? environmentClassName.substring(0, n) : null;
        ObservationProcessor op = new ObservationProcessor(agentClassName);
        RewardProcessor rp = new RewardProcessor(agentClassName);
        setObservationSnippet();
        setSimulationParameterSnippet();

        this.setClassName(className);
        this.setPackageName(packageName);
        this.setObservationClassName(op.getObservationClass().getName().substring(packageName.length() + 1));
        this.setRewardClassName(rp.getRewardClass().getName().substring(packageName.length() + 1));
        this.setRLExperiment(experimentType.equals("RLExperiment"));
        this.isPLE = true;
        this.setNumRewardTerms(rewardTermsSnippet.split("\n").length);

        TemplateLoader loader = new ClassPathTemplateLoader("/ai/skymind/nativerl", ".hbs");
        Handlebars handlebars = new Handlebars(loader);

        handlebars.registerHelpers(ConditionalHelpers.class);
        handlebars.registerHelper("escapePath", (context, options) -> ((File)context).getAbsolutePath().replace("\\", "/"));
        Template template = handlebars.compile("AnyLogicHelper.java");

        String env = template.apply(this);

        return env;
    }

    /** Handle observation snippet, if the content is too long, it will split it into multiple methods. */
    public void setObservationSnippet() throws IOException {
        this.setObs = new ArrayList<>();
        String obsSnippet = this.getObservationSnippet();

        if (obsSnippet.startsWith("file:")) {
            File file = new File(obsSnippet.split(":")[1]);
            if (!file.exists()) {
                throw new RuntimeException("observation file doesn't exist!");
            }

            StringBuilder sb = new StringBuilder();
            List<String> lines = Files.lines(Paths.get(file.getPath()), Charset.defaultCharset())
                    .collect(Collectors.toList());

            // add double[] out = new double[n];
            sb.append(lines.remove(0) + "\n");

            int limit = 3000;
            int numObsSelection = lines.size() / limit + 1;

            for (int i = 0; i < numObsSelection; i++) {
                sb.append("setObs_" + i + "(out);\n");
                setObs.add(String.join("\n", lines.subList((limit * i), Math.min(limit * (i+1), lines.size()))));
            }

            this.observationSnippet = sb.toString();
        }
    }

    /** Handle simulation parameter snippet. */
    public void setSimulationParameterSnippet() throws IOException {
        String simulationParameterSnippet = this.getSimulationParameterSnippet();

        if (simulationParameterSnippet.startsWith("file:")) {
            File file = new File(simulationParameterSnippet.split(":")[1]);
            if (!file.exists()) {
                throw new RuntimeException("simulationParameterSnippet file doesn't exist!");
            }

            byte[] bytes = Files.readAllBytes(Paths.get(file.getPath()));
            if (bytes != null && bytes.length > 0) {
                this.simulationParameterSnippet = new String(bytes);
            } else {
                this.simulationParameterSnippet = "";
            }
        }
    }

    /** The command line interface of this helper. */
    public static void main(String[] args) throws Exception {
        AnyLogicHelper.AnyLogicHelperBuilder helper = AnyLogicHelper.builder();
        File output = null;
        for (int i = 0; i < args.length; i++) {
            if ("-help".equals(args[i]) || "--help".equals(args[i])) {
                System.out.println("usage: AnyLogicHelper [options] [output]");
                System.out.println();
                System.out.println("options:");
                System.out.println("    --algorithm");
                System.out.println("    --environment-class-name");
                System.out.println("    --simulation-class-name");
                System.out.println("    --agent-class-name");
                System.out.println("    --experiment-type");
                System.out.println("    --class-snippet");
                System.out.println("    --reset-snippet");
                System.out.println("    --simulation-parameter-snippet");
                System.out.println("    --observation-snippet");
                System.out.println("    --reward-snippet");
                System.out.println("    --reward-term-weights");
                System.out.println("    --metrics-snippet");
                System.out.println("    --reward-terms-snippet");
                System.out.println("    --policy-helper");
                System.out.println("    --test-iterations");
                System.out.println("    --multi-agent");
                System.out.println("    --named-variables");
                System.exit(0);
            } else if ("--environment-class-name".equals(args[i])) {
                helper.environmentClassName(args[++i]);
            } else if("--simulation-class-name".equals(args[i])) {
                helper.simulationClassName(args[++i]);
            } else if ("--agent-class-name".equals(args[i])) {
                helper.agentClassName(args[++i]);
            } else if ("--experiment-type".equals(args[i])) {
                helper.experimentType(args[++i]);
            } else if ("--class-snippet".equals(args[i])) {
                helper.classSnippet(args[++i]);
            } else if ("--algorithm".equals(args[i])) {
                helper.algorithm(args[++i]);
            } else if ("--output-dir".equals(args[i])) {
                helper.outputDir(new File(args[++i]));
            } else if ("--reset-snippet".equals(args[i])) {
                helper.resetSnippet(args[++i]);
            } else if ("--reward-term-weights".equals(args[i])) {
                helper.rewardTermWeights(args[++i]);
            } else if ("--simulation-parameter-snippet".equals(args[i])) {
                helper.simulationParameterSnippet(args[++i]);
            } else if ("--observation-snippet".equals(args[i])) {
                helper.observationSnippet(args[++i]);
            } else if ("--reward-snippet".equals(args[i])) {
                helper.rewardSnippet(args[++i]);
            } else if ("--metrics-snippet".equals(args[i])) {
                helper.metricsSnippet(args[++i]);
            } else if ("--reward-terms-snippet".equals(args[i])) {
                helper.rewardTermsSnippet(args[++i]);
            } else if ("--policy-helper".equals(args[i])) {
                helper.policyHelper(args[++i]);
            } else if ("--test-iterations".equals(args[i])) {
                helper.testIterations(Integer.parseInt(args[++i]));
            } else if ("--multi-agent".equals(args[i])) {
                helper.multiAgent(true);
            } else if ("--named-variables".equals(args[i])) {
                helper.namedVariables(true);
            } else if (args[i].endsWith(".java")) {
                output = new File(args[i]);
            }
        }
        AnyLogicHelper anyLogicHelper = helper.build();
        if (output == null) {
            output = new File(anyLogicHelper.getEnvironmentClassName().replace('.', '/') + ".java");
        }
        anyLogicHelper.generateEnvironment(output);
    }
}
