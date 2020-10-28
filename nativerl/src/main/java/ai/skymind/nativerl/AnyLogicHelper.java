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
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;

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

    /** The algorithm to use with RLlib for training and the PythonPolicyHelper. */
    @Builder.Default
    String algorithm = "PPO";

    /** The directory where to output the logs of RLlib. */
    @Builder.Default
    File outputDir = null;

    /** Arbitrary code to add to the generated class such as fields or methods. */
    @Builder.Default
    String classSnippet = "";

    /** Arbitrary code to add to the reset() method of the generated class. */
    @Builder.Default
    String resetSnippet = "";

    /** Arbitrary code to add to the getObservation() method of the generated class to filter the observations. */
    @Builder.Default
    String observationSnippet = "";

    /** Arbitrary code to add to the getReward() method of the generated class to calculate the reward. */
    @Builder.Default
    String rewardSnippet = "";

    /** Arbitrary code to add to the test() method of the generated class to compute custom metrics. */
    @Builder.Default
    String metricsSnippet = "";

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
        int n = environmentClassName.lastIndexOf(".");
        String className = environmentClassName.substring(n + 1);
        String packageName = n > 0 ? environmentClassName.substring(0, n) : null;
        ObservationProcessor op = new ObservationProcessor(agentClassName);
        RewardProcessor rp = new RewardProcessor(agentClassName);

        this.setClassName(className);
        this.setPackageName(packageName);
        this.setObservationClassName(op.getObservationClass().getName().substring(packageName.length() + 1));
        this.setRewardClassName(rp.getRewardClass().getName().substring(packageName.length() + 1));

        TemplateLoader loader = new ClassPathTemplateLoader("/ai/skymind/nativerl", ".hbs");
        Handlebars handlebars = new Handlebars(loader);

        handlebars.registerHelpers(ConditionalHelpers.class);
        Template template = handlebars.compile("AnyLogicHelper.java");

        String env = template.apply(this);

        return env;
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
                System.out.println("    --class-snippet");
                System.out.println("    --reset-snippet");
                System.out.println("    --observation-snippet");
                System.out.println("    --reward-snippet");
                System.out.println("    --metrics-snippet");
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
            } else if ("--class-snippet".equals(args[i])) {
                helper.classSnippet(args[++i]);
            } else if ("--algorithm".equals(args[i])) {
                helper.algorithm(args[++i]);
            } else if ("--output-dir".equals(args[i])) {
                helper.outputDir(new File(args[++i]));
            } else if ("--reset-snippet".equals(args[i])) {
                helper.resetSnippet(args[++i]);
            } else if ("--observation-snippet".equals(args[i])) {
                String obsSnippet = args[++i];
                if (obsSnippet.startsWith("file:")) {
                    File file = new File(obsSnippet.split(":")[1]);
                    if (!file.exists()) {
                        throw new RuntimeException("observation file doesn't exist!");
                    }

                    StringBuilder sb = new StringBuilder();
                    Files.lines(Paths.get(file.getPath()), Charset.defaultCharset())
                            .forEach(s -> sb.append(s));
                    obsSnippet = sb.toString();
                }
                helper.observationSnippet(obsSnippet);
            } else if ("--reward-snippet".equals(args[i])) {
                helper.rewardSnippet(args[++i]);
            } else if ("--metrics-snippet".equals(args[i])) {
                helper.metricsSnippet(args[++i]);
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
