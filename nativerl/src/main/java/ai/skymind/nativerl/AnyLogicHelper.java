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
import java.nio.file.Files;

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

    /** The class name of the main AnyLogic agent to use. */
    @Builder.Default
    String agentClassName = "MainAgent";

    /** The number of actions in the case of a discrete action space. */
    @Builder.Default
    long discreteActions;

    /** The number of array elements in the case of a continuous action space. */
    @Builder.Default
    long continuousActions;

    /** The number of array elements in the case of a continuous action space. */
    @Builder.Default
    long continuousObservations;

    /** Currently unused since the PathmindHelper is able to manage this. */
    @Builder.Default
    long stepTime = 1;

    /** Currently unused since the PathmindHelper is able to manage this. */
    @Builder.Default
    long stopTime = 1000;

    /** Arbitrary code to add to the generated class such as fields or methods. */
    @Builder.Default
    String classSnippet = "";

    /** Arbitrary code to add to the reset() method of the generated class. */
    @Builder.Default
    String resetSnippet = "";

    /** Arbitrary code to add to the step() method of the generated class to calculate the reward. */
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

    /** Currently unused since the PathmindHelper is able to manage this. */
    @Builder.Default
    boolean stepless = false;

    /** Indicates that we need multiagent support with the Environment class provided, but where all agents share the same policy. */
    @Builder.Default
    boolean multiAgent = false;

    int actionTupleSize;

    @Builder.Default
    boolean setStepless = false;

    @Setter
    String className, packageName;

    /** Currently unused since the PathmindHelper is able to manage this. */
    AnyLogicHelper checkAgentClass() throws ClassNotFoundException, NoSuchMethodException, NoSuchFieldException {
        int n = agentClassName.lastIndexOf(".");
        String className = agentClassName.substring(n + 1);
        String packageName = n > 0 ? agentClassName.substring(0, n) : null;

        Class.forName((packageName != null ? packageName + "." : "") + "Training");
        Class c = Class.forName(agentClassName);
        if (multiAgent) {
            c.getDeclaredMethod("doAction", int[].class);
        } else {
            c.getDeclaredMethod("doAction", int[].class);
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

    /** Calls {@link #generateEnvironment()} and writes the result to a File. */
    public void generateEnvironment(File file) throws IOException {
        File directory = file.getParentFile();
        if (directory != null) {
            directory.mkdirs();
        }
        Files.write(file.toPath(), generateEnvironment().getBytes());
    }

    /** Takes the parameters from an instance of this class, and returns a Java class that extends AbstractEnvironment. */
    public String generateEnvironment() throws IOException {
        int n = environmentClassName.lastIndexOf(".");
        String className = environmentClassName.substring(n + 1);
        String packageName = n > 0 ? environmentClassName.substring(0, n) : null;

        this.setClassName(className);
        this.setPackageName(packageName);

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
                helper.multiAgent(true);
            } else if ("--action-tuple-size".equals(args[i])) {
                helper.actionTupleSize(Integer.parseInt(args[++i]));
            } else {
                output = new File(args[i]);
            }
        }
        AnyLogicHelper anyLogicHelper = helper.build();
        if (output == null) {
            output = new File(anyLogicHelper.getEnvironmentClassName().replace('.', '/') + ".java");
        }
        anyLogicHelper/*.checkAgentClass()*/.generateEnvironment(output);
    }
}
