package io.skymind.pathmind.analyzer.code;

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
import java.nio.file.Files;

@Getter
@Builder
public class CodeGenerator {

    @Builder.Default
    String agentClassName;
    @Setter
    String packageName;
    @Setter
    String simulationClassName;
    @Setter
    boolean isRLExperiment;

    public boolean getIsRLExperiment() {
        return isRLExperiment;
    }

    private final static String MODEL_ANALYZER_NAME = "ModelAnalyzer.java";
    private final static String TRAINING_NAME = "Training.java";
    private final static String LEARNING_AGENT = "PathmindLearningAgent.java";

    public void generateEnvironment(File file) throws IOException {
        if (!file.exists()) {
            file.mkdirs();
        }

        File modelAnalyzer = new File(file, MODEL_ANALYZER_NAME);
        Files.write(modelAnalyzer.toPath(), generateEnvironment(MODEL_ANALYZER_NAME).getBytes());

        if (!isRLExperiment) {
            File training = new File(file, TRAINING_NAME);
            Files.write(training.toPath(), generateEnvironment(TRAINING_NAME).getBytes());
        }

        File pathmindLearningAgentPath = new File("com/pathmind/anylogic");
        if (!pathmindLearningAgentPath.exists()) {
            pathmindLearningAgentPath.mkdirs();
        }
        File learningAgent = new File(pathmindLearningAgentPath, LEARNING_AGENT);
        Files.write(learningAgent.toPath(), generateEnvironment(LEARNING_AGENT).getBytes());
    }

    public String generateEnvironment(String fileName) throws IOException {
        this.setRLExperiment(simulationClassName.endsWith("RLExperiment"));

        TemplateLoader loader = new ClassPathTemplateLoader();
        loader.setPrefix("/templates/");
        loader.setSuffix(".hbs");
        Handlebars handlebars = new Handlebars(loader);

        handlebars.registerHelpers(ConditionalHelpers.class);
        Template template = handlebars.compile(fileName);

        String env = template.apply(this);
        return env;
    }

    public static void main(String[] args) throws IOException {
        CodeGenerator.CodeGeneratorBuilder builder = CodeGenerator.builder();

        for (int i = 0; i < args.length; i++) {
            if ("--agent-class-name".equals(args[i])) {
                builder.agentClassName(args[++i]);
            } else if ("--package-name".equals(args[i])) {
                builder.packageName(args[++i]);
            } else if ("--simulation-class-name".equals(args[i])) {
                builder.simulationClassName(args[++i]);
            }
        }

        CodeGenerator codeGenerator = builder.build();
        String path = codeGenerator.getPackageName();
        if (path == null || path.isEmpty()) {
            path = ".";
        } else {
            path = path.replaceAll("\\.", File.separator);
        }
        codeGenerator.generateEnvironment(new File(path));
    }
}
