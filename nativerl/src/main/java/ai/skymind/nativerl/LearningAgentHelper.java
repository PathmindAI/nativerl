package ai.skymind.nativerl;

import com.github.jknack.handlebars.Handlebars;
import com.github.jknack.handlebars.Template;
import com.github.jknack.handlebars.helper.ConditionalHelpers;
import com.github.jknack.handlebars.io.ClassPathTemplateLoader;
import com.github.jknack.handlebars.io.TemplateLoader;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

public class LearningAgentHelper {
    public void generateLearningAgent(File file) throws IOException {
        File directory = file.getParentFile();
        if (directory != null) {
            directory.mkdirs();
        }
        Files.write(file.toPath(), generateLearningAgent(file.getName()).getBytes());

    }
    public String generateLearningAgent(String fileName) throws IOException {
        TemplateLoader loader = new ClassPathTemplateLoader("/ai/skymind/nativerl", ".hbs");
        Handlebars handlebars = new Handlebars(loader);

        handlebars.registerHelpers(ConditionalHelpers.class);
        Template template = handlebars.compile(fileName);

        String learningAgent = template.apply(this);
        return learningAgent;
    }
    public static void main(String[] args) throws IOException {
        LearningAgentHelper learningAgentHelper = new LearningAgentHelper();
        learningAgentHelper.generateLearningAgent(new File("com/pathmind/anylogic/PathmindLearningAgent.java"));
    }
}
