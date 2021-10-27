package io.skymind.pathmind.analyzer.config.swagger;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;

@Data
@ConfigurationProperties(prefix = "io.skymind.pathmind.model-analyzer.swagger")
public class SwaggerProperties {

    private String title;
    private String description;
    /**
     * Fully qualified name of the APIs package
     */
    private String basePackage;
}
