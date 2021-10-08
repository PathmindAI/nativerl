package io.skymind.pathmind.analyzer.api.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.swagger.annotations.ApiModelProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.apache.commons.io.FileUtils;

import javax.validation.constraints.NotBlank;
import javax.validation.constraints.NotEmpty;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

@NoArgsConstructor
@AllArgsConstructor
@Data
public class HyperparametersDTO {

    private final static Set<String> KNOWN_OUTPUT = Set.of(
            "model-analyzer-mode",
            "DTOPath"
            );

    @ApiModelProperty(value = "Whether the pathmind helper is enabled or not", example = "true")
    private boolean isEnabled = false;

    @ApiModelProperty(value = "Flag for when old model versions are found", example = "true")
    private boolean oldVersionFound = false;

    private List<SimulationParameter> agentParams;

    @ApiModelProperty(value = "Number of observations extracted from model", example = "10", required =
            true)
//    @NotBlank(message = "Number of observations cannot be blank")
    private String observations;

    @ApiModelProperty(value = "Observations names extracted from model", example = "[\"orderQueueSize\", \"collectQueueSize\"]", required =
            true)
//    @NotBlank(message = "Observation names cannot be empty")
    private List<String> observationNames;

//    @NotBlank(message = "Observation types cannot be empty")
    private List<String> observationTypes;

    @ApiModelProperty(value = "Number of actions extracted from model", example = "5", required = true)
//    @NotBlank(message = "Number of actions cannot be blank")
    private String actions;

    @ApiModelProperty(value = "Whether the action mask is enabled or not", example = "true")
    private boolean isActionMask;

    @ApiModelProperty(value = "Length of reward variables array extracted from model", example = "7", required = true)
//    @NotBlank(message = "Reward variables count cannot be blank")
    private String rewardVariablesCount;

    @ApiModelProperty(value = "Reward variable names extracted from model", example = "[\"var1\", \"var2\"]", required = true)
//    @NotNull(message = "Reward variable names is required")
//    @NotEmpty(message = "Reward variable names cannot be empty")
    private List<String> rewardVariableNames;

    @ApiModelProperty(value = "Reward variable types extracted from model", example = "[\"int\", \"boolean\"]", required = true)
//    @NotNull(message = "Reward variable names is required")
//    @NotEmpty(message = "Reward variable types cannot be empty")
    private List<String> rewardVariableTypes;

    @ApiModelProperty(value = "Reward function definition", required =
            true)
    @NotBlank(message = "Reward function definition cannot be blank")
    private String rewardFunction;

    @JsonInclude(JsonInclude.Include.NON_EMPTY)
    @ApiModelProperty(value = "Steps which failed while extracting hyperparameters")
    private String failedSteps;

    @ApiModelProperty(value = "the number of agents", required = true)
//    @NotBlank(message = "Agents cannot be blank")
    private String agents;

    @ApiModelProperty(value = "Extraction mode (single/multi)", required = true)
//    @NotBlank(message = "Mode cannot be blank")
    private String mode;

    public static HyperparametersDTO of(@NotEmpty List<String> hyperparametersList) {
        Map<String, String> parametersMap = hyperparametersList.stream()
                .map(String::strip)
                .map(l -> l.split(":", 2))
                .filter(p -> HyperparametersDTO.isHyperparameters(p[0]))
                .collect(Collectors.toMap(p -> p[0], p -> p[1].strip()));
        ObjectMapper objectMapper = new ObjectMapper();

        String dtoFilePath = parametersMap.getOrDefault("DTOPath", "");
        try {
            String json = FileUtils.readFileToString(new File(dtoFilePath), Charset.defaultCharset());
            return objectMapper.readValue(json, HyperparametersDTO.class);
        } catch (JsonProcessingException e) {
            e.printStackTrace();
            return null;
        } catch (IOException e) {
            String maMode = parametersMap.get("model-analyzer-mode");
            if (maMode != null && !maMode.isEmpty() && maMode.startsWith("py_")) {
                HyperparametersDTO dto = new HyperparametersDTO();
                dto.setMode(maMode);
                return dto;
            }
            e.printStackTrace();
            return null;
        }
    }

    private static List<String> filterOutEmpty(List<String> source) {
        return source.stream().filter(s -> !s.isEmpty()).collect(Collectors.toList());
    }

    private static boolean isHyperparameters(String parameterCandidate) {
        return KNOWN_OUTPUT.contains(parameterCandidate);
    }

    private static List<SimulationParameter> asParamList(String mapString) {
        ObjectMapper objectMapper = new ObjectMapper();
        try {
            return objectMapper.readValue(mapString, List.class);
        } catch (JsonProcessingException e) {
            e.printStackTrace();
            return null;
        }
    }
}
