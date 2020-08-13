package io.skymind.pathmind.analyzer.api.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import io.swagger.annotations.ApiModelProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import javax.validation.constraints.NotBlank;
import javax.validation.constraints.NotEmpty;
import javax.validation.constraints.NotNull;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

@NoArgsConstructor
@AllArgsConstructor
@Data
public class HyperparametersDTO {

    private final static Set<String> KNOWN_OUTPUT = Set.of(
            "observations",
            "actions",
            "rewardVariablesCount",
            "rewardVariables",
            "reward",
            "failedSteps",
            "model-analyzer-mode");

    @ApiModelProperty(value = "Number of observations extracted from model", example = "10", required =
            true)
    @NotBlank(message = "Number of observations cannot be blank")
    private String observations;

    @ApiModelProperty(value = "Number of actions extracted from model", example = "5", required = true)
    @NotBlank(message = "Number of actions cannot be blank")
    private String actions;

    @ApiModelProperty(value = "Length of reward variables array extracted from model", example = "7", required = true)
    @NotBlank(message = "Reward variables count cannot be blank")
    private String rewardVariablesCount;

    @ApiModelProperty(value = "Reward variables names extracted from model", example = "[\"var1\", \"var2\"]", required = true)
    @NotNull(message = "Reward variables is required")
    @NotEmpty(message = "Reward variables cannot be empty")
    private List<String> rewardVariables;
    
    @ApiModelProperty(value = "Reward function definition", required =
            true)
    @NotBlank(message = "Reward function definition cannot be blank")
    private String rewardFunction;

    @JsonInclude(JsonInclude.Include.NON_EMPTY)
    @ApiModelProperty(value = "Steps which failed while extracting hyperparameters")
    private String failedSteps;

    @ApiModelProperty(value = "Extraction mode (single/multi)", required = true)
    @NotBlank(message = "Mode cannot be blank")
    private String mode;

    public static HyperparametersDTO of(@NotEmpty List<String> hyperparametersList) {
        Map<String, String> parametersMap = hyperparametersList.stream()
                .map(String::strip)
                .map(l -> l.split(":", 2))
                .filter(p -> HyperparametersDTO.isHyperparameters(p[0]))
                .collect(Collectors.toMap(p -> p[0], p -> p[1].strip()));

        return new HyperparametersDTO(
                parametersMap.get("observations"),
                parametersMap.get("actions"),
                parametersMap.get("rewardVariablesCount"),
                Arrays.asList(parametersMap.get("rewardVariables").split("\\|")),
                parametersMap.get("reward"),
                parametersMap.get("failedSteps"),
                parametersMap.get("model-analyzer-mode"));
    }

    private static boolean isHyperparameters(String parameterCandidate) {
        return KNOWN_OUTPUT.contains(parameterCandidate);
    }
}
