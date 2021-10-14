package io.skymind.pathmind.analyzer.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;
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

//    @ApiModelProperty(value = "Whether the pathmind helper is enabled or not", example = "true")
    private boolean isEnabled = false;

//    @ApiModelProperty(value = "Flag for when old model versions are found", example = "true")
    private boolean oldVersionFound = false;

    private List<SimulationParameter> agentParams;

//    @ApiModelProperty(value = "Number of observations extracted from model", example = "10", required = true)
//    @NotBlank(message = "Number of observations cannot be blank")
    private String observations;

//    @ApiModelProperty(value = "Observations names extracted from model", example = "[\"orderQueueSize\", \"collectQueueSize\"]", required = true)
//    @NotBlank(message = "Observation names cannot be empty")
    private List<String> observationNames;

//    @NotBlank(message = "Observation types cannot be empty")
    private List<String> observationTypes;

//    @ApiModelProperty(value = "Number of actions extracted from model", example = "5", required = true)
//    @NotBlank(message = "Number of actions cannot be blank")
    private String actions;

//    @ApiModelProperty(value = "Whether the action mask is enabled or not", example = "true")
    private boolean isActionMask;

//    @ApiModelProperty(value = "Length of reward variables array extracted from model", example = "7", required = true)
//    @NotBlank(message = "Reward variables count cannot be blank")
    private String rewardVariablesCount;

//    @ApiModelProperty(value = "Reward variable names extracted from model", example = "[\"var1\", \"var2\"]", required = true)
//    @NotNull(message = "Reward variable names is required")
//    @NotEmpty(message = "Reward variable names cannot be empty")
    private List<String> rewardVariableNames;

//    @ApiModelProperty(value = "Reward variable types extracted from model", example = "[\"int\", \"boolean\"]", required = true)
//    @NotNull(message = "Reward variable names is required")
//    @NotEmpty(message = "Reward variable types cannot be empty")
    private List<String> rewardVariableTypes;

//    @ApiModelProperty(value = "Reward function definition", required = true)
//    @NotBlank(message = "Reward function definition cannot be blank") TODO: validate not empty reward function
    private String rewardFunction;

//    @JsonInclude(JsonInclude.Include.NON_EMPTY)
//    @ApiModelProperty(value = "Steps which failed while extracting hyperparameters")
    private String failedSteps;

//    @ApiModelProperty(value = "the number of agents", required = true)
//    @NotBlank(message = "Agents cannot be blank")
    private String agents;

//    @ApiModelProperty(value = "Extraction mode (single/multi)", required = true)
//    @NotBlank(message = "Mode cannot be blank")
    private String mode;

    private static List<String> filterOutEmpty(List<String> source) {
        return source.stream().filter(s -> !s.isEmpty()).collect(Collectors.toList());
    }

    public static boolean isHyperparameters(String parameterCandidate) {
        return KNOWN_OUTPUT.contains(parameterCandidate);
    }

}
