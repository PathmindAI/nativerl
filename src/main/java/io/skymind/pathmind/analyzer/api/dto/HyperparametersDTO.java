package io.skymind.pathmind.analyzer.api.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import io.swagger.annotations.ApiModelProperty;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import javax.validation.constraints.NotBlank;
import javax.validation.constraints.NotEmpty;
import java.util.List;
import java.util.stream.Collectors;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class HyperparametersDTO {

    private static List<String> knownOutputs = List.of(
            "observations:",
            "actions:",
            "rewardVariablesCount:",
            "reward:",
            "failed_steps:",
            "model-analyzer-mode:",
            "actionTupleSize:");

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

    @ApiModelProperty(value = "Action tuple size extracted from model", example = "2", required = true)
    @NotBlank(message = "Tuple size cannot be blank")
    private String actionTupleSize;
    
    public static HyperparametersDTO of(@NotEmpty List<String> hyperparametersList) {
        hyperparametersList = hyperparametersList.stream()
                .filter(HyperparametersDTO::isHyperparameters)
                .map(HyperparametersDTO::extractHyperparameters)
                .collect(Collectors.toList());

        return new HyperparametersDTO(
                hyperparametersList.get(0),
                hyperparametersList.get(1),
                hyperparametersList.get(2),
                hyperparametersList.get(3),
                hyperparametersList.get(4),
                hyperparametersList.get(5),
                hyperparametersList.get(6));
    }

    private static boolean isHyperparameters(String output) {
        return knownOutputs.stream().anyMatch(output::contains);
    }

    private static String extractHyperparameters(String output) {
        return output.substring(output.indexOf(":") + 1);
    }
}
