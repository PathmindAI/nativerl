package io.skymind.pathmind.analyzer.api.dto;

import io.swagger.annotations.ApiModelProperty;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import javax.validation.constraints.NotBlank;
import javax.validation.constraints.NotEmpty;
import java.util.List;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class HyperparametersDTO {

    @ApiModelProperty(value = "Number of actions extracted from model", example = "5", required = true)
    @NotBlank(message = "Number of actions cannot be blank")
    private String actions;

    @ApiModelProperty(value = "Number of observations extracted from model", example = "10", required =
            true)
    @NotBlank(message = "Number of observations cannot be blank")
    private String observations;


    @ApiModelProperty(value = "Reward function definition", required =
            true)
    @NotBlank(message = "Reward function definition cannot be blank")
    private String rewardFunction;


    public static HyperparametersDTO of(@NotEmpty final List<String> hyperparametersList) {
        return new HyperparametersDTO(hyperparametersList.get(0), hyperparametersList.get(1),
                hyperparametersList.get(2));
    }
}
