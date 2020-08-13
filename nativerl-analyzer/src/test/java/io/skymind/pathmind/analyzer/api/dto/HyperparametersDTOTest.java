package io.skymind.pathmind.analyzer.api.dto;

import java.util.Arrays;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class HyperparametersDTOTest {
    @Test
    public void testOf() {
        HyperparametersDTO result = HyperparametersDTO.of(Arrays.asList(
                "observations:5",
                "actions:4",
                "rewardVariablesCount:4",
                "reward: not defined",
                "failedSteps:",
                "model-analyzer-mode: single",
                "rewardVariables:vars[0]|vars[1]|vars[2]|vars[3]"
        ));

        HyperparametersDTO expected = new HyperparametersDTO(
                "5",
                "4",
                "4",
                Arrays.asList("vars[0]", "vars[1]", "vars[2]", "vars[3]"),
                "not defined",
                "",
                "single"
        );
        Assertions.assertEquals(expected, result);
    }
}
