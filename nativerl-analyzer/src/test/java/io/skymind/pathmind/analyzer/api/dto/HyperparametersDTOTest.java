package io.skymind.pathmind.analyzer.api.dto;

import java.util.Arrays;
import java.util.Collections;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class HyperparametersDTOTest {
    @Test
    public void testOf_allOk() {
        HyperparametersDTO result = HyperparametersDTO.of(Arrays.asList(
                "isEnabled:true",
                "observations:5",
                "observationsNames:orderQueueSize|collectQueueSize|payBillQueueSize|kitchenCleanlinessLevel|timeOfDay",
                "observationsTypes:double|double|double|double|double",
                "actions:4",
                "rewardVariablesCount:4",
                "reward: not defined",
                "failedSteps:",
                "model-analyzer-mode: single",
                "agents:1",
                "rewardVariables:vars[0]|vars[1]|vars[2]|vars[3]"
        ));

        HyperparametersDTO expected = new HyperparametersDTO(
                true,
                false,
                "5",
                Arrays.asList("orderQueueSize", "collectQueueSize", "payBillQueueSize", "kitchenCleanlinessLevel", "timeOfDay"),
                Arrays.asList("double", "double", "double", "double", "double"),
                "4",
                "4",
                Arrays.asList("vars[0]", "vars[1]", "vars[2]", "vars[3]"),
                "not defined",
                "",
                "1",
                "single"
        );
        Assertions.assertEquals(expected, result);
    }

    @Test
    public void testOf_allOk_butThereIsTheVarOldVersionFoundWithValueFalse() {
        HyperparametersDTO result = HyperparametersDTO.of(Arrays.asList(
                "isEnabled:true",
                "observations:5",
                "observationsNames:orderQueueSize|collectQueueSize|payBillQueueSize|kitchenCleanlinessLevel|timeOfDay",
                "observationsTypes:double|double|double|double|double",
                "actions:4",
                "rewardVariablesCount:4",
                "reward: not defined",
                "failedSteps:",
                "model-analyzer-mode: single",
                "agents:1",
                "rewardVariables:vars[0]|vars[1]|vars[2]|vars[3]"
        ));

        HyperparametersDTO expected = new HyperparametersDTO(
                true,
                false,
                "5",
                Arrays.asList("orderQueueSize", "collectQueueSize", "payBillQueueSize", "kitchenCleanlinessLevel", "timeOfDay"),
                Arrays.asList("double", "double", "double", "double", "double"),
                "4",
                "4",
                Arrays.asList("vars[0]", "vars[1]", "vars[2]", "vars[3]"),
                "not defined",
                "",
                "1",
                "single"
        );
        Assertions.assertEquals(expected, result);
    }


    @Test
    public void testOf_oldVersionFound() {
        HyperparametersDTO result = HyperparametersDTO.of(Arrays.asList(
                "oldVersionFound:true"
        ));

        HyperparametersDTO expected = new HyperparametersDTO();
        expected.setOldVersionFound(true);
        Assertions.assertEquals(expected, result);
    }

    @Test
    public void testOf_withFailedSteps() {
        HyperparametersDTO result = HyperparametersDTO.of(Arrays.asList(
                "isEnabled:true",
                "observations:0",
                "observationsNames:",
                "observationsTypes:",
                "actions:4",
                "rewardVariablesCount:4",
                "reward: not defined",
                "failedSteps:observations,observationsNames",
                "agents:1",
                "model-analyzer-mode: single",
                "rewardVariables:vars[0]|vars[1]|vars[2]|vars[3]"
        ));

        HyperparametersDTO expected = new HyperparametersDTO(
                true,
                false,
                "0",
                Collections.emptyList(),
                Collections.emptyList(),
                "4",
                "4",
                Arrays.asList("vars[0]", "vars[1]", "vars[2]", "vars[3]"),
                "not defined",
                "observations,observationsNames",
                "1",
                "single"
        );
        Assertions.assertEquals(expected, result);
    }


}
