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
                "observationNames:orderQueueSize|collectQueueSize|payBillQueueSize|kitchenCleanlinessLevel|timeOfDay",
                "observationTypes:double|double|double|double|double",
                "actions:4",
                "isActionMask:true",
                "rewardVariablesCount:4",
                "reward: not defined",
                "failedSteps:",
                "model-analyzer-mode: single",
                "agents:1",
                "rewardVariableNames:vars[0]|vars[1]|vars[2]|vars[3]",
                "rewardVariableTypes:int|double|boolean|float"
        ));

        HyperparametersDTO expected = new HyperparametersDTO(
                true,
                false,
                "5",
                Arrays.asList("orderQueueSize", "collectQueueSize", "payBillQueueSize", "kitchenCleanlinessLevel", "timeOfDay"),
                Arrays.asList("double", "double", "double", "double", "double"),
                "4",
                true,
                "4",
                Arrays.asList("vars[0]", "vars[1]", "vars[2]", "vars[3]"),
                Arrays.asList("int", "double", "boolean", "float"),
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
                "observationNames:orderQueueSize|collectQueueSize|payBillQueueSize|kitchenCleanlinessLevel|timeOfDay",
                "observationTypes:double|double|double|double|double",
                "actions:4",
                "isActionMask:true",
                "rewardVariablesCount:4",
                "reward: not defined",
                "failedSteps:",
                "model-analyzer-mode: single",
                "agents:1",
                "rewardVariableNames:vars[0]|vars[1]|vars[2]|vars[3]",
                "rewardVariableTypes:int|double|boolean|float"
        ));

        HyperparametersDTO expected = new HyperparametersDTO(
                true,
                false,
                "5",
                Arrays.asList("orderQueueSize", "collectQueueSize", "payBillQueueSize", "kitchenCleanlinessLevel", "timeOfDay"),
                Arrays.asList("double", "double", "double", "double", "double"),
                "4",
                true,
                "4",
                Arrays.asList("vars[0]", "vars[1]", "vars[2]", "vars[3]"),
                Arrays.asList("int", "double", "boolean", "float"),
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
                "observationNames:",
                "observationTypes:",
                "actions:4",
                "isActionMask:true",
                "rewardVariablesCount:4",
                "reward: not defined",
                "failedSteps:observations,observationsNames",
                "agents:1",
                "model-analyzer-mode: single",
                "rewardVariableNames:vars[0]|vars[1]|vars[2]|vars[3]",
                "rewardVariableTypes:int|double|boolean|float"
        ));

        HyperparametersDTO expected = new HyperparametersDTO(
                true,
                false,
                "0",
                Collections.emptyList(),
                Collections.emptyList(),
                "4",
                true,
                "4",
                Arrays.asList("vars[0]", "vars[1]", "vars[2]", "vars[3]"),
                Arrays.asList("int", "double", "boolean", "float"),
                "not defined",
                "observations,observationsNames",
                "1",
                "single"
        );
        Assertions.assertEquals(expected, result);
    }


}
