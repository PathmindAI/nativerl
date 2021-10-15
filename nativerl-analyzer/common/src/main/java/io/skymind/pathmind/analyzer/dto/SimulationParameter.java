package io.skymind.pathmind.analyzer.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;

@AllArgsConstructor
@NoArgsConstructor
@Getter
public class SimulationParameter {
    private Integer index;
    private String key;
    private String value;
    private Integer type;
}
