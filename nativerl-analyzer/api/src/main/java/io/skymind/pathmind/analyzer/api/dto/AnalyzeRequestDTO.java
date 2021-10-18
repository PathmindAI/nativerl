package io.skymind.pathmind.analyzer.api.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import com.fasterxml.jackson.annotation.JsonSetter;
import com.fasterxml.jackson.annotation.Nulls;

@NoArgsConstructor
@AllArgsConstructor
@Data
public class AnalyzeRequestDTO {
    public enum ModelType {
        ANY_LOGIC,
        PYTHON
    }

    private String id;
    @JsonSetter(nulls = Nulls.AS_EMPTY)
    private ModelType type = ModelType.ANY_LOGIC;
    @JsonSetter(nulls = Nulls.AS_EMPTY)
    private String mainAgent = "";
    @JsonSetter(nulls = Nulls.AS_EMPTY)
    private String experimentClass = "";
    @JsonSetter(nulls = Nulls.AS_EMPTY)
    private String experimentType = "";
    @JsonSetter(nulls = Nulls.AS_EMPTY)
    private String pathmindHelperClass = "";
    @JsonSetter(nulls = Nulls.AS_EMPTY)
    private String environment = "";
}
