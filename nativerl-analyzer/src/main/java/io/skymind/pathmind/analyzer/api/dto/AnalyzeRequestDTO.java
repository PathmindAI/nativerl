package io.skymind.pathmind.analyzer.api.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@NoArgsConstructor
@AllArgsConstructor
@Data
public class AnalyzeRequestDTO {
    private String id;
    private String mainAgent;
    private String experimentClass;
    private String experimentType;
    private String pathmindHelperClass;
}