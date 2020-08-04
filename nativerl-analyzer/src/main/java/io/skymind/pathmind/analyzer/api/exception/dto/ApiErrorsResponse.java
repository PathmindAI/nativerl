package io.skymind.pathmind.analyzer.api.exception.dto;

import io.swagger.annotations.ApiModelProperty;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
@ToString
@AllArgsConstructor
public class ApiErrorsResponse {
    @ApiModelProperty(value = "Code", example = "Response status code")
    private int code;
    @ApiModelProperty(value = "Error", example = "Error message detected during the processing the request")
    private String error;
}
