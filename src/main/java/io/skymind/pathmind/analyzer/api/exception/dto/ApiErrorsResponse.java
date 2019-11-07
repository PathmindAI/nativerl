package io.skymind.pathmind.analyzer.api.exception.dto;

import io.swagger.annotations.ApiModelProperty;
import lombok.*;

import java.util.Collection;
import java.util.List;

@Getter
@Setter
@ToString
@NoArgsConstructor
@AllArgsConstructor
public class ApiErrorsResponse {
    @ApiModelProperty(value = "Code", example = "Response status code")
    private int code;
    @ApiModelProperty(value = "errors", example = "List of error messages detected during the processing the request")
    private Collection<String> errors;

    public ApiErrorsResponse(final int code, final String error) {
        this(code, List.of(error));
    }
}
