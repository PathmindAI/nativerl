package io.skymind.pathmind.analyzer.api.exception.handler;

import io.skymind.pathmind.analyzer.api.exception.dto.ApiErrorsResponse;
import io.skymind.pathmind.analyzer.exception.ProcessingException;
import io.skymind.pathmind.analyzer.exception.UnexpectedScriptResultException;
import io.skymind.pathmind.analyzer.exception.ZipExtractionException;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice
@Slf4j
public class ControllerExceptionHandler {

    @ExceptionHandler(ZipExtractionException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public ApiErrorsResponse handleZipExtractionException(ZipExtractionException ex){
        log.error(ex.getMessage());
        return new ApiErrorsResponse(HttpStatus.BAD_REQUEST.value(), ex.getMessage());
    }

    @ExceptionHandler(ProcessingException.class)
    @ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
    public ApiErrorsResponse handleProcessingException(ProcessingException ex){
        log.error(ex.getMessage());
        return new ApiErrorsResponse(HttpStatus.INTERNAL_SERVER_ERROR.value(), ex.getMessage());
    }

    @ExceptionHandler(UnexpectedScriptResultException.class)
    @ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
    public ApiErrorsResponse handleUnexpectedScriptResultException(UnexpectedScriptResultException ex){
        log.error(ex.getMessage());
        return new ApiErrorsResponse(HttpStatus.INTERNAL_SERVER_ERROR.value(), ex.getMessage());
    }
}
