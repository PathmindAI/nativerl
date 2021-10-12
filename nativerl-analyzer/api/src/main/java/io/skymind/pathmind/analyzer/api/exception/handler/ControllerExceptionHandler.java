package io.skymind.pathmind.analyzer.api.exception.handler;

import io.skymind.pathmind.analyzer.api.exception.dto.ApiErrorsResponse;
import io.skymind.pathmind.analyzer.dto.SimulationParameter;
import io.skymind.pathmind.analyzer.exception.InvalidZipFileException;
import io.skymind.pathmind.analyzer.exception.ProcessingException;
import io.skymind.pathmind.analyzer.exception.UnexpectedScriptResultException;
import io.skymind.pathmind.analyzer.exception.ZipExtractionException;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.multipart.MultipartException;

import static org.springframework.http.HttpStatus.BAD_REQUEST;
import static org.springframework.http.HttpStatus.INTERNAL_SERVER_ERROR;

@RestControllerAdvice
@Slf4j
public class ControllerExceptionHandler {

    @ExceptionHandler(ZipExtractionException.class)
    @ResponseStatus(BAD_REQUEST)
    public ApiErrorsResponse handleZipExtractionException(final ZipExtractionException ex) {
        log.error(ex.getMessage());
        return new ApiErrorsResponse(BAD_REQUEST.value(), ex.getMessage());
    }

    @ExceptionHandler(ProcessingException.class)
    @ResponseStatus(INTERNAL_SERVER_ERROR)
    public ApiErrorsResponse handleProcessingException(final ProcessingException ex) {
        log.error(ex.getMessage());
        return new ApiErrorsResponse(INTERNAL_SERVER_ERROR.value(), ex.getMessage());
    }

    @ExceptionHandler(UnexpectedScriptResultException.class)
    @ResponseStatus(INTERNAL_SERVER_ERROR)
    public ApiErrorsResponse handleUnexpectedScriptResultException(final UnexpectedScriptResultException ex) {
        log.error(ex.getMessage());
        return new ApiErrorsResponse(INTERNAL_SERVER_ERROR.value(), ex.getMessage());
    }

    @ExceptionHandler(InvalidZipFileException.class)
    @ResponseStatus(BAD_REQUEST)
    public ApiErrorsResponse handleInvalidZipFileException(final InvalidZipFileException ex) {
        log.error(ex.getMessage());
        return new ApiErrorsResponse(BAD_REQUEST.value(), ex.getMessage());
    }

    @ExceptionHandler(MultipartException.class)
    @ResponseStatus(BAD_REQUEST)
    public ApiErrorsResponse handleMultipartException(final MultipartException ex) {
        log.error(ex.getMessage());
        return new ApiErrorsResponse(BAD_REQUEST.value(), ex.getMessage());
    }

}
