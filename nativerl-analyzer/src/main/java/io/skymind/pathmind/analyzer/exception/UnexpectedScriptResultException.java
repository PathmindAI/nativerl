package io.skymind.pathmind.analyzer.exception;

public class UnexpectedScriptResultException extends RuntimeException {
    public UnexpectedScriptResultException(String message) {
        super(message);
    }

    public UnexpectedScriptResultException(String message, Throwable cause) {
        super(message, cause);
    }
}
