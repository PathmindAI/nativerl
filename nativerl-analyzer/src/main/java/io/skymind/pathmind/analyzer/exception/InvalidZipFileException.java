package io.skymind.pathmind.analyzer.exception;

public class InvalidZipFileException extends RuntimeException {
    public InvalidZipFileException(String message) {
        super(message);
    }

    public InvalidZipFileException(String message, Throwable cause) {
        super(message, cause);
    }
}
