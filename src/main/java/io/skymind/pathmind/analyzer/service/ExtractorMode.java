package io.skymind.pathmind.analyzer.service;

import java.util.stream.Stream;

public enum ExtractorMode {
    SINGLE_AGENT("1", "single"), MULTI_AGENT("2", "multi");
    private String hyperparametersDimension;
    private String mode;

    ExtractorMode(String hyperparametersDimension, String mode) {
        this.hyperparametersDimension = hyperparametersDimension;
        this.mode = mode;
    }

    public String getHyperparametersDimension() {
        return hyperparametersDimension;
    }

    public static ExtractorMode getByhyperparametersDimension(String dim) {
        return Stream.of(ExtractorMode.values())
                .filter(e -> e.getHyperparametersDimension().equals(dim))
                .findAny()
                .orElse(ExtractorMode.SINGLE_AGENT);
    }

    @Override
    public String toString() {
        return mode;
    }
}
