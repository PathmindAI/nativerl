package io.skymind.pathmind.analyzer.api.controller;


import io.skymind.pathmind.analyzer.api.dto.HyperparametersDTO;
import io.skymind.pathmind.analyzer.service.FileService;
import io.swagger.annotations.*;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.*;
import java.util.List;

@RestController
@RequiredArgsConstructor
@Slf4j
public class MainController {

    private static final String EXTRACT_HYPERPARAMETERS = "/extract-hyperparameters";
    private final FileService fileService;

    @PostMapping(EXTRACT_HYPERPARAMETERS)
    @ApiOperation(value = "Extracting hyperparameters from model",
            notes = "This operation is a helper for Pathmind platform for extracting set of hyperparameters which " +
                    "users set up in their simulation models in AnyLogic. It takes a file as an input, starts a " +
                    "script to setup environment required for an extraction and then responses with a proper value.")
    @ApiResponses(value = {
            @ApiResponse(code = 200, message = "Processed successfully"),
            @ApiResponse(code = 400, message = "Invalid request. There might be a problem with file")
    })
    public HyperparametersDTO extractHyperparameters(
            @ApiParam(value = "Valid ZIP archive contains all needed files to set up environment for extract hyperparameters.")
            @RequestParam("file")
                    MultipartFile multipartFile) throws IOException {

        log.info("Received a request for extracting hyperparameters");
        final List<String> hyperparameters = fileService.processFile(multipartFile);
        return HyperparametersDTO.of(hyperparameters);
    }
}