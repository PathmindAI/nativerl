package io.skymind.pathmind.analyzer.api.controller;


import com.fasterxml.jackson.databind.ObjectMapper;
import io.skymind.pathmind.analyzer.api.dto.AnalyzeRequestDTO;
import io.skymind.pathmind.analyzer.api.dto.HyperparametersDTO;
import io.skymind.pathmind.analyzer.service.FileService;
import io.swagger.annotations.ApiOperation;
import io.swagger.annotations.ApiParam;
import io.swagger.annotations.ApiResponse;
import io.swagger.annotations.ApiResponses;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.convert.converter.Converter;
import org.springframework.stereotype.Component;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;


@Component
class StringToRequestConverter implements Converter<String, AnalyzeRequestDTO> {

    @Autowired
    private ObjectMapper objectMapper;

    @Override
    @SneakyThrows
    public AnalyzeRequestDTO convert(String source) {
        return objectMapper.readValue(source, AnalyzeRequestDTO.class);
    }
}

@RestController
@RequestMapping(MainController.API_VERSION)
@RequiredArgsConstructor
@Slf4j
public class MainController {

    private static final String EXTRACT_HYPERPARAMETERS = "/extract-hyperparameters";
    private final FileService fileService;

    static final String API_VERSION = "/api/v1";

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
            @RequestParam(name = "file") final MultipartFile multipartFile,
            @RequestParam(name = "id", defaultValue="{\"id\":\"Not Defined : \"}") final AnalyzeRequestDTO request) throws IOException {
        log.info(String.format("Received a request for extracting hyperparameters %s ", request.getId()));
        final List<String> hyperparameters = fileService.processFile(multipartFile, request);
        HyperparametersDTO response = HyperparametersDTO.of(hyperparameters);
        log.info(String.format("Extracted Hyperparameters for %s : %s", request.getId(), response));
        return response;
    }
}