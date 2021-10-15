package io.skymind.pathmind.analyzer.api.controller;


import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.skymind.pathmind.analyzer.api.dto.AnalyzeRequestDTO;
import io.skymind.pathmind.analyzer.dto.HyperparametersDTO;
import io.skymind.pathmind.analyzer.dto.SimulationParameter;
import io.skymind.pathmind.analyzer.service.FileService;
import io.swagger.annotations.ApiOperation;
import io.swagger.annotations.ApiParam;
import io.swagger.annotations.ApiResponse;
import io.swagger.annotations.ApiResponses;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.convert.converter.Converter;
import org.springframework.stereotype.Component;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import javax.validation.constraints.NotEmpty;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;


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
            @RequestParam(name = "id", defaultValue="{\"id\":\"Not Defined : \"}") final AnalyzeRequestDTO request) throws IOException, InterruptedException {
        log.info(String.format("Received a request for extracting hyperparameters %s ", request.getId()));
        final List<String> hyperparameters = fileService.processFile(multipartFile, request);
        HyperparametersDTO response = ParametersUtils.of(hyperparameters);
        log.info(String.format("Extracted Hyperparameters for %s : %s", request.getId(), response));
        return response;
    }

    public static class ParametersUtils {
        public static HyperparametersDTO of(@NotEmpty List<String> hyperparametersList) {
            Map<String, String> parametersMap = hyperparametersList.stream()
                    .map(String::strip)
                    .map(l -> l.split(":", 2))
                    .filter(p -> HyperparametersDTO.isHyperparameters(p[0]))
                    .collect(Collectors.toMap(p -> p[0], p -> p[1].strip()));
            ObjectMapper objectMapper = new ObjectMapper();

            String dtoFilePath = parametersMap.getOrDefault("DTOPath", "");
            try {
                String json = FileUtils.readFileToString(new File(dtoFilePath), Charset.defaultCharset());
                return objectMapper.readValue(json, HyperparametersDTO.class);
            } catch (JsonProcessingException e) {
                e.printStackTrace();
                return null;
            } catch (IOException e) {
                String maMode = parametersMap.get("model-analyzer-mode");
                if (maMode != null && maMode.startsWith("py_")) {
                    HyperparametersDTO dto = new HyperparametersDTO();
                    dto.setMode(maMode);
                    return dto;
                }
                e.printStackTrace();
                return null;
            }
        }

        public static List<SimulationParameter> asParamList(String mapString) {
            ObjectMapper objectMapper = new ObjectMapper();
            try {
                return objectMapper.readValue(mapString, List.class);
            } catch (JsonProcessingException e) {
                e.printStackTrace();
                return null;
            }
        }
    }

}