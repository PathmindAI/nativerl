package io.skymind.pathmind.analyzer.service;

import io.skymind.pathmind.analyzer.exception.InvalidZipFileException;
import io.skymind.pathmind.analyzer.exception.ZipExtractionException;
import lombok.extern.slf4j.Slf4j;
import net.lingala.zip4j.ZipFile;
import net.lingala.zip4j.exception.ZipException;
import org.apache.commons.io.IOUtils;
import org.springframework.stereotype.Service;
import org.springframework.util.CollectionUtils;
import org.springframework.util.FileCopyUtils;
import org.springframework.web.multipart.MultipartFile;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
@Slf4j
public class FileService {

    private static final String TEMP_FILE_PREFIX = "temp-file-";
    private static final String ZIP_EXTRACTION_EXCEPTION_MESSAGE = "There was an error while ZIP extracting";
    private static final String INVALID_ZIP_EXCEPTION_MESSAGE = "%s file is an invalid ZIP";
    private static final String CHECK_MODEL_SCRIPT = "/bin/check_model.sh";
    private static final String SINGLE_OR_MULTI_SCRIPT = "/bin/check_single_or_multi.sh";

    public List<String> processFile(final MultipartFile multipartFile) throws IOException {
        log.debug("Processing file {} started", multipartFile.getName());
        final Path unzippedPath = unzipFile(multipartFile);
        final ExtractorMode mode = verifyMode(unzippedPath);
        return extractParameters(unzippedPath, mode);
    }

    private ExtractorMode verifyMode(Path unzippedPath) throws IOException {
        File scriptFile = Paths.get(SINGLE_OR_MULTI_SCRIPT).toFile();
        final File newFile = new File(unzippedPath.normalize().toString(), scriptFile.getName());
        FileCopyUtils.copy(scriptFile, newFile);

        final String[] cmd = new String[]{"bash", newFile.getAbsolutePath(), newFile.getParentFile().getAbsolutePath()};
        final Process proc = Runtime.getRuntime().exec(cmd);
        List<String> result = readResult(proc.getInputStream());

        String modeResult = !CollectionUtils.isEmpty(result) ? result.get(0) : "1";
        ExtractorMode extractorMode = ExtractorMode.getByhyperparametersDimension(modeResult);
        log.info("Mode for path {} is: {}", unzippedPath, extractorMode);
        return extractorMode;
    }

    private Path unzipFile(final MultipartFile multipartFile) throws IOException {
        final UUID uuid = UUID.randomUUID();
        final Path tempDirectory = Files.createTempDirectory(uuid.toString());
        final File tempFile = Files.createFile(tempDirectory.resolve(TEMP_FILE_PREFIX + System.nanoTime())).toFile();

        try (final FileOutputStream outputStream = new FileOutputStream(tempFile)) {
            IOUtils.copy(multipartFile.getInputStream(), outputStream);
        }

        final ZipFile zipFile = new ZipFile(tempFile);
        verifyFile(zipFile, multipartFile.getOriginalFilename());

        try {
            zipFile.extractAll(tempFile.getParentFile().getPath());
        } catch (final ZipException e) {
            throw new ZipExtractionException(ZIP_EXTRACTION_EXCEPTION_MESSAGE, e.getCause());
        }
        log.debug("Archive {} unzipped successfully", zipFile.getFile().getName());
        return tempFile.getParentFile().toPath();
    }

    private void verifyFile(final ZipFile zipFile, final String fileName) {
        if (!zipFile.isValidZipFile()) {
            throw new InvalidZipFileException(String.format(INVALID_ZIP_EXCEPTION_MESSAGE, fileName));
        }
    }

    private List<String> extractParameters(final Path unzippedPath, ExtractorMode mode) throws IOException {
        final File scriptFile = Paths.get(CHECK_MODEL_SCRIPT).toFile();
        final File newFile = new File(unzippedPath.normalize().toString(), scriptFile.getName());
        FileCopyUtils.copy(scriptFile, newFile);

        return runExtractorScript(unzippedPath, newFile, mode);
    }

    private List<String> runExtractorScript(final Path unzippedPath, File newFile, ExtractorMode mode) throws IOException {
        final String[] cmd = new String[]{"bash", newFile.getAbsolutePath(), newFile.getParentFile().getAbsolutePath(), mode.toString()};
        final Process proc = Runtime.getRuntime().exec(cmd);
        List<String> result = readResult(proc.getInputStream());
        log.info("Bash script finished");

        if (result.size() != 5) {
            log.warn("Unexpected output for {} file ({} mode): {}", unzippedPath, mode, String.join(" ", result));

//            boolean runMultiAgent = shouldRunMultiAgentMode(result);
//
//            if(mode == SINGLE_AGENT && runMultiAgent) {
//                return runExtractorScript(unzippedPath, newFile, MULTI_AGENT);
//            }
        }
        result.add("model-analyzer-mode:" + mode.toString());
        return result;
    }

    /**
     * NPE may be thrown because model was multi-agent
     */
    private boolean shouldRunMultiAgentMode(List<String> result) {
        return result.stream()
                .anyMatch(msg -> msg.contains("java.lang.NullPointerException"));
    }

    private List<String> readResult(final InputStream inputStream) {
        return new BufferedReader(new InputStreamReader(inputStream))
                .lines()
                .collect(Collectors.toList());
    }
}
