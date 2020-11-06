package io.skymind.pathmind.analyzer.service;

import io.skymind.pathmind.analyzer.exception.InvalidZipFileException;
import io.skymind.pathmind.analyzer.exception.ZipExtractionException;
import lombok.extern.slf4j.Slf4j;
import net.lingala.zip4j.ZipFile;
import net.lingala.zip4j.exception.ZipException;
import org.apache.commons.io.IOUtils;
import org.springframework.stereotype.Service;
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

    public List<String> processFile(final MultipartFile multipartFile) throws IOException {
        log.debug("Processing file {} started", multipartFile.getName());
        final Path unzippedPath = unzipFile(multipartFile);
        return extractParameters(unzippedPath);
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

    private List<String> extractParameters(final Path unzippedPath) throws IOException {
        final File scriptFile = Paths.get(CHECK_MODEL_SCRIPT).toFile();
        final File newFile = new File(unzippedPath.normalize().toString(), scriptFile.getName());
        FileCopyUtils.copy(scriptFile, newFile);

        return runExtractorScript(unzippedPath, newFile);
    }

    private List<String> runExtractorScript(final Path unzippedPath, File newFile) throws IOException {
        final String[] cmd = new String[]{"bash", newFile.getAbsolutePath(), newFile.getParentFile().getAbsolutePath()};
        final Process proc = Runtime.getRuntime().exec(cmd);
        List<String> result = readResult(proc.getInputStream());
        log.info("Bash script finished");

        if (result.size() < 15) {
            List<String> err = readResult(proc.getErrorStream());
            log.warn("Unexpected output for {} file, result: {}, err: {}", unzippedPath, String.join(" ", result), String.join(" ", err));
        }

        return result;
    }

    private List<String> readResult(final InputStream inputStream) {
        return new BufferedReader(new InputStreamReader(inputStream)).lines().collect(Collectors.toList());
    }
}
