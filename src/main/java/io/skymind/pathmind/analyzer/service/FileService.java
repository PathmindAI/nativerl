package io.skymind.pathmind.analyzer.service;

import io.skymind.pathmind.analyzer.exception.UnexpectedScriptResultException;
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
    private static final String SCRIPT_LOCALIZATION = "/bin/check_model.sh";

    public List<String> processFile(MultipartFile multipartFile) throws IOException {
        log.debug("Processing file {} started", multipartFile.getName());
        final Path unzippedPath = unzipFile(multipartFile);
        return extractParameters(unzippedPath);
    }

    private Path unzipFile(MultipartFile multipartFile) throws IOException {
        final UUID uuid = UUID.randomUUID();
        final Path tempDirectory = Files.createTempDirectory(uuid.toString());
        final File tempFile = Files.createFile(tempDirectory.resolve(TEMP_FILE_PREFIX + System.nanoTime())).toFile();

        try (FileOutputStream outputStream = new FileOutputStream(tempFile)) {
            IOUtils.copy(multipartFile.getInputStream(), outputStream);
        }

        ZipFile zipFile = new ZipFile(tempFile);
        try {
            zipFile.extractAll(tempFile.getParentFile().getPath());
        } catch (ZipException e) {
            throw new ZipExtractionException(ZIP_EXTRACTION_EXCEPTION_MESSAGE, e.getCause());
        }
        log.debug("Archive {} unzipped successfully", zipFile.getFile().getName());
        return tempFile.getParentFile().toPath();
    }

    private List<String> extractParameters(Path unzippedPath) throws IOException {
        final File scriptFile = Paths.get(SCRIPT_LOCALIZATION).toFile();
        File newFile = new File(unzippedPath.normalize().toString(), scriptFile.getName());
        FileCopyUtils.copy(scriptFile, newFile);

        String[] cmd = new String[]{"bash", newFile.getAbsolutePath(), newFile.getParentFile().getAbsolutePath()};
        Process proc = Runtime.getRuntime().exec(cmd);
        List<String> result = readResult(proc.getInputStream());
        log.info("Bash script finished");

        if (result.size() != 2) {
            final String errorMessage = String.join(" ", result);
            throw new UnexpectedScriptResultException(errorMessage);
        }
        return result;
    }

    private List<String> readResult(InputStream inputStream) {
        return new BufferedReader(new InputStreamReader(inputStream))
                .lines()
                .collect(Collectors.toList());
    }
}
