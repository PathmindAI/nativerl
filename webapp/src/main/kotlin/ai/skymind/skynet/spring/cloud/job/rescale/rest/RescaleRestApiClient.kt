package ai.skymind.skynet.spring.cloud.job.rescale.rest

import ai.skymind.skynet.spring.cloud.job.rescale.rest.entities.*
import ai.skymind.skynet.spring.cloud.job.rescale.util.runCommand
import com.fasterxml.jackson.databind.ObjectMapper
import org.springframework.beans.factory.annotation.Value
import org.springframework.http.HttpStatus
import org.springframework.http.MediaType
import org.springframework.stereotype.Service
import org.springframework.web.reactive.function.client.WebClient
import reactor.core.publisher.toMono
import java.io.File
import java.nio.charset.Charset
import java.util.function.Function
import java.util.function.Predicate
import kotlin.streams.toList

@Service
class RescaleRestApiClient(
        @Value("\${skymind.rescale.platform.region}") val platformRegion: String,
        @Value("\${skymind.rescale.platform.key}") val apiKey: String,
        val objectMapper: ObjectMapper,
        webClientBuilder: WebClient.Builder
) {
    val client = webClientBuilder
            .baseUrl("https://$platformRegion/api/v2")
            .defaultHeader("Authorization", "Token $apiKey")
            .defaultHeader("Content-Type", MediaType.APPLICATION_JSON_VALUE)
            .build()

    //final inline fun <reified T> nextPage(pagedResult: T): T = client.get().uri((pagedResult as PagedResult<*>).next!!).retrieve().bodyToMono(T::class.java).block()!!

    fun jobList(): PagedResult<JobSummary> = client
            .get().uri("/jobs/?page_size=9999")
            .retrieve()
            .bodyToMono(typeReference<PagedResult<JobSummary>>()).block()!!

    fun jobCreate(job: Job): Job = client
            .post().uri("/jobs/")
            .contentType(MediaType.APPLICATION_JSON).body(job.toMono(), Job::class.java)
            .retrieve()
            .onStatus(Predicate.isEqual(HttpStatus.BAD_REQUEST), Function {it.bodyToMono(String::class.java).map { java.lang.RuntimeException(it) }})
            .bodyToMono(Job::class.java).block()!!

    fun jobSubmit(job: Job) {
        client.post().uri("/jobs/${job.id}/submit/")
                .retrieve().bodyToMono(Void::class.java).block()
    }

    fun jobStop(jobId: String) {
        client.post().uri("/jobs/$jobId/stop/")
                .retrieve().bodyToMono(Void::class.java).block()
    }

    fun jobDelete(jobId: String) {
        client.delete().uri("/jobs/$jobId/")
                .retrieve().bodyToMono(Void::class.java).block()
    }

    fun jobDetails(jobId: String): Job = client
            .get().uri("/jobs/$jobId/")
            .retrieve()
            .bodyToMono(Job::class.java).block()!!

    fun jobStatusHistory(jobId: String): PagedResult<JobStatus> = client
            .get().uri("/jobs/$jobId/statuses/")
            .retrieve()
            .bodyToMono(typeReference<PagedResult<JobStatus>>()).block()!!

    fun jobRuns(jobId: String): PagedResult<JobRun> = client
            .get().uri("/jobs/$jobId/runs/")
            .retrieve()
            .bodyToMono(typeReference<PagedResult<JobRun>>()).block()!!

    fun directoryContent(jobId: String, run: String): List<DirectoryFileReference> = client
            .get().uri("/jobs/$jobId/runs/$run/directory-contents/?page_size=1000")
            .retrieve()
            .bodyToFlux(DirectoryFileReference::class.java)
            .toStream().toList()

    fun outputFiles(jobId: String): PagedResult<RescaleFile> = client
            .get().uri("/jobs/$jobId/runs/1/files/?page_size=1000")
            .retrieve()
            .bodyToMono(typeReference<PagedResult<RescaleFile>>()).block()!!

    fun tailConsole(jobId: String, run: String) = client
            .get().uri("/jobs/$jobId/runs/$run/tail/process_output.log")
            .retrieve()
            .bodyToMono(String::class.java).block()!!

    fun fileContents(fileId: String): ByteArray = client
            .get().uri("/files/${fileId}/contents/").retrieve().bodyToMono(ByteArray::class.java).block()!!

    fun consoleOutput(jobId: String): String {
        val outputFiles = outputFiles(jobId)
        val consoleFile = outputFiles.results.find { it.name == "process_output.log" && it.isUploaded && !it.isDeleted}!!
        return fileContents(consoleFile.id).toString(Charset.forName("UTF-8"))
    }

    fun compileOutput(jobId: String): String {
        val outputFiles = outputFiles(jobId)
        val consoleFile = outputFiles.results.find { it.name == "compile.out.txt" && it.isUploaded && !it.isDeleted}!!
        return fileContents(consoleFile.id).toString(Charset.forName("UTF-8"))
    }

    /**
     * Uses a hack to work around the fact that rescale doesn't support `Transfer-Encoding: chunked` for file uploads,
     * and WebClient doesn't properly support not using it for file uploads
     */
    fun fileUpload(content: File): RescaleFile {
        if(!content.exists()) {throw IllegalArgumentException("The to be uploaded file $content does not exist!")}

        val command = listOf("curl",
                "-X", "POST",
                "-H", "Content-Type:multipart/form-data",
                "-H", "Authorization: Token $apiKey",
                "-F", "file=@${content.absolutePath}",
                "https://$platformRegion/api/v2/files/contents/"
        )

        val result = command.runCommand(content.parentFile)
        if(result.exitValue != 0){ throw RuntimeException("Could not upload given file. Error: $result") }

        return objectMapper.readValue(result.stdOut, RescaleFile::class.java)
    }

    fun filesList(): PagedResult<RescaleFile> = client
            .get().uri("/files/?page_size=9999")
            .retrieve()
            .bodyToMono(typeReference<PagedResult<RescaleFile>>()).block()!!

    fun deleteFile(fileId: String) {
        client.delete().uri("/files/$fileId/")
                .retrieve()
                .bodyToMono(Void::class.java)
                .block()
    }
}