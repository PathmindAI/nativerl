package ai.skymind.skynet.spring.cloud.job.rescale.rest

import ai.skymind.skynet.spring.cloud.job.rescale.rest.entities.*
import org.springframework.beans.factory.annotation.Value
import org.springframework.http.HttpStatus
import org.springframework.http.MediaType
import org.springframework.stereotype.Service
import org.springframework.web.reactive.function.client.WebClient
import reactor.core.publisher.toMono
import java.nio.charset.Charset
import java.util.function.Function
import java.util.function.Predicate

@Service
class RescaleRestApiClient(
        @Value("\${skymind.rescale.platform.region}") val platformRegion: String,
        @Value("\${skymind.rescale.platform.key}") val apiKey: String,
        webClientBuilder: WebClient.Builder
) {
    val client = webClientBuilder
            .baseUrl("https://$platformRegion/api/v2")
            .defaultHeader("Authorization", "Token $apiKey")
            .defaultHeader("Content-Type", MediaType.APPLICATION_JSON_VALUE)
            .build()

    //inline fun <reified T> nextPage(page: PagedResult<T>): PagedResult<T> = client.get().uri(page.next!!).retrieve().bodyToMono(PagedResult.of<T>()).block()!!
    //inline fun <reified T> previousPage(page: PagedResult<T>): PagedResult<T> = client.get().uri(page.previous!!).retrieve().bodyToMono(PagedResult.of<T>()).block()!!

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

    fun outputFiles(jobId: String): PagedResult<OutputFile> = client
            .get().uri("/jobs/$jobId/runs/1/files/")
            .retrieve()
            .bodyToMono(typeReference<PagedResult<OutputFile>>()).block()!!

    fun getFileContents(fileId: String): ByteArray = client
            .get().uri("/files/${fileId}/contents/").retrieve().bodyToMono(ByteArray::class.java).block()!!

    fun consoleOutput(jobId: String): String {
        val outputFiles = outputFiles(jobId)
        val consoleFile = outputFiles.results.find { it.name == "process_output.log" && it.isUploaded && !it.isDeleted}!!
        return getFileContents(consoleFile.id).toString(Charset.forName("UTF-8"))
    }
}