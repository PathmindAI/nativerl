package ai.skymind.skynet.spring.cloud.job.rescale

import ai.skymind.skynet.spring.cloud.job.api.CloudJobSpec
import ai.skymind.skynet.spring.cloud.job.rescale.rest.RescaleRestApiClient
import com.fasterxml.jackson.databind.ObjectMapper
import org.junit.Assert.*
import org.junit.Ignore
import org.junit.Test
import org.springframework.web.reactive.function.client.WebClient
import java.io.File
import java.net.URI

class RescaleJobExecutorTest {
    val apiClient = RescaleRestApiClient(
            "platform.rescale.jp",
            "0d0601925a547db44d41007e3cc4386b075c761c",
            ObjectMapper().findAndRegisterModules(),
            WebClient.builder()
    )

    @Ignore
    @Test
    fun run() {

        val executor = RescaleJobExecutor(apiClient)
        val jobId = executor.run(
                CloudJobSpec(
                        userId = 0xBEEF,
                        simulationJarUri = URI("file://X:/rescale-test.jar")
                )
        )

        println(jobId)
        assertNotNull(jobId)
    }

    @Ignore
    @Test
    fun status() {
        val executor = RescaleJobExecutor(apiClient)
        executor.status("oYoTdb")
    }

    @Ignore
    @Test
    fun result() {
        val executor = RescaleJobExecutor(apiClient)
        executor.result("oYoTdb")
    }


    @Ignore
    @Test
    fun stop() {

        apiClient.jobStop("GWQop")
    }

    @Ignore
    @Test
    fun uploadFile(){
        val input = File("X:/hello-world.123.txt")
        val uploaded = apiClient.fileUpload(input)
        val file = apiClient.filesList().results.find { it.name == "hello-world.123.txt" && it.id == uploaded.id }

        assertNotNull(file)
        assertEquals(uploaded.id, file!!.id)
        val content = input.readBytes()
        val uploadedContent = apiClient.fileContents(uploaded.id)

        assertArrayEquals(content, uploadedContent)
        apiClient.deleteFile(uploaded.id)
        Thread.sleep(15000)
        assertNull(apiClient.filesList().results.find { it.id == uploaded.id})
    }

    @Ignore
    @Test
    fun listFiles(){
        println(apiClient.filesList())
    }

    @Ignore
    @Test
    fun deleteFile(){
        apiClient.filesList().results.filter { it.name == "hello-world.123.txt"}.forEach{apiClient.deleteFile(it.id)}
    }
}