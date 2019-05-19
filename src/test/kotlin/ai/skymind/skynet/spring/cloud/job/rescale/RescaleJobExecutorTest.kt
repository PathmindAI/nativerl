package ai.skymind.skynet.spring.cloud.job.rescale

import ai.skymind.skynet.spring.cloud.job.api.CloudJobSpec
import ai.skymind.skynet.spring.cloud.job.rescale.rest.RescaleRestApiClient
import org.junit.Assert.assertNotNull
import org.junit.Ignore
import org.junit.Test
import org.springframework.web.reactive.function.client.WebClient
import java.io.File
import java.net.URI

class RescaleJobExecutorTest {
    val apiClient = RescaleRestApiClient(
            "platform.rescale.jp",
            "0d0601925a547db44d41007e3cc4386b075c761c",
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
        println(apiClient.uploadFile("hello-world.txt", File("X:/hello-world.123.txt")))
    }

}