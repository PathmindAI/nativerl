package ai.skymind.skynet.spring.cloud.job.rescale

import ai.skymind.skynet.spring.cloud.job.api.CloudJobExecutor
import ai.skymind.skynet.spring.cloud.job.api.CloudJobResult
import ai.skymind.skynet.spring.cloud.job.api.CloudJobSpec
import ai.skymind.skynet.spring.cloud.job.api.CloudJobStatus
import ai.skymind.skynet.spring.cloud.job.rescale.rest.RescaleRestApiClient
import ai.skymind.skynet.spring.cloud.job.rescale.rest.entities.Job
import ai.skymind.skynet.spring.cloud.job.rescale.rest.entities.JobAnalysis

class RescaleJobExecutor(val apiClient: RescaleRestApiClient): CloudJobExecutor {
    override fun run(spec: CloudJobSpec): String {
        val job = apiClient.jobCreate(Job(
                name = "${spec.userId}-job",
                jobanalyses = listOf(JobAnalysis(
                        "java -version; gcc --version; cmake --version; python --version; python3 --version"
                ))
        ))

        apiClient.jobSubmit(job)

        return job.id!!
    }

    override fun status(jobId: String): CloudJobStatus {
        val job = apiClient.jobDetails(jobId)

        println(apiClient.jobStatusHistory(jobId))
        println(apiClient.jobRuns(jobId))
        return CloudJobStatus(job.id!!)
    }

    override fun result(jobId: String): CloudJobResult {
        println(apiClient.consoleOutput(jobId))

        return CloudJobResult(jobId)
    }
}