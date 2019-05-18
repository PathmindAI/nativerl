package ai.skymind.skynet.spring.cloud.job.api

interface CloudJobExecutor {
    fun run(spec: CloudJobSpec): String // Returns Job ID
    fun status(jobId: String): CloudJobStatus
    fun result(jobId: String): CloudJobResult
}