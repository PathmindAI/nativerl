package ai.skymind.skynet.spring.cloud.job.api

import java.io.File

interface CloudJobExecutor {
    fun upload(file: File): String // Returns File ID
    fun run(spec: CloudJobSpec): String // Returns Job ID
    fun status(jobId: String): CloudJobStatus
    fun result(jobId: String): CloudJobResult
}