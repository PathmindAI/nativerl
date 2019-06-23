package ai.skymind.skynet.spring.cloud.job.api

import ai.skymind.skynet.spring.cloud.job.local.RLConfig
import java.io.File
import java.io.InputStream

interface JobExecutor {
    fun upload(file: File): String

    fun run(rlConfig: RLConfig): String
    fun stop(jobId: String)

    fun getConsoleOutput(jobId: String): String
    fun tailConsoleOutput(jobId: String): String
    fun status(jobId: String): String
    fun getPolicy(jobId: String): InputStream
}