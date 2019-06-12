package ai.skymind.skynet.spring.cloud.job.api

import ai.skymind.skynet.spring.cloud.job.local.RLConfig
import java.io.File

interface JobExecutor {
    fun upload(file: File): String

    fun run(rlConfig: RLConfig): String

    fun getConsoleOutput(jobId: String)
}