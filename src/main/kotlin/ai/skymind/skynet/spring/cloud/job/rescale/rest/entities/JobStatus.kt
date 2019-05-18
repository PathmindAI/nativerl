package ai.skymind.skynet.spring.cloud.job.rescale.rest.entities

import java.time.LocalDateTime


data class JobStatus(
        val status: String,
        val statusDate: LocalDateTime,
        val statusReason: String?
)