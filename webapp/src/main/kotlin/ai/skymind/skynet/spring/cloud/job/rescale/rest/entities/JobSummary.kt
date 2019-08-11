package ai.skymind.skynet.spring.cloud.job.rescale.rest.entities

import java.time.LocalDateTime

data class JobSummary (
       val clusterStatusDisplay: ClusterStatus?,
       val dateInserted: LocalDateTime,
       val name: String,
       val analysisNames: List<String>,
       val storage: Long,
       val jobStatus: JobSummaryStatus,
       val sharedWith: List<String>,
       val isVisible: Boolean,
       val owner: String,
       val id: String
)

data class ClusterStatus(
        val content: String,
        val labelClass: String,
        val useLabel: Boolean
)

data class JobSummaryStatus(
        val content: String,
        val labelClass: String,
        val useLabel: String
)