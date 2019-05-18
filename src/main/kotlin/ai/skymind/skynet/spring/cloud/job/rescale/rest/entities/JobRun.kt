package ai.skymind.skynet.spring.cloud.job.rescale.rest.entities

import java.time.LocalDateTime

data class JobRun(
        val dateCompleted: LocalDateTime?,
        val dateInserted: LocalDateTime,
        val displayOrder: Int,
        val id: String,
        val isOptimal: Boolean,
        val outputFileCount: Int,
        val outputFileSize: Int,
        val parent: String,
        val type: Int, // 1: Optimization, 2: Iteration, 3: Case
        val variables: List<Variable>
){
    data class Variable(
            val displayName: String,
            val isRelative: Boolean,
            val name: String,
            val value: Float
    )
}