package ai.skymind.skynet.spring.services

import org.springframework.stereotype.Service
import java.time.Instant

data class Project(
        val name: String,
        val owner: String? = null,
        val createdAt: Instant = Instant.now(),
        val models: List<Model> = emptyList(),
        val mdps: List<Mdp> = emptyList(),
        val runs: List<Run> = emptyList(),
        val policies: List<Policy> = emptyList()
)

data class Model(
        val name: String,
        val fileId: String,
        val createdAt: Instant = Instant.now()
)

data class Mdp(
        val name: String,
        val modelName: String,
        val code: String,
        val verified: Boolean = false
)

data class Run(
        val id: Int,
        val modelName: String,
        val MdpName: String,
        val status: Status,
        val startedAt: Instant = Instant.now(),
        val finishedAt: Instant? = null
){
    enum class Status {RUNNING, FINISHED}
}

data class Policy(
        val runId: Int,
        val modelName: String,
        val mdpName: String,
        val fileId: String,
        val createdAt: Instant = Instant.now()
)

@Service
class ProjectService(){
    val items = mutableListOf(
            Project("example"),
            Project("example 1"),
            Project("example 2")
    )

    fun find(query: String) = if(query.isNullOrBlank()) items else items.filter { it.name.contains(query, true)}
    fun findAll() = items

    fun add(project: Project) = items.add(project)
}