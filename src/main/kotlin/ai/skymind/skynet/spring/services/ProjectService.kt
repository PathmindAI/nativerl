package ai.skymind.skynet.spring.services

import org.springframework.stereotype.Service
import java.time.LocalDateTime

data class Project(val modelName: String, val dateCreated: LocalDateTime = LocalDateTime.now())

@Service
class ProjectService(){
    val items = mutableListOf(
            Project("example"),
            Project("example 1"),
            Project("example 2")
    )

    fun find(query: String) = if(query.isNullOrBlank()) items else items.filter { it.modelName.contains(query, true)}
    fun findAll() = items

    fun add(project: Project) = items.add(project)
}