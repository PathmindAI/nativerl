package ai.skymind.skynet.spring.services

import org.springframework.stereotype.Service
import java.time.Instant

@Service
class ExperimentService {
    val items = mutableListOf(
        Experiment("Simple")
    )

    fun find(query: String) = if(query.isNullOrBlank()) items else items.filter { it.name.contains(query, true)}
    fun findAll() = items

    fun add(experiment: Experiment) = items.add(experiment)
}

data class Experiment(
        val name: String,
        val createdAt: Instant = Instant.now(),
        val runs: Int = 0
)