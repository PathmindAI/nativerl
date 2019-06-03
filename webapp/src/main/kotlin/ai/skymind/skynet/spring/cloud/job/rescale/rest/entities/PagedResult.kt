package ai.skymind.skynet.spring.cloud.job.rescale.rest.entities

import org.springframework.core.ParameterizedTypeReference
import java.net.URI

inline fun <reified T> typeReference() = object: ParameterizedTypeReference<T>() {}

data class PagedResult<T>(
        val count: Int,
        val previous: URI? = null,
        val next: URI? = null,
        val results: List<T>
)