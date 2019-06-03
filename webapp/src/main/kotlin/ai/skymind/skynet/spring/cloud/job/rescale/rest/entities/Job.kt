package ai.skymind.skynet.spring.cloud.job.rescale.rest.entities

// Stubbed out to actually only require the input that we need
data class Job (
        val name: String,
        val jobanalyses: List<JobAnalysis>,
        val id: String? = null,
        val paramFile: String? = null,
        val caseFile: String? = null,
        val resourceFilters: List<Any>? = null,
        val jobvariables: List<Any> = emptyList(),
        val isTemplateDryRun: Boolean = false,
        val includeNominalRun: Boolean = false,
        val monteCarloIterations: Int? = null
)