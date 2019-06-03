package ai.skymind.skynet.spring.cloud.job.api

data class MdpSpec(
        val getActionSpace: String,
        val getObservationSpace: String,
        val getObservation: String,
        val reset: String,
        val step: String,

        // Nullable Variables are optional, and will use default values if not set.
        val variables: String? = null,
        val close: String? = null,
        val isDone: String? = null
)