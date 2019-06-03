package ai.skymind.skynet.spring.cloud.job.api

import java.net.URI

data class CloudJobSpec(
        val userId: Int,
        val simulationJarUri: URI,
        val mdpSpec: MdpSpec? = null // TODO: Actually Support using a user defined MDP, for now we just run the file as is
)