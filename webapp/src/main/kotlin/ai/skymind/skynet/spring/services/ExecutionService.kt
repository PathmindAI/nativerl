package ai.skymind.skynet.spring.services

import ai.skymind.skynet.data.db.jooq.tables.records.MdpRecord
import ai.skymind.skynet.data.db.jooq.tables.records.RunRecord
import ai.skymind.skynet.spring.cloud.job.api.JobExecutor
import ai.skymind.skynet.spring.cloud.job.local.Environment
import ai.skymind.skynet.spring.cloud.job.local.RLConfig
import ai.skymind.skynet.spring.db.ModelRepository
import ai.skymind.skynet.spring.db.RunRepository
import org.springframework.stereotype.Service

@Service
class ExecutionService(
        val executor: JobExecutor,
        val modelRepository: ModelRepository,
        val runRepository: RunRepository
) {
    fun runMdp(mdp: MdpRecord) {
        val model = modelRepository.findById(mdp.userId, mdp.modelId)
        val env = Environment(listOf("qBaAAd", "rbOcJd", "LZAENb", "XeGNac"))


        val rlConfig = RLConfig("PhasePolicy.zip", env, model, mdp)
        val jobId = executor.run(rlConfig)

        RunRecord().apply {
            externalJobId = jobId
            userId = mdp.userId
            modelId = mdp.modelId
            mdpId = mdp.id
            status = "SUBMITTED"
        }.let {
            runRepository.add(it)
        }

    }
}