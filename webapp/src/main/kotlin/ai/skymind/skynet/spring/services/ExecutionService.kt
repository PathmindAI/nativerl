package ai.skymind.skynet.spring.services

import ai.skymind.skynet.data.db.jooq.tables.records.MdpRecord
import ai.skymind.skynet.spring.cloud.job.api.JobExecutor
import ai.skymind.skynet.spring.cloud.job.local.Environment
import ai.skymind.skynet.spring.cloud.job.local.RLConfig
import ai.skymind.skynet.spring.db.ModelRepository
import org.springframework.stereotype.Service

@Service
class ExecutionService(
        val executor: JobExecutor,
        val modelRepository: ModelRepository
) {
    fun runMdp(mdp: MdpRecord) {
        val model = modelRepository.findById(mdp.userId, mdp.modelId)
        val env = Environment(listOf("VNNaQb", "XeGNac")) // jp file ids
        //val env = Environment(listOf("vbqfEc", "KDiSPc")) // eu file ids

        val rlConfig = RLConfig("PhasePolicy.zip", env, model, mdp)
        executor.run(rlConfig)
    }
}