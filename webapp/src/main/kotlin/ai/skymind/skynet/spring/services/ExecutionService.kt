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
        val env = Environment(listOf(
                "mrezMd", // PathmindPolicy.jar, 2019-08-23
                "LZAENb", // conda
                "XeGNac", // Anylogic Base Environment: baseEnv.zip
                "doRCLd", // nativerl-1.0.0-SNAPSHOT-bin.zip, 2019-08-22
                "fDRBHd"  // OpenJDK8U-jdk_x64_linux_hotspot_8u222b10.tar.gz
        ))


        val rlConfig = RLConfig("policy.zip", env, model, mdp)
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