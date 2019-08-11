package ai.skymind.skynet.spring.cloud.job.rescale

import ai.skymind.skynet.spring.cloud.job.api.JobExecutor
import ai.skymind.skynet.spring.cloud.job.local.RLConfig
import ai.skymind.skynet.spring.cloud.job.rescale.rest.RescaleRestApiClient
import ai.skymind.skynet.spring.cloud.job.rescale.rest.entities.Job
import ai.skymind.skynet.spring.cloud.job.rescale.rest.entities.JobAnalysis
import org.springframework.stereotype.Service
import java.io.File
import java.io.InputStream

private val String?.escaped: String
    get() {
        return this?.replace("'".toRegex(), "\\'") ?: ""
    }

@Service
class RllibRescaleJobExecutor(val apiClient: RescaleRestApiClient): JobExecutor {
    override fun run(rlConfig: RLConfig): String {
        val actions = listOf(
                // SETUP Variables
                "export CLASS_SNIPPET='${rlConfig.mdp.variables.escaped}';",
                "export RESET_SNIPPET='${rlConfig.mdp.reset.escaped}';",
                "export REWARD_SNIPPET='${rlConfig.mdp.reward.escaped}';",
                "export METRICS_SNIPPET='${rlConfig.mdp.metrics.escaped}';",
                "export DISCRETE_ACTIONS=${rlConfig.mdp.actionSpaceSize};",
                "export CONTINUOUS_OBSERVATIONS=${rlConfig.mdp.observationSpaceSize};",
                "export STEP_TIME=${rlConfig.model.stepSize};",
                "export STOP_TIME=${rlConfig.mdp.simulationStepsLength};",
                "export TIME_UNIT=${rlConfig.timeUnit()};",
                "export MAX_ITERATIONS=${rlConfig.mdp.epochs};",
                "export RANDOM_SEED=1;",
                "export MAX_REWARD_MEAN=${Int.MAX_VALUE};", //TODO: allow setting this via UI
                "export TEST_ITERATIONS=0;", // TODO: Disabled for now
                // Setup Environment
                "mkdir conda; cd conda; tar xf ../rllibpack.tar.gz; rm ../rllibpack.tar.gz; source bin/activate; cd ..;",
                "unzip baseEnv.zip; rm baseEnv.zip; mv baseEnv work;",
                "mv PathmindPolicy.jar work/lib/;",
                "cd work; unzip ../model.zip; rm ../model.zip;",
                "unzip ../nativerl-1.0.0-SNAPSHOT-bin.zip; rm ../nativerl-1.0.0-SNAPSHOT-bin.zip; mv nativerl-bin/* .; mv examples/train.sh .;",
                " echo > setup.sh; mkdir -p database; touch database/db.properties;",
                "source train.sh;",
                "mv policy.zip ..; cd ..; rm -rf work conda"
        )

        val command = actions.joinToString(" ");

        val jobSpec = Job(
                name = "user-${rlConfig.model.userId}-model-${rlConfig.model.id}-rllib-${rlConfig.mdp.id}",
                jobanalyses = listOf(JobAnalysis(
                        command = command,
                        inputFiles = rlConfig.environment.fileIds.map { JobAnalysis.FileReference(it, decompress=false) } +
                                JobAnalysis.FileReference(rlConfig.model.fileId, decompress = false)

                ))
        )

        val job = apiClient.jobCreate(jobSpec)
        apiClient.jobSubmit(job)

        return job.id!!
    }

    override fun stop(jobId: String) {
        apiClient.jobStop(jobId)
    }

    override fun getConsoleOutput(jobId: String): String {
        return apiClient.consoleOutput(jobId)
    }

    override fun tailConsoleOutput(jobId: String): String {
        return apiClient.tailConsole(jobId, "1")
    }


    override fun upload(file: File): String = apiClient.fileUpload(file).id

    override fun status(jobId: String): String {
        val first = apiClient.jobStatusHistory(jobId).results.first()
        return when(first.statusReason){
            "Completed successfully" -> first.status
            "User terminated" -> first.statusReason
            else -> first.status
        }
    }

    override fun getPolicy(jobId: String): InputStream {
        return apiClient.policyFile(jobId)
    }
}