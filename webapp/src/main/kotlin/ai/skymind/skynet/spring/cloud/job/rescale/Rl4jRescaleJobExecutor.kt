package ai.skymind.skynet.spring.cloud.job.rescale

import ai.skymind.skynet.spring.cloud.job.api.JobExecutor
import ai.skymind.skynet.spring.cloud.job.local.RLConfig
import ai.skymind.skynet.spring.cloud.job.rescale.rest.RescaleRestApiClient
import ai.skymind.skynet.spring.cloud.job.rescale.rest.entities.Job
import ai.skymind.skynet.spring.cloud.job.rescale.rest.entities.JobAnalysis
import java.io.File
import java.io.InputStream

//@Service
class Rl4jRescaleJobExecutor(val apiClient: RescaleRestApiClient): JobExecutor {
    override fun run(rlConfig: RLConfig): String {
        val preProcess = StringBuilder().apply{
            append("unzip model.zip -d baseEnv; ")
            append("cd baseEnv;")
            append("export CLASSPATH=`find . -name '*.jar'| paste -sd ':' -`;")
            append("echo CLASSPATH=\$CLASSPATH; ")
            append("export MODEL_PACKAGE=`unzip -l model.jar | grep Main.class | awk '{print $4}' | cut -d/ -f1`;")
            append("echo MODEL_PACKAGE=\$MODEL_PACKAGE; ")
            append("mkdir \$MODEL_PACKAGE;")
            append("echo \"package \$MODEL_PACKAGE;\" > \$MODEL_PACKAGE/NewTraining.java;")
            append("cat << EOF >> \$MODEL_PACKAGE/NewTraining.java\n${rlConfig.toTrainingFile()}\nEOF\n")
            //append("javac -cp \$CLASSPATH \$MODEL_PACKAGE/NewTraining.java &> compile.out.txt;")
            append("javac -cp \$CLASSPATH \$MODEL_PACKAGE/NewTraining.java;")
        }.toString()

        val run = "java -cp \$CLASSPATH:. \$MODEL_PACKAGE.NewTraining; "

        val postProcess = StringBuilder().apply{
            append("cd ..; ")
            append("mkdir output;")
            append("cp baseEnv/${rlConfig.outputFileName} output/PhasePolicy.zip;")
            append("cp baseEnv/compile.out.txt output/;")
            append("cp baseEnv/training.out.txt output/;")
            append("rm -rf baseEnv; ")
        }.toString()

        val jobSpec = Job(
                name = "user-${rlConfig.model.userId}-model-${rlConfig.model.id}-mdp-${rlConfig.mdp.id}",
                jobanalyses = listOf(JobAnalysis(
                        command = preProcess + run + postProcess,
                        inputFiles = rlConfig.environment.fileIds.map { JobAnalysis.FileReference(it) } +
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