package ai.skymind.skynet.spring.cloud.job.local

import ai.skymind.skynet.spring.cloud.job.api.JobExecutor
import ai.skymind.skynet.spring.cloud.job.rescale.util.runCommand
import java.io.File
import java.io.InputStream
import java.nio.file.Files

class LocalJobExecutor(val fileIdMap: Map<String, String>) : JobExecutor {
    override fun stop(jobId: String) {
        TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

    override fun getPolicy(jobId: String): InputStream {
        TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

    override fun status(externalJobId: String): String {
        TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

    override fun upload(file: File): String {
        TODO("just provide an appropriate fileIdMap when instantiating the LocalJobExecutor for now.")
    }
    /**
     * Creates Working directory, writes pre-process, run and post process scripts to it.
     *
     * Controls actual execution in separate thread
     */
    override fun run(rlConfig: RLConfig): String {
        // Phase -1: Setup a working directory (= job id in this case)
        val s = "${rlConfig.model.id}-${rlConfig.mdp.id}"
        val x = Thread {this._run(s, rlConfig)}
        x.start()
        x.join()

        return s
    }

    override fun getConsoleOutput(jobId: String): String {
        TODO()
    }

    override fun tailConsoleOutput(jobId: String): String {
        TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

    private fun _run(jobId: String, rlConfig: RLConfig) {
        // Phase 0: Things that will be "input files" on rescale
        // Step 0. Set up environment
        // Step 0a. Unzip Base Files
        // Step 0b. Setup Classpath environment variables
        // Step 1. Extract Model archive into environment
        // Step 1b.Write base code to helper file

        val wd = Files.createTempDirectory(jobId)
        wd.toFile().deleteOnExit()

        val workingDir = wd.toAbsolutePath().toString().let{
            "/" + it.replace(":\\", "/").replace('\\', '/')
        }
        val setup = StringBuilder().apply {
            append("cd $workingDir;")
            rlConfig.environment.fileIds.forEach{
                val path = fileIdMap[it]
                append("unzip $path -d $workingDir;")
            }
            append("unzip ${fileIdMap[rlConfig.model.fileId]} -d $workingDir/baseEnv/;")
        }.toString()

        // Phase 1. Prepare environment = PreProcess Script
        // Step 2. Figure out package name for model
        // Step 3. Create appropriate folder structure for package name
        // Step 4. Write `package $packagename;` to target file
        // Step 5. Append base code from helper file to target file
        // Step 6. Compile target file, capturing output
        val preProcess = StringBuilder().apply{
            append("cd $workingDir/baseEnv;")
            append("export CLASSPATH=`find . -name '*.jar'| paste -sd '${File.pathSeparator}' -`;")
            append("export MODEL_PACKAGE=`unzip -l model.jar | grep Main.class | awk '{print $4}' | cut -d/ -f1`;")
            append("mkdir \$MODEL_PACKAGE;")
            append("echo \"package \$MODEL_PACKAGE;\" > \$MODEL_PACKAGE/NewTraining.java;")
            append("cat << EOF >> \$MODEL_PACKAGE/NewTraining.java\n${rlConfig.toTrainingFile()}\nEOF\n")
            append("javac -cp \$CLASSPATH \$MODEL_PACKAGE/NewTraining.java &> compile.out.txt;")
        }.toString()

        // Phase 2. Run training = command
        // Step 7. Run training
        val run = StringBuilder().apply{
            append("java -cp \$CLASSPATH:. \$MODEL_PACKAGE.NewTraining &> training.out.txt;")
        }.toString()

        // Phase 3. Cleanup and Collect = PostProcess Script
        // Step 8. Copy all output files to output directory
        // Step 9. Delete Working Directory
        val postProcess = StringBuilder().apply{
            append("mkdir -p /x/skynet-results/$jobId;")
            append("cp $workingDir/baseEnv/${rlConfig.outputFileName} /x/skynet-results/$jobId/;")
            append("cp $workingDir/baseEnv/compile.out.txt /x/skynet-results/$jobId/;")
            append("cp $workingDir/baseEnv/training.out.txt /x/skynet-results/$jobId/;")
        }.toString()

        val fullScript = setup + preProcess + run + postProcess
        val executeSh = wd.resolve("execute.sh").toFile()
        executeSh.writeText(fullScript)

        val result = "bash $executeSh".runCommand(wd.toFile())
        println(result.stdOut)
        println(result.stdErr)

        wd.toFile().deleteRecursively()
    }
}

