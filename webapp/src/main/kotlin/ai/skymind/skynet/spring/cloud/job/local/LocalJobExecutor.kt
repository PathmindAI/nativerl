package ai.skymind.skynet.spring.cloud.job.local

import ai.skymind.skynet.data.db.jooq.tables.records.MdpRecord
import ai.skymind.skynet.data.db.jooq.tables.records.ModelRecord
import ai.skymind.skynet.spring.cloud.job.api.JobExecutor
import ai.skymind.skynet.spring.cloud.job.rescale.util.runCommand
import java.io.File
import java.nio.file.Files

class LocalJobExecutor(val fileIdMap: Map<String, String>) : JobExecutor {
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

    override fun getConsoleOutput(jobId: String) {
        TODO()
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

data class Environment(
        val fileIds: List<String>
)

data class RLConfig(
        val outputFileName: String,
        val environment: Environment,
        val model: ModelRecord,
        val mdp: MdpRecord
){
    fun toTrainingFile() = """

import com.anylogic.engine.Agent;
import com.anylogic.engine.AgentConstants;
import com.anylogic.engine.AgentList;
import com.anylogic.engine.AnyLogicInternalCodegenAPI;
import com.anylogic.engine.Engine;
import com.anylogic.engine.ExperimentCustom;
import com.anylogic.engine.Utilities;
import java.io.File;
import java.io.IOException;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning.QLConfiguration;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense.Configuration;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.space.ArrayObservationSpace;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.deeplearning4j.rl4j.util.DataManager;
import org.json.JSONObject;
import org.nd4j.linalg.learning.config.RmsProp;

public class NewTraining extends ExperimentCustom {
    public NewTraining() {
        super(null);
    }

    public MDP getMDP(){
     return ${mdp.code};
    }

    public void run() {
        MDP mdp = getMDP();

        try {
            DataManager manager = new DataManager(true);
            QLConfiguration AL_QL = new QLConfiguration(1, 2880, 288000, 288000, 128, 500, 10, 0.1D, 0.99D, 1.0D, 0.1F, 1000, true);
            Configuration AL_NET = Configuration.builder().l2(0.0D).updater(new RmsProp(0.001D)).numHiddenNodes(300).numLayer(2).build();
            QLearningDiscreteDense<Encodable> dql = new QLearningDiscreteDense(mdp, AL_NET, AL_QL, manager);
            dql.train();
            DQNPolicy<Encodable> pol = dql.getPolicy();
            pol.save("$outputFileName");
            mdp.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public void setupEngine_xjal(Engine engine) {
        engine.setVMethods(427313);
        engine.setTimeUnit(AgentConstants.SECOND);
    }

    public static void main(String[] args) {
        NewTraining ex = new NewTraining();
        ex.run();
    }
}
    """.trimIndent()
}