package ai.skymind.skynet.spring.cloud.job.rescale

import ai.skymind.skynet.data.db.jooq.tables.records.MdpRecord
import ai.skymind.skynet.data.db.jooq.tables.records.ModelRecord
import ai.skymind.skynet.spring.cloud.job.local.Environment
import ai.skymind.skynet.spring.cloud.job.local.RLConfig
import ai.skymind.skynet.spring.cloud.job.rescale.rest.RescaleRestApiClient
import ai.skymind.skynet.spring.cloud.job.rescale.rest.entities.Job
import ai.skymind.skynet.spring.cloud.job.rescale.rest.entities.JobAnalysis
import com.fasterxml.jackson.databind.ObjectMapper
import org.junit.Assert.*
import org.junit.Ignore
import org.junit.Test
import org.springframework.web.reactive.function.client.WebClient
import java.io.File

@Ignore
class RescaleJobExecutorTest {
    val apiClient = RescaleRestApiClient(
            //"eu.rescale.com",
            //"e6300684e6355b3bc34e95cfa368a1164e12edc7",
            "platform.rescale.jp",
            "0d0601925a547db44d41007e3cc4386b075c761c",
            ObjectMapper().findAndRegisterModules(),
            WebClient.builder()
    )

    @Ignore
    @Test
    fun mpiInformationJob(){
        val jobSpec = Job(
                name = "mpi-files",
                jobanalyses = listOf(JobAnalysis(
                        command = "for f in hosts mpd.hosts machinefile machinefile.gpu machinefile.openmpi hosts " +
                                "rhosts mpd.hosts.string PCF.xml; do echo \$f; cat \$HOME/\$f; echo; done ",
                        inputFiles = emptyList(),
                        hardware = JobAnalysis.Hardware("emerald", 8)

                ))
        )

        val job = apiClient.jobCreate(jobSpec)
        apiClient.jobSubmit(job)

        println(job.id)
    }

    @Ignore
    @Test
    fun collectMpiJobOutput(){
        val jobId = "dBPqW"

        println(apiClient.consoleOutput(jobId))
    }

    @Ignore
    @Test
    fun run() {
        val mdpCode = """
new MDP<Encodable, Integer, DiscreteSpace>() {
    Engine engine;
    Main root;
    int simCount = 0;
    String combinations[][] = {
            {"constant_moderate", "constant_moderate"},
            {"none_til_heavy_afternoon_peak", "constant_moderate"},
            {"constant_moderate", "none_til_heavy_afternoon_peak"},
            {"peak_afternoon", "peak_morning"},
            {"peak_morning", "peak_afternoon"}
    };

    ObservationSpace<Encodable> observationSpace = new ArrayObservationSpace<>(new int[]{10});
    DiscreteSpace actionSpace = new DiscreteSpace(2);

    public ObservationSpace<Encodable> getObservationSpace() {
        return observationSpace;
    }

    public DiscreteSpace getActionSpace() {
        return actionSpace;
    }

    public Encodable getObservation() {
        return new Encodable() {
            double[] a = root.getState();

            public double[] toArray() {
                return a;
            }
        };
    }

    public Encodable reset() {
        simCount++;
        if (engine != null) {
            engine.stop();
        }
        // Create Engine, initialize random number generator:
        engine = createEngine();
        // Fixed seed (reproducible simulation runs)
        engine.getDefaultRandomGenerator().setSeed(1);
        // Selection mode for simultaneous events:
        engine.setSimultaneousEventsSelectionMode(Engine.EVENT_SELECTION_LIFO);
        // Set stop time:
        engine.setStopTime(28800);
        // Create new root object:
        root = new Main(engine, null, null);
        root.policy = null;
        // TODO Setup parameters of root object here
        root.setParametersToDefaultValues();
        root.usePolicy = false;
        root.manual = false;
        root.schedNameNS = combinations[simCount % combinations.length][0];
        root.schedNameEW = combinations[simCount % combinations.length][1];

        // Prepare Engine for simulation:
        engine.start(root);


        return getObservation();
    }

    public void close() {
        // Destroy the model:
        engine.stop();
    }

    public StepReply<Encodable> step(Integer action) {
        double[] s0 = root.getState();
        int p0 = root.trafficLight.getCurrentPhaseIndex();
        // {modified step function to ignore action if in yellow phase}
        root.step(action);
        engine.runFast(root.time() + 10);
        double[] s1 = root.getState();
        int p1 = root.trafficLight.getCurrentPhaseIndex();

        double sum0 = s0[0] + s0[2] + s0[4] + s0[6];
        double sum1 = s1[0] + s1[2] + s1[4] + s1[6];
        double reward = sum0 < sum1 ? -1 : sum0 > sum1 ? 1 : 0;

        return new StepReply(getObservation(), reward, isDone(), null);
    }

    public boolean isDone() {
        return false;
    }

    public MDP<Encodable, Integer, DiscreteSpace> newInstance() {
        return null; // not required for DQN
    }
};
    """.trimIndent()

        val env = Environment(listOf("VNNaQb", "XeGNac")) // jp file ids
        //val env = Environment(listOf("vbqfEc", "KDiSPc")) // eu file ids
        val model = ModelRecord().apply {
            id = 7
            fileId = "JCNaQb" // jp file id
            //fileId = "aDWBGc" // eu file id
        }
        val mdp = MdpRecord().apply {
            id = 8
            //code = mdpCode
        }
        val rlConfig = RLConfig("PhasePolicy.zip", env, model, mdp)

        val executor = Rl4jRescaleJobExecutor(apiClient)
        println(executor.run(rlConfig))


    }

    @Ignore
    @Test
    fun rllibTest(){
        val env = Environment(listOf("qBaAAd", "rbOcJd", "LZAENb", "XeGNac")) // jp file ids
        val model = ModelRecord().apply {
            id = 7
            fileId = "vZdrMd"
            stepSize = 1
            timeUnit = "minutes"
        }
        val mdp = MdpRecord().apply {
            id = 8
            reward = "reward = 7;"
            epochs = 1
            actionSpaceSize = 4
            observationSpaceSize = 7
            simulationStepsLength = 720
        }
        val rlConfig = RLConfig("policy.zip", env, model, mdp)

        val executor = RllibRescaleJobExecutor(apiClient)
        println(executor.run(rlConfig))

    }

    @Ignore
    @Test
    fun status4cores() {
        val jobId = "afHxeb"
        //val jobId = "ZCRBeb"
        println(apiClient.jobStatusHistory(jobId))
        println(apiClient.jobRuns(jobId))
        val page = apiClient.directoryContent(jobId, "1")
        println(page)
        println(apiClient.tailConsole(jobId, "1"))
        println(apiClient.consoleOutput(jobId))
        println(apiClient.compileOutput(jobId))
        println(apiClient.outputFiles(jobId))
    }

    @Ignore
    @Test
    fun stop() {

        apiClient.jobStop("GWQop")
    }

    @Ignore
    @Test
    fun uploadFile(){
        val input = File("X:/hello-world.123.txt")
        val uploaded = apiClient.fileUpload(input)
        val file = apiClient.filesList().results.find { it.name == "hello-world.123.txt" && it.id == uploaded.id }

        assertNotNull(file)
        assertEquals(uploaded.id, file!!.id)
        val content = input.readBytes()
        val uploadedContent = apiClient.fileContents(uploaded.id)

        assertArrayEquals(content, uploadedContent)
        apiClient.deleteFile(uploaded.id)
        Thread.sleep(15000)
        assertNull(apiClient.filesList().results.find { it.id == uploaded.id})
    }

    @Ignore
    @Test
    fun listFiles(){
        println(apiClient.filesList())
    }

    @Ignore
    @Test
    fun deleteFile(){
        apiClient.filesList().results.filter { it.name == "hello-world.123.txt"}.forEach{apiClient.deleteFile(it.id)}
    }

    @Ignore
    @Test
    fun deleteAllFiles(){
        apiClient.filesList().results.filter{!it.isDeleted}.forEach{apiClient.deleteFile(it.id)}
    }
}