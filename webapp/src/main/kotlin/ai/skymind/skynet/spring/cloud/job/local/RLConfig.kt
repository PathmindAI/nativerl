package ai.skymind.skynet.spring.cloud.job.local

import ai.skymind.skynet.data.db.jooq.tables.records.MdpRecord
import ai.skymind.skynet.data.db.jooq.tables.records.ModelRecord
import ai.skymind.skynet.spring.views.CharSequenceJavaFileObject
import java.util.*
import javax.tools.DiagnosticCollector
import javax.tools.JavaFileObject
import javax.tools.ToolProvider

data class RLConfig(
        val outputFileName: String,
        val environment: Environment,
        val model: ModelRecord,
        val mdp: MdpRecord
){
    fun timeUnit() = when(model.timeUnit){
        "milliseconds" -> "MILLISECOND"
        "seconds" -> "SECOND"
        "minutes" -> "MINUTE"
        "hours" -> "HOUR"
        "days" -> "DAY"
        "weeks" -> "WEEK"
        "months" -> "MONTH"
        "years" -> "YEAR"
        else -> throw IllegalArgumentException("${model.timeUnit} is an Invalid Time Unit")
    }


    data class CompileErrors(val imports: MutableList<String> = mutableListOf(), val variables: MutableList<String> = mutableListOf(), val reset: MutableList<String> = mutableListOf(), val reward: MutableList<String> = mutableListOf(), val metrics: MutableList<String> = mutableListOf())
    data class CodeRanges(val imports: Pair<Int?, Int?> = null to null, val variables: Pair<Int?, Int?> = null to null, val reward: Pair<Int?, Int?> = null to null, val metrics: Pair<Int?, Int?> = null to null, val reset: Pair<Int?, Int?> = null to null)
    fun compileErrors(): CompileErrors {
        val code = toTrainingFile()
        val lines = code.split("\n")
        val range = lines.mapIndexed { idx, line -> idx to line }.fold(CodeRanges()) { acc, pair ->
            val (index, line) = pair
            when (line) {
                "// START CUSTOM IMPORTS 8279520310" -> acc.copy(imports = acc.imports.copy(first = index))
                "// END CUSTOM IMPORTS 8279520310" -> acc.copy(imports = acc.imports.copy(second = index))

                "// START CUSTOM VARIABLES 4998274237" -> acc.copy(variables = acc.variables.copy(first = index))
                "// END CUSTOM VARIABLES 4998274237" -> acc.copy(variables = acc.variables.copy(second = index))

                "// START CUSTOM RESET 0119724160" -> acc.copy(reset = acc.reset.copy(first = index))
                "// END CUSTOM RESET 0119724160" -> acc.copy(reset = acc.reset.copy(second = index))

                "// START CUSTOM REWARD 2934951523" -> acc.copy(reward = acc.reward.copy(first = index))
                "// END CUSTOM REWARD 2934951523" -> acc.copy(reward = acc.reward.copy(second = index))

                "// START CUSTOM METRICS 2867541152" -> acc.copy(metrics = acc.metrics.copy(first = index))
                "// END CUSTOM METRICS 2934951523" -> acc.copy(metrics = acc.metrics.copy(second = index))

                else -> acc
            }
        }

        val diagnostics = DiagnosticCollector<JavaFileObject>()
        val compiler = ToolProvider.getSystemJavaCompiler()
        compiler.getTask(null, null, diagnostics, null, null, listOf(CharSequenceJavaFileObject("NewTraining", code))).call()

        return diagnostics.diagnostics.fold(CompileErrors()){ acc, diag ->
            when(diag.lineNumber){
                in range.imports.first!!..range.imports.second!! -> acc.imports.add("${diag.kind}: Line ${diag.lineNumber - range.imports.first!! - 1}: ${diag.getMessage(Locale.ROOT)}")
                in range.variables.first!!..range.variables.second!! -> acc.variables.add("${diag.kind}: Line ${diag.lineNumber - range.variables.first!! - 1}: ${diag.getMessage(Locale.ROOT)}")
                in range.reset.first!!..range.reset.second!! -> acc.reset.add("${diag.kind}: Line ${diag.lineNumber - range.reset.first!! - 1}: ${diag.getMessage(Locale.ROOT)}")
                in range.reward.first!!..range.reward.second!! -> acc.reward.add("${diag.kind}: Line ${diag.lineNumber - range.reward.first!! - 1}: ${diag.getMessage(Locale.ROOT)}")
                in range.metrics.first!!..range.metrics.second!! -> acc.metrics.add("${diag.kind}: Line ${diag.lineNumber - range.metrics.first!! - 1}: ${diag.getMessage(Locale.ROOT)}")
            }
            acc
        }
    }

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
import org.nd4j.linalg.learning.config.AdaDelta;

// START CUSTOM IMPORTS 8279520310
${mdp.imports ?: ""}
// END CUSTOM IMPORTS 8279520310

public class NewTraining extends ExperimentCustom {
    public class AnylogicToRLAdapter {
// START CUSTOM VARIABLES 4998274237
      ${mdp.variables  ?: ""}
// END CUSTOM VARIABLES 4998274237
    
      public ObservationSpace<Encodable> getObservationSpace(){
        ObservationSpace<Encodable> space = new ArrayObservationSpace<>(new int[] {${mdp.observationSpaceSize}});
        return space;
      }
      
      public DiscreteSpace getActionSpace(){
        DiscreteSpace space = new DiscreteSpace(${mdp.actionSpaceSize});
        return space;
      }
      
      public void reset(Main agent){
// START CUSTOM RESET 0119724160
        ${mdp.reset  ?: ""}
// END CUSTOM RESET 0119724160
      }
      
      
      public double reward(Main agent, double[] before, double[] after) {
        double reward = 0;
// START CUSTOM REWARD 2934951523
        ${mdp.reward  ?: ""}
// END CUSTOM REWARD 2934951523
        return reward;
      }
      
      public double[] metrics(Main agent){
        double[] metrics = null;
// START CUSTOM METRICS 2867541152
        ${mdp.metrics  ?: ""}
// END CUSTOM METRICS 2934951523
        return metrics;
      }
    
    }

    public NewTraining() {
        super(null);
    }

    public MDP getMDP(){
     return new MDP<Encodable, Integer, DiscreteSpace>() {
            Engine engine;
            Main root;
            
            AnylogicToRLAdapter adapter = new AnylogicToRLAdapter();
        
            public ObservationSpace<Encodable> getObservationSpace() {
                return adapter.getObservationSpace();
            }
        
            public DiscreteSpace getActionSpace() {
                return adapter.getActionSpace();
            }
        
            public Encodable getObservation() {
                return new Encodable() {
                    double[] a = root.getObservation();
        
                    public double[] toArray() {
                        return a;
                    }
                };
            }
        
            public Encodable reset() {      
                if (engine != null) {
                    engine.stop();
                }
                // Create Engine, initialize random number generator:
                engine = createEngine();
                // Fixed seed (reproducible simulation runs)
                engine.getDefaultRandomGenerator().setSeed(1);
                // Selection mode for simultaneous events:
                engine.setSimultaneousEventsSelectionMode(Engine.EVENT_SELECTION_LIFO);
                // Create new root object:
                root = new Main(engine, null, null);
                // TODO Setup parameters of root object here
                root.setParametersToDefaultValues();
        
                adapter.reset(root);
        
                // Prepare Engine for simulation:
                engine.start(root);
        
        
                return getObservation();
            }
        
            public void close() {
                // Destroy the model:
                engine.stop();
            }
        
            public StepReply<Encodable> step(Integer action) {
                double[] s0 = root.getObservation();
                
                root.doAction(action);
                engine.runFast(root.time() + ${timeUnit()}.convertTo(${model.stepSize}, com.anylogic.engine.TimeUnits.SECOND));
                double[] s1 = root.getObservation();
               
                double reward = adapter.reward(root, s0, s1);
                
                return new StepReply(getObservation(), reward, isDone(), null);
            }
        
            public boolean isDone() {
                return false;
            }
        
            public MDP<Encodable, Integer, DiscreteSpace> newInstance() {
                return null; // not required for DQN
            }
        };
    }

    public void run() {
        MDP mdp = getMDP();

        try {
            DataManager manager = new DataManager(true);
            QLConfiguration AL_QL = new QLConfiguration(1, ${mdp.simulationStepsLength}, ${mdp.simulationStepsLength * mdp.epochs}, ${mdp.experienceReplaySteps}, ${mdp.batchSize}, ${mdp.stepsPerUpdate}, ${mdp.warmupSteps}, 0.1D, 0.99D, 1.0D, 0.1F, 1000, true);
            Configuration AL_NET = Configuration.builder().l2(0.0D).updater(new AdaDelta()).numHiddenNodes(300).numLayer(2).build();
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
        engine.setTimeUnit( ${timeUnit()} );
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