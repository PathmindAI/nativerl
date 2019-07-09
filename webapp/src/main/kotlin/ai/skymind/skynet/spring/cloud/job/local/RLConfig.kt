package ai.skymind.skynet.spring.cloud.job.local

import ai.skymind.skynet.data.db.jooq.tables.records.MdpRecord
import ai.skymind.skynet.data.db.jooq.tables.records.ModelRecord

data class RLConfig(
        val outputFileName: String,
        val environment: Environment,
        val model: ModelRecord,
        val mdp: MdpRecord
){
    fun timeUnit() = "com.anylogic.engine.TimeUnits." + when(model.timeUnit){
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

${mdp.imports ?: ""}

public class NewTraining extends ExperimentCustom {
    public class AnylogicToRLAdapter {
      ${mdp.variables  ?: ""}
    
      public ObservationSpace<Encodable> getObservationSpace(){
        ObservationSpace<Encodable> space = new ArrayObservationSpace<>(new int[] {${mdp.observationSpaceSize}});
        return space;
      }
      
      public DiscreteSpace getActionSpace(){
        DiscreteSpace space = new DiscreteSpace(${mdp.actionSpaceSize});
        return space;
      }
      
      public void reset(Main agent){
        ${mdp.reset  ?: ""}
      }
      
      
      public double reward(Main agent, double[] before, double[] after) {
        double reward = 0;
        ${mdp.reward  ?: ""}
        return reward;
      }
      
      public double[] metrics(Main agent){
        double[] metrics = null;
        ${mdp.metrics  ?: ""}
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
                engine.runFast(root.time() + ${model.stepSize});
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