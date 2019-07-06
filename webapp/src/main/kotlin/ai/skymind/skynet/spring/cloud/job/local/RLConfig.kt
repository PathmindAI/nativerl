package ai.skymind.skynet.spring.cloud.job.local

import ai.skymind.skynet.data.db.jooq.tables.records.MdpRecord
import ai.skymind.skynet.data.db.jooq.tables.records.ModelRecord

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
     return TODO
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