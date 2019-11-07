#!/usr/bin/env bash
workDir=$1
libDir="/lib/pathmind"
cd $workDir

export MODEL_PACKAGE=$(unzip -l model.jar | grep Main.class | awk '{print $4}' | xargs dirname)
export MODEL_PACKAGE_NAME=$(echo $MODEL_PACKAGE | sed 's/\//\./g')
export ENVIRONMENT_CLASS="$MODEL_PACKAGE_NAME.PathmindEnvironment"
export AGENT_CLASS="$MODEL_PACKAGE_NAME.Main"
export OUTPUT_DIR=$(pwd)

mkdir -p $MODEL_PACKAGE

cat <<EOF > $MODEL_PACKAGE/Training.java
package $MODEL_PACKAGE_NAME;
import com.anylogic.engine.AgentConstants;
import com.anylogic.engine.AnyLogicInternalCodegenAPI;
import com.anylogic.engine.Engine;
import com.anylogic.engine.ExperimentCustom;
import com.anylogic.engine.Utilities;

public class Training extends ExperimentCustom {
    @AnyLogicInternalCodegenAPI
    public static String[] COMMAND_LINE_ARGUMENTS_xjal = new String[0];

    public Training(Object parentExperiment) {
        super(parentExperiment);
        this.setCommandLineArguments_xjal(COMMAND_LINE_ARGUMENTS_xjal);
    }

    public void run() {
    }

    @AnyLogicInternalCodegenAPI
    public void setupEngine_xjal(Engine engine) {
        Simulation sim = new Simulation();
        sim.setupEngine(engine);
        sim.initDefaultRandomNumberGenerator(engine);
    }

    @AnyLogicInternalCodegenAPI
    public static void main(String[] args) {
        COMMAND_LINE_ARGUMENTS_xjal = args;
        Utilities.prepareBeforeExperimentStart_xjal(Training.class);
        Training ex = new Training((Object)null);
        ex.setCommandLineArguments_xjal(args);
        ex.run();
    }
}
EOF

export CLASSPATH=$(find $libDir -iname '*.jar' -printf '%p:')
export CLASSPATH=$PWD:$PWD/model.jar:$CLASSPATH

java ai.skymind.nativerl.AnyLogicHelper \
    --environment-class-name "$ENVIRONMENT_CLASS" \
    --agent-class-name "$AGENT_CLASS" \
    --policy-helper RLlibPolicyHelper


cat <<EOF > $MODEL_PACKAGE/VerifySettings.java
package $MODEL_PACKAGE_NAME;

public class VerifySettings extends PathmindEnvironment {
    public static void main(String[] args) {
        PathmindEnvironment e = new PathmindEnvironment(null);
        e.reset();
        System.out.println(e.agent.pathmindHelper.possibleActionCount);
        System.out.println(e.agent.pathmindHelper.observationForTraining().length);
        System.exit(0);
    }
}
EOF

javac $(find -iname '*.java')

java $MODEL_PACKAGE/VerifySettings