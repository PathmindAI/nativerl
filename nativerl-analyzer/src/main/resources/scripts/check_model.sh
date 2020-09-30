#!/usr/bin/env bash
workDir=$1
libDir="/lib"

cd ${workDir}

export CLASSPATH=$(find ${libDir}/pathmind -iname '*.jar' -printf '%p:')
export CLASSPATH=$(find $PWD/lib -iname '*.jar' -printf '%p:'):${CLASSPATH}
export CLASSPATH=$PWD:$PWD/model.jar:${CLASSPATH}:${POLICY}:/pathmind-model-analyzer.jar
export MODEL_PACKAGE=$(for m in $(ls model.jar lib/model*.jar 2> /dev/null) ; do unzip -l $m | grep Main.class; done | awk '{print $4}' | xargs dirname)
export MODEL_PACKAGE_NAME=$(echo ${MODEL_PACKAGE} | sed 's/\//\./g')
export AGENT_CLASS="$MODEL_PACKAGE_NAME.Main"
export SIMULATION_PACKAGE=$(for m in $(ls model.jar lib/model*.jar 2> /dev/null) ; do unzip -l $m | grep Simulation.class; done | awk '{print $4}' | xargs dirname)
export SIMULATION_PACKAGE_NAME=$(echo $SIMULATION_PACKAGE | sed 's/\//\./g')
export SIMULATION_CLASS="$SIMULATION_PACKAGE_NAME.Simulation"

java -cp /pathmind-model-analyzer.jar -Dloader.main=io.skymind.pathmind.analyzer.code.CodeGenerator org.springframework.boot.loader.PropertiesLauncher \
    --agent-class-name "$AGENT_CLASS" \
    --simulation-class-name "$SIMULATION_CLASS" \
    --package-name "$MODEL_PACKAGE_NAME"

javac $(find -iname '*.java')

java ${MODEL_PACKAGE_NAME}.ModelAnalyzer