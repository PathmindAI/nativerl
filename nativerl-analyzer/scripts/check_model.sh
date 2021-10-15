#!/usr/bin/env bash
workDir=$1
mainAgent=$2
experimentClass=$3
EXPERIMENT_TYPE=$4
PATHMIND_HELPER_CLASS=$5
libDir="/lib"

if [[ -z "$mainAgent" ]]; then
    echo "main agent missing"
    mainAgent="Main"
fi
if [[ -z "$experimentClass" ]]; then
    experimentClass="Simulation"
fi
if [[ -z "$EXPERIMENT_TYPE" ]]; then
    EXPERIMENT_TYPE="Simulation"
fi
if [[ -z "$PATHMIND_HELPER_CLASS" ]]; then
    PATHMIND_HELPER_CLASS="pathmindHelper"
fi

cd ${workDir}

export CLASSPATH=$(find ${libDir}/pathmind -iname '*.jar' -print0 | sort -z | xargs --null -i printf "{}:")
export CLASSPATH=$(find $PWD/lib -iname '*.jar' -print0 | sort -z | xargs --null -i printf "{}:"):${CLASSPATH}
export CLASSPATH=$PWD:$PWD/model.jar:${CLASSPATH}
export CLASSPATH=/ma-common.jar:${CLASSPATH}

export MODEL_PACKAGE=$(for m in $(ls model.jar lib/model*.jar 2> /dev/null) ; do unzip -l $m | grep /${mainAgent}.class; done | awk '{print $4}' | xargs dirname)
export MODEL_PACKAGE_NAME=$(echo ${MODEL_PACKAGE} | sed 's/\//\./g')
export AGENT_CLASS="$MODEL_PACKAGE_NAME.${mainAgent}"
export SIMULATION_PACKAGE=$(for m in $(ls model.jar lib/model*.jar 2> /dev/null) ; do unzip -l $m | grep /${experimentClass}.class | grep -v pathmind/policyhelper; done | awk '{print $4}' | xargs dirname)
export SIMULATION_PACKAGE_NAME=$(echo $SIMULATION_PACKAGE | sed 's/\//\./g')
export SIMULATION_CLASS="$SIMULATION_PACKAGE_NAME.${experimentClass}"

java -jar /pathmind-ma-code-generator-jar-with-dependencies.jar \
    --agent-class-name "$AGENT_CLASS" \
    --simulation-class-name "$SIMULATION_CLASS" \
    --package-name "$MODEL_PACKAGE_NAME" \
    --pathmind-helper-class-name "$PATHMIND_HELPER_CLASS" \
    --experiment-type "$EXPERIMENT_TYPE"

javac $(find -iname '*.java')

java ${MODEL_PACKAGE_NAME}.ModelAnalyzer