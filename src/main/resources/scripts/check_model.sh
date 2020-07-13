#!/usr/bin/env bash
workDir=$1
mode=$2
libDir="/lib"

cd ${workDir}

export CLASSPATH=$(find ${libDir}/pathmind -iname '*.jar' -printf '%p:')
export POLICY=$(find ${libDir}/policy -iname "*${mode}.jar" -printf '%p:')
export CLASSPATH=$PWD:$PWD/model.jar:${CLASSPATH}:${POLICY}:/pathmind-model-analyzer.jar
export MODEL_PACKAGE=$(unzip -l model.jar | grep Main.class | awk '{print $4}' | xargs dirname)
export MODEL_PACKAGE_NAME=$(echo ${MODEL_PACKAGE} | sed 's/\//\./g')
export AGENT_CLASS="$MODEL_PACKAGE_NAME.Main"

java -cp /pathmind-model-analyzer.jar -Dloader.main=io.skymind.pathmind.analyzer.code.CodeGenerator org.springframework.boot.loader.PropertiesLauncher \
    --agent-class-name "$AGENT_CLASS" \
    --package-name "$MODEL_PACKAGE_NAME"

javac $(find -iname '*.java')

java ${MODEL_PACKAGE_NAME}.ModelAnalyzer