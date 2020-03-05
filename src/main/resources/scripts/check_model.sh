#!/usr/bin/env bash
workDir=$1
mode=$2
libDir="/lib"

cd ${workDir}

export CLASSPATH=$(find ${libDir}/pathmind -iname '*.jar' -printf '%p:')
export POLICY=$(find ${libDir}/policy -iname "*${mode}.jar" -printf '%p:')
export CLASSPATH=/bin/${mode}_extractor.jar:$PWD:$PWD/model.jar:${CLASSPATH}:${POLICY}
export MODEL_PACKAGE=$(unzip -l model.jar | grep Main.class | awk '{print $4}' | xargs dirname)
export PROJECT=$(echo ${MODEL_PACKAGE} | sed 's/\//\./g')

java io.skymind.Helper ${PROJECT}