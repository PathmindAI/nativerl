#!/usr/bin/env bash
workDir=$1
libDir="/lib/pathmind"

cd $workDir

export CLASSPATH=$(find $libDir -iname '*.jar' -printf '%p:')
export CLASSPATH=/bin/hyperparameters_extractor.jar:$PWD:$PWD/model.jar:$CLASSPATH
export MODEL_PACKAGE=$(unzip -l model.jar | grep Main.class | awk '{print $4}' | xargs dirname)
export PROJECT=$(echo $MODEL_PACKAGE | sed 's/\//\./g')

java io.skymind.Helper $PROJECT