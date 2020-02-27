#!/bin/bash
cd $1

export MAIN_CLASS=$(jar tf model.jar | grep Main.class | sed 's/\.class//; s/\//\./')
export RESULT=$(javap -cp model.jar -private ${MAIN_CLASS} | grep "_pathmindHelper_observationForReward_xjal" | awk '{print $2}' | awk -F"[][[]]" '{print NF-1}')
echo $RESULT