#!/bin/bash
mkdir -p target/classes/
unzip -o PathmindHelper.jar -d target/classes/
unzip -o ../nativerl-policy/target/nativerl-policy-*-SNAPSHOT.jar -d target/classes/
cp -a Assets/pathmind-single-original.png Assets/pathmind-single-??x??.png target/classes/
sed -i '/<ClassPathEntry>/,/<\/ClassPathEntry>/d' target/classes/library.xml
cd target/classes/ && zip -r ../PathmindHelper.jar .
