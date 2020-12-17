#!/bin/bash
mkdir -p target/classes/
unzip -o PathmindHelper.jar -d target/classes/
cp -a Assets/pathmind-single-original.png Assets/pathmind-single-??x??.png target/classes/
cp -a ../nativerl-policy/target/nativerl-policy-*-SNAPSHOT.jar target/PathmindPolicy.jar
cd target/classes/
sed -i '/<ClassPathEntry>/,/<\/ClassPathEntry>/d' library.xml
sed -i '/^\s*$/d' META-INF/MANIFEST.MF
echo "Class-Path: PathmindPolicy.jar" >> META-INF/MANIFEST.MF
zip -r ../PathmindHelper.jar .
