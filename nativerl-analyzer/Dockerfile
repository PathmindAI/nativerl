# syntax = docker/dockerfile:1.0-experimental

FROM azul/zulu-openjdk:11.0.3

#Define ENV
ARG S3BUCKET
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG NATIVERL_FOLDER

# Install all required tools and dependencies
RUN apt-get update && apt-get install -y \
    unzip \
    curl \
    maven \
    python-pip \
  && pip install awscli \
  && rm -rf /var/lib/apt/lists/*

ARG DOWNLOAD_LIB_SCRIPT=scripts/download_lib.sh
COPY ${DOWNLOAD_LIB_SCRIPT} /bin/

WORKDIR /lib/pathmind/conda

RUN bash /bin/download_lib.sh ${S3BUCKET} "conda/1_3_0" rllibpack.tar.gz rllibpack.tar.gz

WORKDIR /lib/pathmind

RUN bash /bin/download_lib.sh ${S3BUCKET} "pathmindhelper/1_7_0" PathmindPolicy.jar PathmindPolicy-1.7.0-SNAPSHOT.jar \
  && bash /bin/download_lib.sh ${S3BUCKET} ${NATIVERL_FOLDER} nativerl-1.8.0-SNAPSHOT-bin.zip nativerl-bin.zip \
  && bash /bin/download_lib.sh ${S3BUCKET} "anylogic/8_7_7" baseEnv.zip baseEnv-8.7.7.zip

WORKDIR /

#Build pathmind-model-analyzer.jar
COPY . .

RUN mvn clean install \
  && cp common/target/ma-common.jar ./ \
  && cp api/target/pathmind-ma-api-spring-boot.jar ./ \
  && cp generator/target/pathmind-ma-code-generator-jar-with-dependencies.jar ./

ARG CHECK_MODEL_SCRIPT=scripts/check_model.sh
COPY ${CHECK_MODEL_SCRIPT} bin

EXPOSE 8080
ENTRYPOINT ["java","-jar","/pathmind-ma-api-spring-boot.jar"]

