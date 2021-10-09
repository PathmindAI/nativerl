# syntax = docker/dockerfile:1.0-experimental

FROM azul/zulu-openjdk:11.0.3

#Define ENV
ARG S3BUCKET
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

# Install all required tools and dependencies
RUN apt-get update && apt-get install -y \
    unzip \
    curl \
    maven \
    python-pip \
 && pip install awscli \
 && rm -rf /var/lib/apt/lists/*

ARG DOWNLOAD_LIB_SCRIPT=src/main/resources/scripts/download_lib.sh
COPY ${DOWNLOAD_LIB_SCRIPT} /bin/

WORKDIR /lib/pathmind/conda

RUN bash /bin/download_lib.sh ${S3BUCKET} "conda/1_3_0" rllibpack.tar.gz rllibpack.tar.gz

WORKDIR /lib/pathmind

RUN bash /bin/download_lib.sh ${S3BUCKET} "pathmindhelper/1_7_0" PathmindPolicy.jar PathmindPolicy-1.7.0-SNAPSHOT.jar

RUN bash /bin/download_lib.sh ${S3BUCKET} "nativerl/1_7_2" nativerl-1.7.2-SNAPSHOT-bin.zip nativerl/1_7_2/nativerl-bin.zip

RUN bash /bin/download_lib.sh ${S3BUCKET} "anylogic/8_7_7" baseEnv.zip baseEnv-8.7.7.zip

WORKDIR /

#Build pathmind-model-analyzer.jar
COPY . .
RUN mvn clean install \
  && cp target/pathmind-model-analyzer.jar ./

RUN unzip target/pathmind-model-analyzer.jar -d /lib/ma

ARG CHECK_MODEL_SCRIPT=src/main/resources/scripts/check_model.sh
COPY ${CHECK_MODEL_SCRIPT} bin

EXPOSE 8080
ENTRYPOINT ["java","-jar","/pathmind-model-analyzer.jar"]

