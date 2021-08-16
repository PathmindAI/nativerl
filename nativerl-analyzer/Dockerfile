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

RUN bash /bin/download_lib.sh ${S3BUCKET} rllibpack.tar.gz rllibpack.tar.gz

WORKDIR /lib/pathmind

RUN bash /bin/download_lib.sh ${S3BUCKET} PathmindPolicy.jar PathmindPolicy-1.7.1-SNAPSHOT.jar

RUN bash /bin/download_lib.sh ${S3BUCKET} nativerl-bin.zip nativerl-1.7.1-SNAPSHOT-bin.zip

RUN bash /bin/download_lib.sh ${S3BUCKET} baseEnv.zip baseEnv-8.7.6.zip

WORKDIR /

#Build pathmind-model-analyzer.jar
COPY . .
RUN mvn clean install \
  && cp target/pathmind-model-analyzer.jar ./

ARG CHECK_MODEL_SCRIPT=src/main/resources/scripts/check_model.sh
COPY ${CHECK_MODEL_SCRIPT} bin

EXPOSE 8080
ENTRYPOINT ["java","-jar","/pathmind-model-analyzer.jar"]

