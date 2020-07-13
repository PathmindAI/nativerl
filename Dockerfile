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
 
WORKDIR /lib/policy

RUN aws s3 cp s3://${S3BUCKET}/PathmindPolicy_single.jar ./
RUN aws s3 cp s3://${S3BUCKET}/PathmindPolicy_multi.jar ./

WORKDIR /lib/pathmind

RUN aws s3 cp s3://${S3BUCKET}/nativerl-1.0.0-SNAPSHOT-bin.zip ./ \
 && unzip nativerl-1.0.0-SNAPSHOT-bin.zip \
 && rm nativerl-1.0.0-SNAPSHOT-bin.zip

RUN aws s3 cp s3://${S3BUCKET}/baseEnv.zip ./ \
 && unzip baseEnv.zip \
 && rm baseEnv.zip

RUN curl -s https://www.benf.org/other/cfr/cfr-0.148.jar -o cfr-0.148.jar
 
WORKDIR /
 
#Build pathmind-model-analyzer.jar
COPY . .
RUN mvn clean package \
  && cp target/pathmind-model-analyzer.jar ./

ARG CHECK_MODEL_SCRIPT=src/main/resources/scripts/check_model.sh
COPY ${CHECK_MODEL_SCRIPT} bin

ARG SINGLE_OR_MULTI_SCRIPT=src/main/resources/scripts/check_single_or_multi.sh
COPY ${SINGLE_OR_MULTI_SCRIPT} bin

EXPOSE 8080
ENTRYPOINT ["java","-jar","/pathmind-model-analyzer.jar"]
