# syntax = docker/dockerfile:1.0-experimental

FROM azul/zulu-openjdk:11.0.3

# Install all required tools and dependencies
RUN apt-get update && apt-get install -y \
    unzip \
    curl \
 && rm -rf /var/lib/apt/lists/*
 
WORKDIR /lib/pathmind

RUN --mount=type=secret,id=rescale_token,required,dst=/tmp/rescale_api_token.txt curl -s -H @/tmp/rescale_api_token.txt https://platform.rescale.jp/api/v2/files/kuQJAd/contents/ -o PathmindPolicy.jar

RUN --mount=type=secret,id=rescale_token,required,dst=/tmp/rescale_api_token.txt curl -s -H @/tmp/rescale_api_token.txt https://platform.rescale.jp/api/v2/files/jKjXa/contents/ -o nativerl-1.0.0-SNAPSHOT-bin.zip \
 && unzip nativerl-1.0.0-SNAPSHOT-bin.zip \
 && rm nativerl-1.0.0-SNAPSHOT-bin.zip

RUN --mount=type=secret,id=rescale_token,required,dst=/tmp/rescale_api_token.txt curl -s -H @/tmp/rescale_api_token.txt https://platform.rescale.jp/api/v2/files/FcrKm/contents/ -o baseEnv.zip \
 && unzip baseEnv.zip \
 && rm baseEnv.zip

RUN curl -s https://www.benf.org/other/cfr/cfr-0.148.jar -o cfr-0.148.jar
 
WORKDIR /
 
# Copy a script which executes extraction JAR
ARG SCRIPT=src/main/resources/scripts/check_model.sh
COPY ${SCRIPT} bin

# Copy JAR used to extract hyperparameters from a model
ARG EXTRACTOR_JAR=src/main/resources/hyperparameters_extractor.jar
COPY ${EXTRACTOR_JAR} bin

# Copy builded model-analyzer service
ARG JAR_FILE=target/pathmind-model-analyzer.jar
COPY ${JAR_FILE} pathmind-model-analyzer.jar

EXPOSE 8080
ENTRYPOINT ["java","-jar","/pathmind-model-analyzer.jar"]