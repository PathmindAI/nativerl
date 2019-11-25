FROM azul/zulu-openjdk:11.0.3
VOLUME /tmp

# Install all required dependencies into lib/pathmind directory
ARG THIRD_PARTY_DEPENDENCIES=pathmind-lib
COPY ${THIRD_PARTY_DEPENDENCIES} lib/pathmind

# Install needed tools
RUN apt-get update && apt-get install -y \
    unzip

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