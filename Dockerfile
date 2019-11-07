FROM azul/zulu-openjdk:11.0.3
VOLUME /tmp
ARG JAR_FILE=target/pathmind-model-analyzer.jar
ARG THIRD_PARTY_DEPENDENCIES=pathmind-lib
ARG SCRIPT=src/main/resources/scripts/check_model.sh
COPY ${THIRD_PARTY_DEPENDENCIES} lib/pathmind
COPY ${SCRIPT} bin
RUN apt-get update && apt-get install -y \
    unzip
COPY ${JAR_FILE} pathmind-model-analyzer.jar
EXPOSE 8080
ENTRYPOINT ["java","-jar","/pathmind-model-analyzer.jar"]