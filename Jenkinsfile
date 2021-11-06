def SLACK_URL = 'https://hooks.slack.com/services/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'
def icon = ':heavy_check_mark:'
/*
    nativerl pipeline
    The pipeline is made up of following steps
    1. Git clone and setup
    2. Build and s3 push
    4. Optionally deploy to production and test
 */

/*
    Build a docker image
*/
def buildNativerl(image_name) {
    echo "Building the nativerl Docker Image for branch ${env.BRANCH_NAME}"
    sh """
        set +x
        docker image ls | grep nativerl | awk '{print \$3}' | xargs -I {} docker rmi {} -f
        docker build -t ${image_name} -f ${WORKSPACE}/Dockerfile ${WORKSPACE}
    """
    sh "docker run --mount \"src=${WORKSPACE}/,target=/app,type=bind\" nativerl mvn clean install -Djavacpp.platform=linux-x86_64"
    sh "aws s3 cp ${WORKSPACE}/nativerl/target/nativerl-1.8.1-SNAPSHOT-bin.zip s3://${env.BRANCH_NAME}-training-static-files.pathmind.com/nativerl/1_8_1/nativerl-1.8.1-SNAPSHOT-bin.zip"
}

/*
    Build the conda package
*/
def buildRllibpack(image_name) {
    echo "Building the rllibpack Docker Image for branch ${env.BRANCH_NAME}"
    sh """
        set +x
        docker image ls | grep rllibpack | awk '{print \$3}' | xargs -I {} docker rmi {} -f
        docker build -t ${image_name} -f ${WORKSPACE}/rllibpack/Dockerfile ${WORKSPACE}/rllibpack/
    """
    sh "docker run -v ${WORKSPACE}/rllibpack/:/home/rllibpack/app rllibpack"
    sh "aws s3 cp ${WORKSPACE}/rllibpack/rllibpack.tar.gz s3://slin-training-dynamic-files.pathmind.com/test/conda/1_3_0/rllibpack.tar.gz"
}

/*
    This is the main pipeline section with the stages of the CI/CD
 */
pipeline {
    options {
        // Build auto timeout
        timeout(time: 60, unit: 'MINUTES')
        disableConcurrentBuilds()
    }

    // Some global default variables
    environment {
        IMAGE_NAME = 'nativerl'
        DEPLOY_PROD = false
    }

    //all is built and run from the master
    agent { node { label 'master' } }

    // Pipeline stages
    stages {
        stage('Git clone and setup') {
            when {
                anyOf {
                    environment name: 'GIT_BRANCH', value: 'dev'
                    environment name: 'GIT_BRANCH', value: 'test'
                    environment name: 'GIT_BRANCH', value: 'staging'
                    environment name: 'GIT_BRANCH', value: 'prod'
                }
            }
            steps {
                echo 'Notifying slack'
                sh "set +x; curl -X POST -H 'Content-type: application/json' --data '{\"text\":\":building_construction: Starting Jenkins Job\nBranch: ${env.BRANCH_NAME}\nUrl: ${env.RUN_DISPLAY_URL}\"}' ${SLACK_URL}"
                echo 'Check out code'
                checkout scm
            }
        }

        stage('Build Docker Images') {
            parallel {
                stage('Build nativerl image') {
                    when {
                        anyOf {
                            environment name: 'GIT_BRANCH', value: 'dev'
                            environment name: 'GIT_BRANCH', value: 'test'
                            environment name: 'GIT_BRANCH', value: 'staging'
                            environment name: 'GIT_BRANCH', value: 'prod'
                        }
                    }
                    steps {
                        buildNativerl("${IMAGE_NAME}")
                    }
                }
                stage('Build Rllibpack') {
                    when {
                        anyOf {
                            environment name: 'GIT_BRANCH', value: 'dev'
                            environment name: 'GIT_BRANCH', value: 'test'
                            environment name: 'GIT_BRANCH', value: 'staging'
                            environment name: 'GIT_BRANCH', value: 'prod'
                        }
                    }
                    steps {
                        buildRllibpack('rllibpack')
                    }
                }
            }
        }
    }
    post {
        always {
            echo 'Notifying Slack'
            script {
                if (currentBuild.result != 'SUCCESS') {
                    icon = ':x:'
                }
            }
            echo 'Notifying slack'
            sh "set +x; curl -X POST -H 'Content-type: application/json' --data '{\"text\":\"${icon} Jenkins Job Finished\nBranch: ${env.BRANCH_NAME}\nUrl: ${env.RUN_DISPLAY_URL}\nStatus: ${currentBuild.result}\"}' ${SLACK_URL}"
        }
    }
}
