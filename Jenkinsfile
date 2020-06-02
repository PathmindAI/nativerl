def ENVIRONMENT
def SLACK_URL = "https://hooks.slack.com/services/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"
def icon = ":heavy_check_mark:"
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
def buildNativerl(image_name, environment, version) {
    echo "Building the nativerl Docker Image"
    sh "docker build -t ${image_name} -f ${WORKSPACE}/nativerl/Dockerfile ${WORKSPACE}/nativerl"
    sh "docker run --mount \"src=${WORKSPACE}/nativerl/nativerl,target=/app,type=bind\" nativerl mvn clean package -Djavacpp.platform=linux-x86_64"
    sh "aws s3 cp ${WORKSPACE}/nativerl/nativerl/target/nativerl-1.0.0-SNAPSHOT-bin.zip s3://${environment}-training-static-files.pathmind.com/nativerl/${version}/nativerl-1.0.0-SNAPSHOT-bin.zip"
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

    parameters {
        string(name: 'GIT_BRANCH', defaultValue: 'test', description: 'Git branch to build')
        booleanParam(name: 'DEPLOY_TO_PROD', defaultValue: false, description: 'If build and tests are good, proceed and deploy to production without manual approval')

    }

    //all is built and run from the master
    agent { node { label 'master' } }

    // Pipeline stages
    stages {
        stage('Git clone and setup') {
            //when {
            //    anyOf {
            //        environment name: 'GIT_BRANCH', value: 'dev'
            //        environment name: 'GIT_BRANCH', value: 'test'
            //        environment name: 'GIT_BRANCH', value: 'master'
            //    }
            //}
            steps {
                echo "Notifying slack"
                sh "set +x; curl -X POST -H 'Content-type: application/json' --data '{\"text\":\":building_construction: Starting Jenkins Job\nBranch: ${env.BRANCH_NAME}\nUrl: ${env.RUN_DISPLAY_URL}\"}' ${SLACK_URL}"
                sh "env"
                script {
                    ENVIRONMENT = "dev"
                    if (env.BRANCH_NAME == 'master') {
                        ENVIRONMENT = "prod"
                    }
                    if (env.BRANCH_NAME == 'dev') {
                        ENVIRONMENT = "dev"
                    }
                    if (env.BRANCH_NAME == 'test') {
                        ENVIRONMENT = "test"
                    }
                }
                echo "Check out code"
                checkout scm
            }
        }

        stage('Build Docker Images') {
            when {
                anyOf {
                    environment name: 'GIT_BRANCH', value: 'dev'
                    environment name: 'GIT_BRANCH', value: 'test'
                    environment name: 'GIT_BRANCH', value: 'master'
                }
            }
            parallel {
                stage('Build nativerl image') {
                    steps {
                        buildNativerl("${IMAGE_NAME}", "${ENVIRONMENT}", "${VERSION}")
                    }
                }
            }
        }
    }
    post {
        always {
            echo 'Notifying Slack'
            script {
                if (currentBuild.result != "SUCCESS") {
                    icon = ":x:"
                }
            }
            echo "Notifying slack"
            sh "set +x; curl -X POST -H 'Content-type: application/json' --data '{\"text\":\"${icon} Jenkins Job Finished\nBranch: ${env.BRANCH_NAME}\nUrl: ${env.RUN_DISPLAY_URL}\nStatus: ${currentBuild.result}\"}' ${SLACK_URL}"
        }
    }
}

