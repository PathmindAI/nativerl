def DOCKER_TAG
def SLACK_URL="https://hooks.slack.com/services/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"
def icon=":heavy_check_mark:"
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
def buildDockerImage(image_name, image_id) {
        echo "Building the nativerl Docker Image"
        sh "docker build -t ${image_name} -f ${WORKSPACE}/Dockerfile ${WORKSPACE}/"
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
        string (name: 'GIT_BRANCH', defaultValue: 'test', description: 'Git branch to build')
        booleanParam (name: 'DEPLOY_TO_PROD', defaultValue: false, description: 'If build and tests are good, proceed and deploy to production without manual approval')

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
                    environment name: 'GIT_BRANCH', value: '58-nativrl-ci-cd'
                    environment name: 'GIT_BRANCH', value: 'master'
                }
            }
            steps {
                echo "Notifying slack"
		sh "set +x; curl -X POST -H 'Content-type: application/json' --data '{\"text\":\":building_construction: Starting Jenkins Job\nBranch: ${env.BRANCH_NAME}\nUrl: ${env.RUN_DISPLAY_URL}\"}' ${SLACK_URL}"
		script {
		        DOCKER_TAG = "dev"
		        if(env.BRANCH_NAME == 'master'){
		                DOCKER_TAG = "prod"
		        }
		        if(env.BRANCH_NAME == 'dev'){
		                DOCKER_TAG = "dev"
		        }
		        if(env.BRANCH_NAME == 'test'){
		                DOCKER_TAG = "test"
		        }
		        if(env.BRANCH_NAME == '58-nativrl-ci-cd'){
		                DOCKER_TAG = "test"
		        }
		}
                echo "Check out code"
		checkout scm

                // Define a unique name for the tests container and helm release
                script {
                    branch = GIT_BRANCH.replaceAll('/', '-').replaceAll('\\*', '-')
                    NATIVERL_ID = "${IMAGE_NAME}-${DOCKER_TAG}-${branch}"
                    echo "Global nativerl Id set to ${NATIVERL_ID}"
                }
            }
        }

        stage('Build Docker Images') {
            when {
                anyOf {
                    environment name: 'GIT_BRANCH', value: 'dev'
                    environment name: 'GIT_BRANCH', value: 'test'
                    environment name: 'GIT_BRANCH', value: '58-nativrl-ci-cd'
                    environment name: 'GIT_BRANCH', value: 'master'
                }
            }
		parallel {
			stage('Build nativerl image') {
				steps {
					buildDockerImage("${IMAGE_NAME}","${NATIVERL_ID}")
				}
			}
		}
        }

        stage('Build Docker Images') {
            when {
                anyOf {
                    environment name: 'GIT_BRANCH', value: 'dev'
                    environment name: 'GIT_BRANCH', value: 'test'
                    environment name: 'GIT_BRANCH', value: '58-nativrl-ci-cd'
                }
            }
                parallel {
                        stage('Build nativerl image') {
                                steps {
					script{
						sh "env"
					}
                                }
                        }
                }
        }

        // Waif for user manual approval, or proceed automatically if DEPLOY_TO_PROD is true
        stage('Go for Production?') {
            when {
                allOf {
                    environment name: 'GIT_BRANCH', value: 'master'
                    environment name: 'DEPLOY_TO_PROD', value: 'false'
                }
            }

            steps {
                // Prevent any older builds from deploying to production
                milestone(1)
                input 'Proceed and deploy to Production?'
                milestone(2)

                script {
                    DEPLOY_PROD = true
                }
            }
        }

	stage('Deploying to Production') {
	    when {
                anyOf {
                    expression { DEPLOY_PROD == true }
                    environment name: 'DEPLOY_TO_PROD', value: 'true'
                }
            }
            steps {
		script {
                	DEPLOY_PROD = true
		}
            }
        }
   }
   post {
        always {
		echo 'Notifying Slack'
		script {
			if ( currentBuild.result != "SUCCESS" ) {
				icon=":x:"
			}
		}
                echo "Notifying slack"
		sh "set +x; curl -X POST -H 'Content-type: application/json' --data '{\"text\":\"${icon} Jenkins Job Finished\nBranch: ${env.BRANCH_NAME}\nUrl: ${env.RUN_DISPLAY_URL}\nStatus: ${currentBuild.result}\"}' ${SLACK_URL}"
        }
    }
}
