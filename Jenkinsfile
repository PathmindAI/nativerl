def DOCKER_TAG
/*
    pathmind-webapp pipeline
    The pipeline is made up of following steps
    1. Git clone and setup
    2. Build and local tests
    3. Publish Docker and Helm
    4. Optionally deploy to production and test
 */

/*
    Build a docker image
*/
def buildDockerImage(image_name, image_id) {
        echo "Building the pathmind Docker Image"
        //Dont quit if there is an error
        NULL = sh(returnStatus: true, script: "docker image rm pathmind-ma")
        sh """
            set +x
            docker build -t ${image_name} -f ${WORKSPACE}/Dockerfile --build-arg S3BUCKET='${env.S3BUCKET}' --build-arg AWS_ACCESS_KEY_ID=`kubectl get secret awsaccesskey -o=jsonpath='{.data.AWS_ACCESS_KEY_ID}' | base64 --decode` --build-arg AWS_SECRET_ACCESS_KEY=`kubectl get secret awssecretaccesskey -o=jsonpath='{.data.AWS_SECRET_ACCESS_KEY}' | base64 --decode` ${WORKSPACE}/
        """
}

/*
    Publish a docker image
*/
def publishDockerImage(image_name,DOCKER_TAG) {
	echo "Logging to aws ecr"
	sh "aws ecr get-login --no-include-email --region us-east-1 | sh"
	echo "Tagging and pushing the pathmind Docker Image"                
	sh "docker tag ${image_name} ${DOCKER_REG}/${image_name}:${DOCKER_TAG}"
	sh "docker push ${DOCKER_REG}/${image_name}"
}


/*
    This is the main pipeline section with the stages of the CI/CD
 */
pipeline {
    options {
        // Build auto timeout
        timeout(time: 60, unit: 'MINUTES')
    }

    // Some global default variables
    environment {
        IMAGE_NAME = 'pathmind-ma'
        DOCKER_REG = "839270835622.dkr.ecr.us-east-1.amazonaws.com"
	DEPLOY_PROD = false
    }

    parameters {
        string (name: 'GIT_BRANCH',           defaultValue: 'test',  description: 'Git branch to build')
        booleanParam (name: 'DEPLOY_TO_PROD', defaultValue: false,     description: 'If build and tests are good, proceed and deploy to production without manual approval')

    }

    //all is built and run from the master
    agent { node { label 'master' } }

    // Pipeline stages
    stages {

        ////////// Step 1 //////////
        stage('Git clone and setup') {
            when {
                anyOf {
                    environment name: 'GIT_BRANCH', value: 'dev'
                    environment name: 'GIT_BRANCH', value: 'test'
                    environment name: 'GIT_BRANCH', value: 'master'
                }
            }
            steps {
		script {
        		DOCKER_TAG = 'test'
		        if(env.BRANCH_NAME == 'master'){
		                DOCKER_TAG = "prod"
		        }
		        if(env.BRANCH_NAME == 'dev'){
		                DOCKER_TAG = "dev"
		        }
		        if(env.BRANCH_NAME == 'test'){
		                DOCKER_TAG = "test"
		        }
		}
                echo "Check out code"
		checkout scm

                // Validate kubectl
                sh "kubectl cluster-info"

                // Helm version
                sh "helm version"

                // Define a unique name for the tests container and helm release
                script {
                    branch = GIT_BRANCH.replaceAll('/', '-').replaceAll('\\*', '-')
                    PATHMIND_ID = "${IMAGE_NAME}-${DOCKER_TAG}-${branch}"
                    echo "Global pathmind Id set to ${PATHMIND_ID}"
                }
            }
        }

        ////////// Step 2 //////////
        stage('Build Docker Images') {
            when {
                anyOf {
                    environment name: 'GIT_BRANCH', value: 'dev'
                    environment name: 'GIT_BRANCH', value: 'test'
                    environment name: 'GIT_BRANCH', value: 'master'
                }
            }
		parallel {
			stage('Build pathmind image') {
				steps {
					buildDockerImage("${IMAGE_NAME}","${PATHMIND_ID}")
				}
			}
		}
        }

	////////// Step 3 //////////
        stage('Publish Docker Images') {
            when {
                anyOf {
                    environment name: 'GIT_BRANCH', value: 'dev'
                    environment name: 'GIT_BRANCH', value: 'test'
                    environment name: 'GIT_BRANCH', value: 'master'
                }
            }
		parallel {
			stage('Publish pathmind image') {
				steps {
					publishDockerImage("${IMAGE_NAME}","${DOCKER_TAG}")
				}
			}
		}
        } 

	////////// Step 4 //////////
	stage('Deploying helm chart') {
            when {
                anyOf {
                    environment name: 'GIT_BRANCH', value: 'dev'
                    environment name: 'GIT_BRANCH', value: 'test'
                }
            }
            steps {
		script {
				echo "Cloning infra repository"
                                sh "if [ -d pathmind-webapp ]; then rm -rf pathmind-webapp; fi"
				sh "git clone git@github.com:SkymindIO/pathmind-webapp.git"
				echo "Updating helm chart"
				sh "cd pathmind-webapp;helm upgrade --install pathmind-ma infra/helm/pathmind-ma -f infra/helm/pathmind-ma/values_${DOCKER_TAG}.yaml -n ${DOCKER_TAG}"
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

	////////// Step 6 //////////
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
			echo "Cloning infra repository"
                        sh "if [ -d pathmind-webapp ]; then rm -rf pathmind-webapp; fi"
			sh "git clone git@github.com:SkymindIO/pathmind-webapp.git"
			echo "Updating helm chart"
			sh "cd pathmind-webapp;helm upgrade --install pathmind-ma infra/helm/pathmind-ma -f infra/helm/pathmind-ma/values_${DOCKER_TAG}.yaml"
			sh "sleep 30"
		}
            }
        }

	////////// Step 7 //////////
	stage('Testing in Production') {
            when {
                expression { DEPLOY_PROD == true }
            }
            steps {
		echo "Testing in Production"
		//sh "python ${WORKSPACE}/src/jenkins/tests/web_prod.py"
            }
        } 
   }
}

