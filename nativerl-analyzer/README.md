# pathmind-model-analyzer

A service that processes an AnyLogic model to extract data we need to use in Pathmind

## API

Service exposes one endpoint for retrieving a ZIP file contains `model.jar` and all other dependencies needed to perform an extraction hyperparameters process. <br/>
API specification is also available via [Swagger](https://swagger.io/), which can is accessible at `/swagger-ui.html` path.

### POST `/extract-hyperparametrs`

It requires one input `file` which has to be a valid ZIP file. Server will return JSON contains hyperparameters, reward function, information if model is single-agent or multi-agent (and a list of errors if any occurred):

```
{
	"actions": "4",
	"observations": "5",
	"rewardFunction": "new double[]{this.kitchenCleanliness, this.payBill.out.count(), this.custFailExit.countPeds(), this.serviceTime_min.mean()}",
	"mode": "single"
}
```

## Setting up local env

### Docker container

Install `pathmind-model-analyzer` docker image using [GitHub package](https://github.com/SkymindIO/pathmind-model-analyzer/packages/63675).<br/>
To run a service into docker container run:

```bash
$ docker run -p <HOST_PORT>:8080 pathmind-model-analyzer
```

where `<HOST_PORT>` is a port on which you want to communicate with container. <br/>
To check if service started (or if it is running) use [actuator](https://docs.spring.io/spring-boot/docs/current/reference/html/production-ready-features.html) healthcheck endpoint:

```bash
$ curl localhost:<HOST_PORT>/actuator/health
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100    15    0    15    0     0    238      0 --:--:-- --:--:-- --:--:--   238{"status":"UP"}
```

### IDE

**NOTE: Current implementation uses shared library `jniNativeRL.so` which is built for Unix systems, therefore, it is impossible to run it locally on Windows OS without containerizing or using a Virtual Machine** <br/>

To run local service instance using IDE:

- prepare `/lib/pathmind` directory contains:
  - unzipped content of `nativerl-1.7.1-SNAPSHOT-bin.zip`
  - unzipped content of `baseEnv.zip`
  - `cfr-0.148.jar` (curl -s https://www.benf.org/other/cfr/cfr-0.148.jar -o cfr-0.148.jar)
- prepare `/lib/policy` directory contains (naming is important):
  - PathmindPolicy_single.jar
  - PathmindPolicy_multi.jar
- copy both `check_model.sh` and `check_single_or_multi.sh` from `<repo>/resources/scripts` to `/bin`
- copy both `multi_extractor.jar` and `single_extractor.jar` from `<repo>/resources/` to `/bin`

You can also manually modify hardcoded paths in scripts, `FileService#CHECK_MODEL_SCRIPT` and `FileService#SINGLE_OR_MULTI_SCRIPT` variable to match your local ones.

### Building docker image

- At `Dockerfile` directory level run `$ docker build -t <image_name> --build-arg S3BUCKET='<s3_bucket>' --build-arg AWS_ACCESS_KEY_ID='<key_id>' --build-arg AWS_SECRET_ACCESS_KEY='<accesss_key>' .`
