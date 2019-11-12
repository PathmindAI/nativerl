# pathmind-model-analyzer
A service that processes an AnyLogic model to extract data we need to use in Pathmind

## API
Service exposes one endpoint for retrieving a ZIP file contains `model.jar` and all other dependencies needed to perform an extraction hyperparameters process. <br/>
API specification is also available via [Swagger](https://swagger.io/), which can is accessible at `/swagger-ui.html` path.  

### POST `/extract-hyperparametrs`
It requires one input `file` which has to be a valid ZIP file. In case of successful extraction, server will return JSON contains hyperparameters: `{"actions": "4", "observations": "5"}`.


## Setting up local env
### Docker container
To run a service into docker container run:
```bash
$ docker run -p <HOST_PORT>:8080 pathmind-model-analyzer
```
where `<HOST_PORT>` is a port on which you want to communicate with container. <br/>
To check if service started (or if it is running) use [actuator](https://docs.spring.io/spring-boot/docs/current/reference/html/production-ready-features.html) healthcheck endpoint:
``` bash
$ curl localhost:<HOST_PORT>/actuator/health
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100    15    0    15    0     0    238      0 --:--:-- --:--:-- --:--:--   238{"status":"UP"}
```

### IDE
<i>(below instructions are based on current implementation which is likely to be changed and improved in the future)</i><br/>
**NOTE: Current implementation uses shared library `jniNativeRL.so` which is built for Unix systems, therefore, it is impossible to run it locally on Windows OS without containerizing or using a Virtual Machine** <br/>

To run local service instance using IDE:
* Create `/pathmind-lib` folder contains `baseEnv`, `nativer-bin`, `PathmindPolicy`
* Adjust `libDir` variable in `check_model.sh` script <br/>

### Building docker image

* Create `/pathmind-lib` folder contains `baseEnv`, `nativer-bin`, `PathmindPolicy`
* Run maven `package` goal to create new `jar` version 
* At `Dockerfile` directory level run `$ docker build -t pathmind-model-analyzer .`
