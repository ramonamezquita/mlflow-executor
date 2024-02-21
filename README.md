End-to-end training and deployment of MLflow projects.


# Example
Train and deploy [this]("https://github.com/mlflow/mlflow-example") MLflow 
example project using the Ray executor.


```python

from mlflow_executor import project, backends, deployers

#Project uri.
uri = "https://github.com/mlflow/mlflow-example"

# Project params.
params = {"alpha": 0.5, "l1_ratio": 0.01}
    
# Project env manager.
# Different options are
#   - local: use the local environment
#   - conda: use conda (must have conda installed).
env_manager = "virtualenv"


# The object ``promise`` contains a reference to the executed task.
promise = project.run(
    uri=uri, 
    parameters=params, 
    env_manager=env_manager 
    backend=backends.RayBackend()
)


# Return the result of the call that ``promise`` represents.
result = promise.result()


# Choose your preferred deployer (e.g., :class:`deployer.LocalDeployer`).
# All deployers return a :class:`predictor.Predictor` instance that handles
# POST requests to the model inference service.
deployer = deployers.LocalDeployer()
predictor = deployer.deploy(result['model_uri'])

# Check health of the model inference service.
predictor.health()
```

