import abc

from mlflow_executor import endpoint, predictor


class Deployer(abc.ABC):

    def deploy(self, model_uri: str) -> predictor.Predictor:
        url = self.run_server(model_uri)

        endpoint.Endpoint(url, endpoint_prefix=model_uri)
        return predictor.Predictor(endpoint)

    @abc.abstractmethod
    def run_server(self, model_uri: str) -> str:
        pass
