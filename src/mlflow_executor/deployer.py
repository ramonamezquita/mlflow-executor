import abc

from mlflow_executor import predictor


class Deployer(abc.ABC):

    def deploy(self, model_uri: str) -> predictor.Predictor:
        host = self.run_server(model_uri)

        return predictor.Predictor(host)

    @abc.abstractmethod
    def run_server(self, model_uri: str) -> str:
        pass
