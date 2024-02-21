from functools import cached_property
from typing import Any, Protocol
from urllib.parse import urljoin

from mlflow_executor import endpoint, serializers


class Serializer(Protocol):
    def serialize(self, data: Any) -> Any: ...


class Predictor:
    """Real time inference for MLflow hosted models."""

    def __init__(
        self,
        host: str,
        serializer: Serializer = serializers.IdentitySerializer(),
    ):
        self.host = host
        self.serializer = serializer

    @cached_property
    def predict_endpoint(self) -> endpoint.PostEndpoint:
        url = urljoin(self.host, "invocations")
        return endpoint.PostEndpoint(url)

    @cached_property
    def health_endpoint(self) -> endpoint.GetEndpoint:
        url = urljoin(self.host, "health")
        return endpoint.GetEndpoint(url)

    def predict(self, data: Any):
        """Returns the inference from the specified endpoint.

        Parameters
        ----------
        data : object
            Input data for which you want the model to provide inference.
            If a serializer was specified when creating the Predictor, the
            result of the serializer is sent as input data. Otherwise the data
            must be json serializable and the predict method then sends the
            data in the request body as is.
        """
        data = self.serializer.serialize(data)
        response = self.predict_endpoint.post(json=data)
        response.raise_for_status()

        return response.json()

    def health(self) -> None:
        response = self.health_endpoint.get()
        response.raise_for_status()
