import unittest

from mlflow_executor import backend, backends
from mlflow_executor.testing.runners import URLLoader


class BackendTest:

    #: Backend executor for the test.
    _backend: backend.BackendExecutor = None

    def test_urlloader(self):
        runner = URLLoader()
        promise = self._backend.run(runner)


class TestLocalBackend(BackendTest, unittest.TestCase):
    _backend = backends.LocalBackend()


class TestrRayBackend(BackendTest, unittest.TestCase):
    _backend = backends.RayBackend()
