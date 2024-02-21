import unittest

from mlflow_executor import backends, project


class RunTest:

    def __init__(self, run_summary, expected_cmd) -> None:
        self.run_summary = run_summary
        self.expected_cmd = expected_cmd

    def test_all(self):
        self.test_status()
        self.test_exit_code()
        self.test_run_cmd()

    def test_status(self) -> None:
        assert self.run_summary["status"] == "FINISHED"

    def test_exit_code(self) -> None:
        assert self.run_summary["exit_code"] == 0

    def test_run_cmd(self) -> None:
        assert self.run_summary["run_cmd"] == self.expected_cmd


class TestMLflowExample(unittest.TestCase):

    #: Project uri.
    uri = "https://github.com/mlflow/mlflow-example"

    #: Project params.
    params = {"alpha": 0.5, "l1_ratio": 0.01}

    #: Project env manager.
    env_manager = "virtualenv"

    #: Project expected command.
    expected_cmd = "python train.py 0.5 0.01"

    def run_project(self, backend):
        return project.run(
            uri=self.uri,
            parameters=self.params,
            env_manager=self.env_manager,
            backend=backend,
        )

    def test_local_backend(self) -> None:
        promise = self.run_project(backends.LocalBackend())
        result = promise.result()

        test = RunTest(result, self.expected_cmd)
        test.test_all()

    def test_ray_backend(self) -> None:
        promise = self.run_project(backends.RayBackend())
        result = promise.result()

        test = RunTest(result, self.expected_cmd)
        test.test_all()
