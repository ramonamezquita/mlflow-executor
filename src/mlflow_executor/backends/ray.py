from typing import Any

import ray

from mlflow_executor import backend


@ray.remote
def _run(runner: backend.Runner):
    """Runs given task."""
    return runner.run()


class RayPromise(backend.Promise):
    def __init__(self, future: ray.ObjectRef):
        self.future = future

    def result(self) -> Any:
        return ray.get(self.future)


class RayBackend(backend.BackendExecutor):
    def run(self, runner: backend.BackendExecutor) -> RayPromise:
        future = _run.remote(runner)
        return RayPromise(future)
