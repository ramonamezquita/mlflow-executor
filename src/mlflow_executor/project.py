from typing import Any, Literal, TypedDict

from mlflow import projects

from mlflow_executor import (
    backend,
    backends,
    callback,
    deployer,
    deployers,
    execution,
    predictor,
)


class ProjectKwargs(TypedDict):
    uri: str | None
    entry_point: str
    version: str | None
    parameters: dict[str, Any] | None
    experiment_name: str | None
    experiment_id: str | None
    storage_dir: str | None
    run_name: str | None
    env_manager: Literal["local", "virtualenv", "conda"] | None
    environment: dict[str, Any] | None


class SubmittedRun:
    """Wrapper for mlflow.projects.SubmittedRun.

    Parameters
    ----------
    submitted_run : mlflow.projects.SubmittedRun.
    """

    def __init__(self, submitted_run: projects.SubmittedRun) -> None:
        self.submitted_run = submitted_run

    @property
    def model_uri(self) -> str:
        return f"runs:/{self.get_run_id()}/model"

    def is_done(self) -> bool:
        return self.submitted_run.get_status() == "FINISHED"

    def check_is_done(self) -> None:
        if not self.is_done():
            raise ValueError("This run has not completed yet.")

    def get_status(self) -> bool:
        return self.submitted_run.get_status()

    def deploy(
        self, deployer: deployer.Deployer = deployers.LocalDeployer()
    ) -> predictor.Predictor:
        self.check_is_done()

        return deployer.deploy(self.model_uri)

    def wait(self) -> bool:
        """Wait for the run to finish, returning True if the run succeeded and
        false otherwise
        """
        return self.submitted_run.wait()

    def get_run_id(self) -> str:
        return self.submitted_run.run_id

    def get_run_cmd(self) -> str:
        """Returns the command ran by MLFlow."""
        return self.submitted_run.command_proc.args[-1].split("&& ")[-1]

    def get_exit_code(self) -> int:
        """Returns exit code from MLFlow run."""
        return self.submitted_run.command_proc.returncode


def run(
    uri: str,
    parameters: dict[str, Any],
    callbacks: list[callback.Callback] = (),
    backend: backend.BackendExecutor = backends.LocalBackend(),
    entry_point: str = "main",
    experiment_name: str | None = None,
    experiment_id: str | None = None,
    storage_dir: str | None = None,
    run_name: str | None = None,
    env_manager: Literal["local", "virtualenv", "conda"] | None = None,
    environment: dict | None = None,
) -> SubmittedRun:
    """Runs MLflow project on the configured backend executor.

    Parameters
    ----------
    callbacks : list of callbacks.Callback, default=()
        Which callbacks to enable.

    backend : backend.BackendExecutor, default=LocalExecutor()
        Backend executor. Default local.

    entry_point : str, default="main"
        MLflow project entrypoint.

    experiment_name : str, default=None
        MLflow experiment name.

    experiment_id : str, default=None
        MLflow experiment id.

    storage_dir : str, default=None
        MLflow storage dir.

    enviroment : dict
        Enviroment variables to set for the run.
    """
    kwargs = ProjectKwargs(
        uri=uri,
        entry_point=entry_point,
        parameters=parameters,
        environment=environment,
        experiment_name=experiment_name,
        experiment_id=experiment_id,
        storage_dir=storage_dir,
        env_manager=env_manager,
        run_name=run_name,
    )

    executor = execution.TasksExecutor()
    promise = executor.execute(
        name="mlflow_executor.tasks.mlflow.run_mlflow",
        kwargs=kwargs,
        callbacks=callbacks,
        backend=backend,
    )

    return promise
