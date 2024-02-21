from typing import Any, Literal, TypedDict

from mlflow import projects

from mlflow_executor.registry import Registry

registry = Registry()


def set_environmet(
    enviroment: dict[str, Any], upper_case: bool = False
) -> None:
    """Sets enviroment variables."""
    import os

    for k, v in enviroment.items():
        if upper_case:
            k = k.upper()
        os.environ[k] = v


class MLflowDictSummary(TypedDict):
    """Represent a MLflow run symmary.

    Parameters
    ---------
    run_id : str
        Run id associated to the MLflow run.

    run_cmd : str
        MLflow run executed command.

    exit_code : int
        MLflow run exit code.

    status : str
        MLflow run status.
    """

    run_id: str
    run_cmd: str
    exit_code: int
    status: str
    model_uri: str


class MLflowRunSummary:

    def __init__(self, submitted_run: projects.SubmittedRun) -> None:
        self.submitted_run = submitted_run

    def dict(self) -> MLflowDictSummary:
        """Returns dictionary containing all the summary properties."""
        return MLflowDictSummary(
            run_id=self.run_id,
            run_cmd=self.run_cmd,
            exit_code=self.exit_code,
            status=self.status,
            model_uri=self.model_uri,
        )

    @property
    def model_uri(self) -> str:
        return f"runs:/{self.run_id}/model"

    @property
    def run_id(self) -> str:
        """Returns the run id of the MLFlow run."""
        return self.submitted_run.run_id

    @property
    def run_cmd(self) -> str:
        """Returns the command ran by MLFlow."""
        return self.submitted_run.command_proc.args[-1].split("&& ")[-1]

    @property
    def exit_code(self) -> int:
        """Returns exit code from MLFlow run."""
        return self.submitted_run.command_proc.returncode

    @property
    def status(self) -> str:
        """Returns the status of the MLFlow run."""
        return self.submitted_run.get_status()


@registry()
def run_mlflow(
    uri: str | None = None,
    entry_point: str = "main",
    version: str | None = None,
    parameters: dict[str, Any] | None = None,
    experiment_name: str | None = None,
    experiment_id: str | None = None,
    storage_dir: str | None = None,
    run_name: str | None = None,
    env_manager: Literal["local", "virtualenv", "conda"] | None = None,
    environment: dict[str, Any] | None = None,
) -> projects.SubmittedRun:
    if environment is not None:
        set_environmet(environment, upper_case=True)

    submitted_run = projects.run(
        uri=uri,
        entry_point=entry_point,
        version=version,
        parameters=parameters,
        experiment_name=experiment_name,
        experiment_id=experiment_id,
        storage_dir=storage_dir,
        run_name=run_name,
        env_manager=env_manager,
    )

    submitted_run.wait()

    return MLflowRunSummary(submitted_run).dict()
