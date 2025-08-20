from pathlib import Path
from typing import Dict, Any, Optional

import mlflow


class MlflowLogger:
    def __init__(self, tracking_uri: str, experiment_name: str, run_name: Optional[str] = None) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run_ctx = mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: Path) -> None:
        mlflow.log_artifact(str(path))

    def end(self) -> None:
        mlflow.end_run()


