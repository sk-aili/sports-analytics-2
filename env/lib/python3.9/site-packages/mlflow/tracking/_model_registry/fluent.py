from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS, ErrorCode
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.tracking import MlflowClient
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.utils.logging_utils import eprint


def register_model(
    model_uri, name, await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS
) -> ModelVersion:
    """
    Create a new model version in model registry for the model files specified by ``model_uri``.
    Note that this method assumes the model registry backend URI is the same as that of the
    tracking backend.

    :param model_uri: URI referring to the MLmodel directory. Use a ``runs:/`` URI if you want to
                      record the run ID with the model in model registry. ``models:/`` URIs are
                      currently not supported.
    :param name: Name of the registered model under which to create a new model version. If a
                 registered model with the given name does not exist, it will be created
                 automatically.
    :param await_registration_for: Number of seconds to wait for the model version to finish
                            being created and is in ``READY`` status. By default, the function
                            waits for five minutes. Specify 0 or None to skip waiting.
    :return: Single :py:class:`mlflow.entities.model_registry.ModelVersion` object created by
             backend.

    .. code-block:: python
        :caption: Example

        import mlflow.sklearn
        from sklearn.ensemble import RandomForestRegressor

        mlflow.set_tracking_uri("sqlite:////tmp/mlruns.db")
        params = {"n_estimators": 3, "random_state": 42}

        # Log MLflow entities
        with mlflow.start_run() as run:
           rfr = RandomForestRegressor(**params).fit([[0, 1]], [1])
           mlflow.log_params(params)
           mlflow.sklearn.log_model(rfr, artifact_path="sklearn-model")

        model_uri = "runs:/{}/sklearn-model".format(run.info.run_id)
        mv = mlflow.register_model(model_uri, "RandomForestRegressionModel")
        print("Name: {}".format(mv.name))
        print("Version: {}".format(mv.version))

    .. code-block:: text
        :caption: Output

        Name: RandomForestRegressionModel
        Version: 1
    """
    raise NotImplementedError(
        "This method was deleted intentionally. Please refer to git history to restore it"
    )
