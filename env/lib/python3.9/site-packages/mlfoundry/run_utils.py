import argparse
import importlib
import json
import os
import tempfile
import time
import typing
from collections.abc import Mapping
from urllib.parse import urljoin, urlsplit

import mlflow
import numpy as np

from mlfoundry import constants, env_vars
from mlfoundry.exceptions import MlFoundryException


def get_module(
    module_name: str, error_message: typing.Optional[str] = None, required: bool = False
):
    try:
        return importlib.import_module(module_name)
    except Exception as ex:
        msg = error_message or f"Error importing module {module_name}"
        if required:
            raise MlFoundryException(msg) from ex


def resolve_tracking_uri(tracking_uri: typing.Optional[str]):
    return (
        tracking_uri
        or os.getenv(env_vars.TRACKING_HOST_GLOBAL)
        or constants.DEFAULT_TRACKING_URI
    )


def append_path_to_rest_tracking_uri(tracking_uri: str):
    if urlsplit(tracking_uri).netloc.startswith("localhost"):
        return tracking_uri
    return urljoin(tracking_uri, "/api/ml")


def append_servicefoundry_path_to_tracking_ui(tracking_uri: str):
    if urlsplit(tracking_uri).netloc.startswith("localhost"):
        return os.getenv("SERVICEFOUNDRY_SERVER_URL")
    return urljoin(tracking_uri, "/api/svc")


def append_monitoring_path_to_rest_tracking_uri(tracking_uri: str):
    if urlsplit(tracking_uri).netloc.startswith("localhost"):
        return tracking_uri
    return urljoin(tracking_uri, "/api/monitoring")


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


ParamsType = typing.Union[typing.Mapping[str, typing.Any], argparse.Namespace]


def process_params(params: ParamsType) -> typing.Dict[str, typing.Any]:
    if isinstance(params, Mapping):
        return params
    if isinstance(params, argparse.Namespace):
        return vars(params)
    # TODO: add absl support if required
    # move to a different file then
    raise MlFoundryException(
        "params should be either argparse.Namespace or a Mapping (dict) type"
    )


def log_artifact_blob(
    mlflow_client: mlflow.tracking.MlflowClient,
    run_id: str,
    blob: typing.Union[str, bytes],
    file_name: str,
    artifact_path: typing.Optional[str] = None,
):
    with tempfile.TemporaryDirectory(prefix=run_id) as tmpdirname:
        local_path = os.path.join(tmpdirname, file_name)
        mode = "wb" if isinstance(blob, bytes) else "w"
        with open(local_path, mode) as local_file:
            local_file.write(blob)
        mlflow_client.log_artifact(run_id, local_path, artifact_path=artifact_path)


def mapping_to_mlflow_metric(
    metrics: typing.Mapping[str, float],
    timestamp: typing.Optional[int] = None,
    step: int = 0,
) -> typing.Dict[str, mlflow.entities.Metric]:
    if timestamp is None:
        timestamp = int(time.time() * 1000)
    mlflow_metrics = {
        key: mlflow.entities.Metric(key, value, timestamp, step=step)
        for key, value in metrics.items()
    }
    return mlflow_metrics


def flatten_dict(
    input_dict: typing.Dict[typing.Any, typing.Any], parent_key="", sep="."
) -> typing.Dict[str, typing.Any]:
    """Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a.b': 'c'}``.
    All the keys will be converted to str.

    Args:
        input_dict: Dictionary containing the keys
        sep: Delimiter to express the hierarchy. Defaults to ``'.'``.

    Returns:
        Flattened dict.

    Examples:
        >>> flatten_dict({'a': {'b': 'c'}})
        {'a.b': 'c'}
        >>> flatten_dict({'a': {'b': 123}})
        {'a.b': 123}
        >>> flatten_dict({'a': {'b': 'c'}}, parent_key="param")
        {'param.a.b': 'c'}
    """
    new_dict_items = []
    for k, v in input_dict.items():
        new_key = parent_key + sep + str(k) if parent_key else k
        if isinstance(v, Mapping):
            new_dict_items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            new_dict_items.append((new_key, v))
    return dict(new_dict_items)
