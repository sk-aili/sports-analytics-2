import os
import posixpath
import re
import tempfile
from contextlib import contextmanager

from mlfoundry.exceptions import MlFoundryException

KEY_REGEX = re.compile(r"^[a-zA-Z0-9-_]+$")


def validate_key_name(key: str):
    if not key or not KEY_REGEX.match(key):
        raise MlFoundryException(
            f"Invalid run image key: {key} should only contain alphanumeric, hyphen or underscore"
        )


@contextmanager
def save_artifact_helper(run: "mlfoundry.MlFoundryRun", artifact_path: str):
    if not artifact_path:
        raise MlFoundryException("artifact_path cannot be None or empty string")
    artifact_path = posixpath.normpath(artifact_path)
    file_name = posixpath.basename(artifact_path)
    artifact_dir = posixpath.dirname(artifact_path)

    # Setting to None to make sure the file gets saved at the
    # run's root artifact_path if the artifact_path is like "foo.txt"
    artifact_dir = artifact_dir if artifact_dir else None
    if not file_name:
        raise MlFoundryException(f"could not infer file name from {artifact_path}")

    with tempfile.TemporaryDirectory() as local_dir:
        local_path = os.path.join(local_dir, file_name)
        yield local_path
        run.mlflow_client.log_artifact(
            run_id=run.run_id, local_path=local_path, artifact_path=artifact_dir
        )
