import collections
import json
import os.path
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

from mlflow.entities import ArtifactType
from mlflow.tracking import MlflowClient

from mlfoundry.exceptions import MlFoundryException
from mlfoundry.log_types.artifacts.artifact import (
    ArtifactPath,
    ArtifactVersion,
    ArtifactVersionInternalMetadata,
    _log_artifact_version_helper,
)
from mlfoundry.log_types.artifacts.constants import FILES_DIR, INTERNAL_METADATA_PATH
from mlfoundry.log_types.artifacts.utils import (
    _copy_additional_files,
    _validate_artifact_metadata,
    _validate_description,
)
from mlfoundry.logger import logger


def _log_artifact_version(
    run,
    name: str,
    artifact_paths: List[Union[ArtifactPath, Tuple[str, Optional[str]]]],
    mlflow_client: Optional[MlflowClient] = None,
    ml_repo_id: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    step: int = 0,
) -> ArtifactVersion:

    for i, artifact_path in enumerate(artifact_paths):
        if isinstance(artifact_path, ArtifactPath):
            continue
        elif isinstance(artifact_path, collections.abc.Sequence) and (
            0 < len(artifact_path) <= 2
        ):
            artifact_paths[i] = ArtifactPath(*artifact_path)
        else:
            raise ValueError(
                "`artifact_path` should be an instance of `mlfoundry.ArtifactPath` or a tuple of (src, dest) path strings"
            )

    metadata = metadata or {}
    step = step or 0

    _validate_description(description)
    _validate_artifact_metadata(metadata)

    logger.info("Logging the artifact, this might take a while ...")
    temp_dir = tempfile.TemporaryDirectory(prefix="truefoundry-")

    internal_metadata = ArtifactVersionInternalMetadata(
        files_dir=FILES_DIR,
    )

    try:
        local_files_dir = os.path.join(temp_dir.name, internal_metadata.files_dir)
        os.makedirs(local_files_dir, exist_ok=True)

        logger.info("Copying the files to log")
        _copy_additional_files(
            root_dir=temp_dir.name,
            files_dir=internal_metadata.files_dir,
            model_dir=None,
            additional_files=artifact_paths,
        )

    # TODO(nikp1172) verify error messag when artifact doesn't exist
    except Exception as e:
        temp_dir.cleanup()
        raise MlFoundryException("Failed to log artifact") from e

    # save internal metadata
    local_internal_metadata_path = os.path.join(temp_dir.name, INTERNAL_METADATA_PATH)
    os.makedirs(os.path.dirname(local_internal_metadata_path), exist_ok=True)
    with open(local_internal_metadata_path, "w") as f:
        json.dump(internal_metadata.dict(), f)

    return _log_artifact_version_helper(
        run=run,
        mlflow_client=mlflow_client,
        ml_repo_id=ml_repo_id,
        name=name,
        artifact_type=ArtifactType.ARTIFACT,
        artifact_dir=temp_dir,
        description=description,
        metadata=metadata,
        step=step,
    )
