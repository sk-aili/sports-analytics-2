import copy
import datetime
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

from mlflow.entities import Artifact, ArtifactType
from mlflow.entities import ArtifactVersion as _ArtifactVersion
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Extra

from mlfoundry.artifact.truefoundry_artifact_repo import MlFoundryArtifactsRepository
from mlfoundry.exceptions import MlFoundryException
from mlfoundry.log_types.artifacts.constants import INTERNAL_METADATA_PATH
from mlfoundry.log_types.artifacts.utils import (
    _get_mlflow_client,
    _validate_artifact_metadata,
    _validate_description,
)
from mlfoundry.logger import logger

# TODO: Add some progress indicators for upload and download
# TODO: Support async download and upload


class ArtifactPath(NamedTuple):
    src: str
    dest: str = None


class ArtifactVersionInternalMetadata(BaseModel):
    class Config:
        extra = Extra.allow

    files_dir: str  # relative to root


class ArtifactVersionDownloadInfo(BaseModel):
    download_dir: str
    content_dir: str


class ArtifactVersion:
    def __init__(self, artifact_version: _ArtifactVersion, artifact: Artifact) -> None:
        self._mlflow_client = _get_mlflow_client()
        self._artifact_version: _ArtifactVersion = artifact_version
        self._artifact: Artifact = artifact
        self._deleted = False
        self._description: str = ""
        self._metadata: Dict[str, Any] = {}
        self._set_mutable_attrs()

    @classmethod
    def from_fqn(cls, fqn: str) -> "ArtifactVersion":
        mlflow_client = _get_mlflow_client()
        artifact_version = mlflow_client.get_artifact_version_by_fqn(fqn=fqn)
        artifact = mlflow_client.get_artifact_by_id(
            artifact_id=artifact_version.artifact_id
        )
        return cls(artifact_version=artifact_version, artifact=artifact)

    def _ensure_not_deleted(self):
        if self._deleted:
            raise MlFoundryException(
                "Artifact Version was deleted, cannot access a deleted version"
            )

    def _set_mutable_attrs(self, refetch=False):
        if refetch:
            self._artifact_version = self._mlflow_client.get_artifact_version_by_id(
                version_id=self._artifact_version.id
            )
        self._description = self._artifact_version.description or ""
        self._metadata = copy.deepcopy(self._artifact_version.artifact_metadata)

    def __repr__(self):
        return f"{self.__class__.__name__}(fqn={self.fqn!r})"

    @property
    def name(self) -> str:
        return self._artifact.name

    @property
    def artifact_fqn(self) -> str:
        return self._artifact.fqn

    @property
    def version(self) -> int:
        return self._artifact_version.version

    @property
    def fqn(self) -> str:
        return self._artifact_version.fqn

    @property
    def step(self) -> int:
        return self._artifact_version.step

    @property
    def description(self) -> Optional[str]:
        return self._description

    @description.setter
    def description(self, value: str):
        _validate_description(value)
        self._description = value

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @metadata.setter
    def metadata(self, value: Dict[str, Any]):
        _validate_artifact_metadata(value)
        self._metadata = value

    @property
    def created_by(self) -> str:
        return self._artifact_version.created_by

    @property
    def created_at(self) -> datetime.datetime:
        return self._artifact_version.created_at

    @property
    def updated_at(self) -> datetime.datetime:
        return self._artifact_version.updated_at

    def raw_download(self, path: Optional[Union[str, Path]]) -> str:
        logger.info(
            "Downloading artifact version contents, this might take a while ..."
        )
        mlfa_repo = MlFoundryArtifactsRepository(
            version_id=self._artifact_version.id, mlflow_client=self._mlflow_client
        )
        return mlfa_repo.download_artifacts(artifact_path="", dst_path=path)

    def _download(
        self, path: Optional[Union[str, Path]]
    ) -> Tuple[ArtifactVersionInternalMetadata, str]:
        self._ensure_not_deleted()
        download_dir = self.raw_download(path=path)
        internal_metadata_path = os.path.join(download_dir, INTERNAL_METADATA_PATH)
        if not os.path.exists(internal_metadata_path):
            raise MlFoundryException(
                f"Artifact version seems to be corrupted or in invalid format due to missing artifact metadata. "
                f"You can still use .raw_download(path='/your/path/here') to download and inspect files."
            )
        with open(internal_metadata_path) as f:
            internal_metadata = ArtifactVersionInternalMetadata.parse_obj(json.load(f))
        download_path = os.path.join(download_dir, internal_metadata.files_dir)
        return internal_metadata, download_path

    def download(self, path: Optional[Union[str, Path]]) -> str:
        _, download_path = self._download(path=path)
        return download_path

    def delete(self) -> bool:
        self._ensure_not_deleted()
        self._mlflow_client.delete_artifact_version(
            version_id=self._artifact_version.id
        )
        self._deleted = True
        return True

    def update(self):
        self._ensure_not_deleted()

        self._artifact_version = self._mlflow_client.update_artifact_version(
            version_id=self._artifact_version.id,
            description=self.description,
            artifact_metadata=self.metadata,
        )
        self._set_mutable_attrs()


def calculate_artifact_size(artifact_dir: tempfile.TemporaryDirectory):
    total_size = 0
    for path, dirs, files in os.walk(artifact_dir.name):
        for f in files:
            file_path = os.path.join(path, f)
            total_size += os.stat(file_path).st_size
    return total_size


def _log_artifact_version_helper(
    run,
    name: str,
    artifact_type: ArtifactType,
    artifact_dir: tempfile.TemporaryDirectory,
    mlflow_client: Optional[MlflowClient] = None,
    ml_repo_id: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    step: int = 0,
):
    if (run and mlflow_client) or (not run and not mlflow_client):
        raise MlFoundryException("Exactly one of run, mlflow_client should be passed")
    if mlflow_client and not ml_repo_id:
        raise MlFoundryException(
            "If mlflow_client is passed, ml_repo_id must also be passed"
        )
    if run:
        mlflow_client: MlflowClient = run.mlflow_client

    version_id = mlflow_client.create_artifact_version(
        experiment_id=int(run._experiment_id) if run else ml_repo_id,
        artifact_type=artifact_type,
        name=name,
    )
    artifacts_repo = MlFoundryArtifactsRepository(
        version_id=version_id, mlflow_client=mlflow_client
    )
    total_size = calculate_artifact_size(artifact_dir)
    try:
        logger.info(
            "Packaging and uploading files to remote with Artifact Size: %.6f MB",
            total_size / 1000000.0,
        )
        artifacts_repo.log_artifacts(local_dir=artifact_dir.name, artifact_path=None)
    except Exception as e:
        mlflow_client.notify_failure_for_artifact_version(version_id=version_id)
        raise MlFoundryException("Failed to log Artifact") from e
    finally:
        artifact_dir.cleanup()
    artifact_version = mlflow_client.finalize_artifact_version(
        version_id=version_id,
        run_uuid=run.run_id if run else None,
        description=description,
        artifact_metadata=metadata,
        data_path=INTERNAL_METADATA_PATH,
        step=step,
        artifact_size=total_size,
    )

    return ArtifactVersion.from_fqn(fqn=artifact_version.fqn)
