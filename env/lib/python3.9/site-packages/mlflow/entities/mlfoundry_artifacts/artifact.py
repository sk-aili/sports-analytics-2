import datetime
import posixpath

import uuid
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, constr

from mlflow.entities.experiment import Experiment
from mlflow.entities.mlfoundry_artifacts import utils
from mlflow.entities.mlfoundry_artifacts.enums import (
    ArtifactType,
    ArtifactVersionStatus,
)
from mlflow.exceptions import MlflowException
from mlflow.protos import mlfoundry_artifacts_pb2 as mlfa_pb2
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.proto_json_utils import get_field_if_set
from mlflow.utils.uri import append_to_uri_path

_ARTIFACT_FQN_FORMAT = "{artifact_type}:{tenant_name}/{experiment_name}/{artifact_name}"
_ARTIFACT_VERSION_FQN_FORMAT = "{artifact_fqn}:{version}"

_ARTIFACT_VERSION_USAGE_CODE_SNIPPET = """import mlfoundry as mlf
client = mlf.get_client()

# Get the artifact version directly
artifact_version = client.get_artifact("{fqn}")

# OR reference it another run
run = client.create_run(project_name="<YOUR-PROJECT-NAME>", run_name="<YOUR-RUN-NAME>")
artifact_version = run.use_artifact("{fqn}")

# download it to disk
artifact_version.download(path="your/download/location")"""


class BaseArtifactMixin(BaseModel):
    @staticmethod
    def generate_fqn(
        experiment: Experiment, artifact_type: ArtifactType, artifact_name: str
    ) -> str:
        if not experiment.tenant_name:
            raise ValueError(f"Attributes `tenant_name` on `experiment` cannot be None")
        return _ARTIFACT_FQN_FORMAT.format(
            artifact_type=artifact_type.value,
            tenant_name=experiment.tenant_name,
            experiment_name=experiment.name,
            artifact_name=artifact_name,
        )

    @staticmethod
    def generate_storage_root(
        experiment: Experiment, artifact_type: ArtifactType, artifact_name: str
    ) -> str:
        # noinspection PyTypeChecker
        return append_to_uri_path(
            experiment.artifact_location, "artifacts", artifact_type.value, artifact_name, posixpath.sep
        )


class Artifact(BaseArtifactMixin):
    class Config:
        orm_mode = True
        validate_assignment = True
        use_enum_values = True
        allow_mutation = False
        smart_union = True

    id: uuid.UUID
    experiment_id: int
    type: ArtifactType
    name: constr(regex=r"^[A-Za-z0-9_\-]+$", max_length=256)
    fqn: str
    description: Optional[constr(max_length=1024)] = None
    artifact_storage_root: str
    created_by: constr(max_length=256)
    latest_version: Optional["ArtifactVersion"] = None
    run_steps: List[int] = Field(default_factory=list)

    created_at: datetime.datetime
    updated_at: datetime.datetime

    def to_proto(self) -> mlfa_pb2.Artifact:
        message = mlfa_pb2.Artifact(
            id=str(self.id),
            experiment_id=str(self.experiment_id),
            type=ArtifactType(self.type).to_proto(),
            name=self.name,
            fqn=self.fqn,
            artifact_storage_root=self.artifact_storage_root,
            description=self.description or "",
            created_by=self.created_by,
            created_at=self.created_at.astimezone(tz=datetime.timezone.utc).isoformat(),
            updated_at=self.updated_at.astimezone(tz=datetime.timezone.utc).isoformat(),
            run_steps=self.run_steps,
        )
        if self.latest_version:
            message.latest_version.MergeFrom(self.latest_version.to_proto())
        return message

    @classmethod
    def from_proto(cls, message: mlfa_pb2.Artifact) -> "Artifact":
        description = get_field_if_set(message, "description", default=None)
        latest_version = get_field_if_set(message, "latest_version", default=None)
        if latest_version:
            latest_version = ArtifactVersion.from_proto(latest_version)
        return cls(
            id=uuid.UUID(message.id),
            experiment_id=int(message.experiment_id),
            type=ArtifactType.from_proto(message.type),
            name=message.name,
            fqn=message.fqn,
            artifact_storage_root=message.artifact_storage_root,
            description=description,
            created_by=message.created_by,
            latest_version=latest_version,
            created_at=datetime.datetime.fromisoformat(message.created_at),
            updated_at=datetime.datetime.fromisoformat(message.updated_at),
            run_steps=[rs for rs in message.run_steps],
        )


class BaseArtifactVersionMixin(BaseModel):
    @property
    def fqn(self) -> str:
        raise NotImplementedError()

    @staticmethod
    def get_artifact_fqn_and_version(fqn: str) -> Tuple[str, int]:
        try:
            artifact_fqn, version = fqn.rsplit(":", 1)
        except ValueError:
            raise MlflowException(
                f"Invalid value for fqn: {fqn!r}. Expected format "
                "{type}:{tenant}/{username}/{project}/{model_name}:{version}",
                error_code=INVALID_PARAMETER_VALUE,
            )

        if (
            version == "latest"
        ):  # Temporarily added as convenience, ideally should be managed using tags
            version = -1
        else:
            version = int(version)
        return artifact_fqn, version


class ArtifactVersion(BaseArtifactVersionMixin):
    class Config:
        orm_mode = True
        validate_assignment = True
        use_enum_values = True
        allow_mutation = True
        smart_union = True

    id: uuid.UUID
    artifact_id: uuid.UUID
    artifact_name: str  # from relation
    artifact_fqn: str  # from relation
    experiment_id: int  # from relation
    version: int
    artifact_storage_root: str
    artifact_metadata: Dict[str, Any] = Field(default_factory=dict)
    data_path: Optional[str] = None
    description: Optional[constr(max_length=1024)] = None
    status: ArtifactVersionStatus
    step: Optional[int] = None
    run_id: Optional[str] = None  # also stored in events
    created_by: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    internal_metadata: Optional[Dict[str, Any]] = None
    artifact_size: Optional[int] = None

    @property
    def fqn(self) -> str:
        return _ARTIFACT_VERSION_FQN_FORMAT.format(
            artifact_fqn=self.artifact_fqn, version=self.version
        )

    @property
    def _usage_code_snippet(self) -> str:
        return _ARTIFACT_VERSION_USAGE_CODE_SNIPPET.format(fqn=self.fqn)

    def to_proto(self) -> mlfa_pb2.ArtifactVersion:
        return mlfa_pb2.ArtifactVersion(
            id=str(self.id),
            artifact_id=str(self.artifact_id),
            artifact_name=self.artifact_name,
            artifact_fqn=self.artifact_fqn,
            experiment_id=str(self.experiment_id),
            version=self.version,
            fqn=self.fqn,
            artifact_storage_root=self.artifact_storage_root,
            artifact_metadata=utils.dict_to_proto(self.artifact_metadata),
            data_path=self.data_path,
            description=self.description or "",
            status=ArtifactVersionStatus(self.status).to_proto(),
            step=self.step,
            usage_code_snippet=self._usage_code_snippet,
            run_id=self.run_id,
            created_by=self.created_by,
            created_at=self.created_at.astimezone(tz=datetime.timezone.utc).isoformat(),
            updated_at=self.updated_at.astimezone(tz=datetime.timezone.utc).isoformat(),
            internal_metadata=utils.dict_to_proto(self.internal_metadata or {}),
            artifact_size=self.artifact_size,
        )

    @classmethod
    def from_proto(cls, message: mlfa_pb2.ArtifactVersion) -> "ArtifactVersion":
        artifact_metadata = get_field_if_set(message, "artifact_metadata", default={})
        if artifact_metadata:
            artifact_metadata = utils.dict_from_proto(artifact_metadata)
        internal_metadata = get_field_if_set(message, "internal_metadata", default={})
        if internal_metadata:
            internal_metadata = utils.dict_from_proto(internal_metadata)
        data_path = get_field_if_set(message, "data_path", default=None)
        description = get_field_if_set(message, "description", default=None)
        step = get_field_if_set(message, "step", default=0)
        run_id = get_field_if_set(message, "run_id", default=None)
        return cls(
            id=uuid.UUID(message.id),
            artifact_id=uuid.UUID(message.artifact_id),
            artifact_name=message.artifact_name,
            artifact_fqn=message.artifact_fqn,
            experiment_id=int(message.experiment_id),
            version=message.version,
            artifact_storage_root=message.artifact_storage_root,
            artifact_metadata=artifact_metadata,
            data_path=data_path,
            description=description,
            status=ArtifactVersionStatus.from_proto(message.status),
            step=step,
            run_id=run_id,
            created_by=message.created_by,
            created_at=datetime.datetime.fromisoformat(message.created_at),
            updated_at=datetime.datetime.fromisoformat(message.updated_at),
            internal_metadata=internal_metadata,
            artifact_size=message.artifact_size,
        )


Artifact.update_forward_refs()
