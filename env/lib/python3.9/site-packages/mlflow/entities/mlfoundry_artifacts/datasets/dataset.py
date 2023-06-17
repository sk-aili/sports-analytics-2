import datetime
import posixpath
import uuid
from typing import Any, Dict, Optional

from pydantic import BaseModel, constr

from mlflow.entities.experiment import Experiment
from mlflow.entities.mlfoundry_artifacts import utils
from mlflow.protos import mlfoundry_artifacts_pb2 as mlfa_pb2
from mlflow.utils.proto_json_utils import get_field_if_set
from mlflow.utils.uri import append_to_uri_path

_DATASET_FQN_FORMAT = "dataset:{tenant_name}/{experiment_name}/{artifact_name}"

_DATASET_USAGE_CODE_SNIPPET = """import mlfoundry as mlf
client = mlf.get_client()

# Get the dataset directly
dataset = client.get_dataset("{fqn}")

# download it to disk
dataset.download(path="your/download/location")"""


class Dataset(BaseModel):
    @staticmethod
    def generate_fqn(experiment: Experiment, name: str) -> str:
        if not experiment.tenant_name:
            raise ValueError(f"Attributes `tenant_name` on `experiment` cannot be None")
        return _DATASET_FQN_FORMAT.format(
            tenant_name=experiment.tenant_name,
            experiment_name=experiment.name,
            artifact_name=name,
        )

    @staticmethod
    def generate_storage_root(experiment: Experiment, name: str) -> str:
        # noinspection PyTypeChecker
        return append_to_uri_path(
            experiment.artifact_location, "artifacts", "datasets", name, posixpath.sep
        )

    class Config:
        orm_mode = True
        validate_assignment = True
        use_enum_values = True
        allow_mutation = False
        smart_union = True
        arbitrary_types_allowed = True  # added because experiment is not a pydantic model

    id: uuid.UUID
    experiment_id: int
    experiment: Optional[Experiment] = None
    name: constr(regex=r"^[A-Za-z0-9_\-]+$", max_length=256)
    fqn: str
    description: Optional[constr(max_length=1024)] = None
    storage_root: str
    dataset_metadata: Optional[Dict[str, Any]] = None
    created_by: constr(max_length=256)
    created_at: datetime.datetime
    updated_at: datetime.datetime

    @property
    def _usage_code_snippet(self) -> str:
        return _DATASET_USAGE_CODE_SNIPPET.format(fqn=self.fqn)

    def to_proto(self) -> mlfa_pb2.Dataset:
        message = mlfa_pb2.Dataset(
            id=str(self.id),
            experiment_id=str(self.experiment_id),
            name=self.name,
            fqn=self.fqn,
            description=self.description,
            storage_root=self.storage_root,
            created_by=self.created_by,
            created_at=self.created_at.isoformat(),
            updated_at=self.updated_at.isoformat(),
            dataset_metadata=utils.dict_to_proto(self.dataset_metadata),
            usage_code_snippet=self._usage_code_snippet,
        )
        return message

    @classmethod
    def from_proto(cls, message: mlfa_pb2.Dataset) -> "Dataset":
        description = get_field_if_set(message, "description", default=None)
        dataset_metadata = get_field_if_set(message, "dataset_metadata", default={})
        if dataset_metadata:
            dataset_metadata = utils.dict_from_proto(dataset_metadata)
        return cls(
            id=uuid.UUID(message.id),
            experiment_id=int(message.experiment_id),
            name=message.name,
            fqn=message.fqn,
            storage_root=message.storage_root,
            description=description,
            created_by=message.created_by,
            created_at=datetime.datetime.fromisoformat(message.created_at),
            updated_at=datetime.datetime.fromisoformat(message.updated_at),
            dataset_metadata=dataset_metadata,
        )
